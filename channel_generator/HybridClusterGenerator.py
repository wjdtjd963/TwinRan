from sionna.phy.channel.tr38901.rays import RaysGenerator
from .hybrid_cluster.preprocessor import HybridClusterPreprocessor
from .hybrid_cluster.ray_builder import HybridClusterRayBuilder
from .hybrid_cluster.synthesizer import HybridChannelSynthesizer
from .hybrid_cluster.exceptions import ValidationError, ConfigurationError, HybridClusterError

import tensorflow as tf
import numpy as np
from sionna.phy.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.channel.tr38901 import LSPGenerator
import pickle
import os


class HybridClusterGenerator(RaysGenerator):
    """
    Main orchestrator for hybrid cluster generation (TR38901 Steps 4-13)
    
    This class coordinates the entire pipeline by combining:
    - HybridClusterPreprocessor (Steps 4-8): Cluster generation
    - HybridClusterRayBuilder (Steps 9-12): Ray generation  
    - HybridChannelSynthesizer (Step 13): Channel synthesis
    """

    def __init__(self, scenario):
        super().__init__(scenario)
        self._scenario = scenario
        self._lsp_generator = None
        
        # Initialize modular components
        self.preprocessor = HybridClusterPreprocessor(scenario)
        self.ray_builder = HybridClusterRayBuilder(scenario)
        self.synthesizer = HybridChannelSynthesizer(scenario)

    def _get_lsp_generator(self):
        """
        Lazily create and cache the LSPGenerator based on the stored scenario.
        """
        if self._lsp_generator is None:
            self._lsp_generator = LSPGenerator(self._scenario)
        return self._lsp_generator

    @staticmethod
    def inspect_user_rt_metadata(user):
        """
        Inspect user's PKL to extract basic metadata for scenario setup.
        Returns (rx_info, num_tx, num_rx).
        """
        if user is None or getattr(user, 'pkl', None) is None:
            raise ValidationError("user or user.pkl is not provided")
        file_path = user.pkl
        if not os.path.isfile(file_path):
            raise ValidationError(f"PKL file not found: {file_path}")
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            rt_params_raw = loaded
            rx_info = None
        elif isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
            rt_params_raw = loaded[0]
            rx_info = loaded[1] if len(loaded) > 1 else None
        else:
            raise ValidationError("Unsupported PKL format: expected dict or [dict, rx_info]")
        a_re = np.array(rt_params_raw['a_re'])
        num_rx = a_re.shape[0]
        num_tx = a_re.shape[2]
        return rx_info, int(num_tx), int(num_rx)

    def _load_rt_from_user(self, user):
        """
        Load ray-tracing raw data from user's PKL path.
        Accepts PKL formats:
          - dict of rt params
          - list/tuple: [rt_params_raw, rx_info]
        Returns (rt_params_raw, rx_info_or_None)
        """
        if user is None or getattr(user, 'pkl', None) is None:
            raise ValidationError("user or user.pkl is not provided")
        file_path = user.pkl
        if not os.path.isfile(file_path):
            raise ValidationError(f"PKL file not found: {file_path}")
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            return loaded, None
        if isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
            rt_params_raw = loaded[0]
            rx_info = loaded[1] if len(loaded) > 1 else None
            return rt_params_raw, rx_info
        raise ValidationError("Unsupported PKL format: expected dict or [dict, rx_info]")

    def _preprocess_rt(self, rt_params_raw, drop_db: float = 25.0):
        """
        Convert raw RT params to numpy arrays with batch dimension.
        Apply power-based path filtering (<= Pmax/10^(drop_db/10) is dropped),
        set tau=-1.0 for dropped paths and other params to 0, then sort by tau and
        normalize tau per (UT, BS) so min valid tau becomes 0.

        Also compute los_flag_np similarly to the given reference (based on
        interactions and valid if available).

        Returns (rt_params_np, los_flag_np)
        """
        if not isinstance(rt_params_raw, dict):
            raise ValidationError("rt_params_raw must be a dict")

        # Required keys
        required_keys = ['a_re', 'a_im', 'tau', 'phi_r', 'phi_t', 'theta_r', 'theta_t']
        for k in required_keys:
            if k not in rt_params_raw:
                raise ValidationError(f"Missing key in rt_params_raw: {k}")

        # Base arrays from raw (fixed, standardized order)
        # a_re/a_im: [UT, rx_ant, BS, tx_ant, P]
        # tau/phi/theta: [UT, BS, P]
        a_re_raw = np.array(rt_params_raw['a_re'])
        a_im_raw = np.array(rt_params_raw['a_im'])
        tau_raw = np.array(rt_params_raw['tau'])
        phi_r_raw = np.array(rt_params_raw['phi_r'])
        phi_t_raw = np.array(rt_params_raw['phi_t'])
        theta_r_raw = np.array(rt_params_raw['theta_r'])
        theta_t_raw = np.array(rt_params_raw['theta_t'])

        # Validate shapes match the standardized order
        if a_re_raw.ndim != 5 or a_im_raw.ndim != 5:
            raise ValidationError(f"a_re/a_im must be 5D. Got a_re={a_re_raw.shape}, a_im={a_im_raw.shape}")
        if tau_raw.ndim != 3 or phi_r_raw.ndim != 3 or phi_t_raw.ndim != 3 or theta_r_raw.ndim != 3 or theta_t_raw.ndim != 3:
            raise ValidationError("tau/phi/theta must be 3D [UT, BS, P]")

        num_ut, num_rx_ant, num_bs, num_tx_ant, num_path = a_re_raw.shape

        # Power per path aggregated over antenna dims → [UT, BS, P]
        power_sum = (a_re_raw ** 2 + a_im_raw ** 2).sum(axis=(1, 3))

        # # DEBUG: Print power_sum information
        # print(f"DEBUG: power_sum size = {power_sum.size}")
        # print(f"DEBUG: power_sum min = {power_sum.min() if power_sum.size > 0 else 'empty'}")
        # print(f"DEBUG: power_sum max = {power_sum.max() if power_sum.size > 0 else 'empty'}")

        # Check if power_sum is empty or has zero-size dimensions
        if power_sum.size == 0:
            # DEBUG: Check if user pkl attribute is None when power_sum is empty
            user_pkl = getattr(self._current_user, 'pkl', None) if hasattr(self, '_current_user') else None
            if user_pkl is None:
                print(f"DEBUG: power_sum is empty and user pkl attribute is None")
            else:
                print(f"DEBUG: power_sum is empty but user pkl attribute is not None: {user_pkl}")
            
            # Return None to indicate empty power_sum
            print(f"DEBUG: Returning None due to empty power_sum")
            return None, None
        
        # Threshold per (UT, BS)
        pmax = power_sum.max(axis=-1, keepdims=True)
        threshold = pmax / (10.0 ** (drop_db / 10.0))
        keep_mask = power_sum >= threshold  # [UT, BS, P]

        # Determine Kmax (max kept across all UT, BS)
        keep_counts = keep_mask.sum(axis=-1)
        Kmax = int(keep_counts.max()) if keep_counts.size > 0 else 0
        Kmax = max(Kmax, 0)

        # Pre-allocate filtered arrays without batch
        tau_f = np.full((num_ut, num_bs, Kmax), -1.0)
        phi_t_f = np.zeros((num_ut, num_bs, Kmax))
        phi_r_f = np.zeros((num_ut, num_bs, Kmax))
        theta_t_f = np.zeros((num_ut, num_bs, Kmax))
        theta_r_f = np.zeros((num_ut, num_bs, Kmax))
        a_re_f = np.zeros((num_ut, num_rx_ant, num_bs, num_tx_ant, Kmax))
        a_im_f = np.zeros((num_ut, num_rx_ant, num_bs, num_tx_ant, Kmax))

        # pad_mask True means padding (invalid), False means valid
        pad_mask = np.ones((num_ut, num_bs, Kmax), dtype=bool)

        for r in range(num_ut):
            for t in range(num_bs):
                if Kmax == 0:
                    continue
                valid_indices = np.nonzero(keep_mask[r, t, :])[0]
                k = len(valid_indices)
                if k > 0:
                    k = min(k, Kmax)
                    pad_mask[r, t, :k] = False
                    sel = valid_indices[:k]
                    tau_f[r, t, :k] = tau_raw[r, t, sel]
                    phi_t_f[r, t, :k] = phi_t_raw[r, t, sel]
                    phi_r_f[r, t, :k] = phi_r_raw[r, t, sel]
                    theta_t_f[r, t, :k] = theta_t_raw[r, t, sel]
                    theta_r_f[r, t, :k] = theta_r_raw[r, t, sel]
                    # Advanced indexing moves the indexed axis to the front: (k, rx_ant, tx_ant)
                    # Transpose to (rx_ant, tx_ant, k) before assignment
                    tmp_re = a_re_raw[r, :, t, :, sel].transpose(1, 2, 0)
                    tmp_im = a_im_raw[r, :, t, :, sel].transpose(1, 2, 0)
                    a_re_f[r, :, t, :, :k] = tmp_re
                    a_im_f[r, :, t, :, :k] = tmp_im

        # Sort by tau ascending per (UT, BS), padding to end
        tau_key = np.where(pad_mask, np.inf, tau_f)
        sort_indices = np.argsort(tau_key, axis=-1, kind='stable')  # [UT, BS, Kmax]

        def sort_3d(arr):
            return np.take_along_axis(arr, sort_indices, axis=-1)

        tau_sorted = sort_3d(tau_f)
        phi_t_sorted = sort_3d(phi_t_f)
        phi_r_sorted = sort_3d(phi_r_f)
        theta_t_sorted = sort_3d(theta_t_f)
        theta_r_sorted = sort_3d(theta_r_f)
        pad_mask_sorted = sort_3d(pad_mask)

        # Sort 5D amplitude arrays
        a_re_sorted = np.zeros_like(a_re_f)
        a_im_sorted = np.zeros_like(a_im_f)
        for r in range(num_ut):
            for t in range(num_bs):
                order = sort_indices[r, t, :]
                a_re_sorted[r, :, t, :, :] = a_re_f[r, :, t, :, :][:, :, order]
                a_im_sorted[r, :, t, :, :] = a_im_f[r, :, t, :, :][:, :, order]

        # Tau normalization: min valid tau → 0, padding stays -1
        tau_valid = np.where(pad_mask_sorted, np.nan, tau_sorted)
        with np.errstate(all='ignore'):
            tau_min = np.nanmin(tau_valid, axis=-1, keepdims=True)
        tau_norm = np.where(pad_mask_sorted, -1.0, tau_sorted - tau_min)

        # Compute los_flag_np: require interactions and valid
        los_flag = None
        if ('interactions' in rt_params_raw) and ('valid' in rt_params_raw):
            interactions = np.array(rt_params_raw['interactions'])
            valid = np.array(rt_params_raw['valid'])  # [UT, BS, P]
            try:
                from sionna.rt.constants import InteractionType
                none_val = int(InteractionType.NONE)
                # Assume interactions shape [..., UT, BS, P], reduce over bounce axis 0
                is_interaction_free = np.all(interactions == none_val, axis=0)
            except Exception:
                # Fallback: if interactions already [UT, BS, P]
                is_interaction_free = (interactions == 0)
            path_is_los = is_interaction_free & valid                              # [UT, BS, P]
            # LoS must survive power filtering; otherwise, treat as NLoS
            path_is_los_kept = path_is_los & keep_mask                             # [UT, BS, P]
            link_has_los = (np.sum(path_is_los_kept, axis=2) > 0)[..., np.newaxis]  # [UT, BS, 1]
            los_flag = link_has_los.astype(bool)
        else:
            raise ValidationError("Missing required keys for LoS computation: 'interactions' and 'valid'")

        # Assemble batched rt_params_np with expected shapes
        rt_params_np = {
            'a_re': np.expand_dims(a_re_sorted, axis=0),                    # [B, UT, rx_ant, BS, tx_ant, Kmax]
            'a_im': np.expand_dims(a_im_sorted, axis=0),
            'tau': np.expand_dims(tau_norm, axis=0),                        # [B, UT, BS, Kmax]
            'phi_r': np.expand_dims(phi_r_sorted, axis=0),
            'phi_t': np.expand_dims(phi_t_sorted, axis=0),
            'theta_r': np.expand_dims(theta_r_sorted, axis=0),
            'theta_t': np.expand_dims(theta_t_sorted, axis=0),
        }

        los_flag_np = np.expand_dims(los_flag, axis=0)  # [B, UT, BS, 1]
        # Squeeze last dim to match expected [B, UT, BS]
        los_flag_np = np.squeeze(los_flag_np, axis=-1)

        return rt_params_np, los_flag_np

    def prepare_from_user(self, user):
        """
        Public helper to load & preprocess RT inputs from a User instance.
        Returns (rt_params_np, los_flag_np).
        """
        rt_params_raw, _ = self._load_rt_from_user(user)
        return self._preprocess_rt(rt_params_raw)

    def __call__(self, user, sample_times=None, normalize: bool = False,
                 num_subcarriers: int = 2048, subcarrier_spacing: float = 15000.0):
        """
        Main orchestration function for hybrid cluster generation.
        By default, generates a single snapshot OFDM channel at t=0.
        
        Parameters
        ----------
        user : User
            Scenario generator User instance with .pkl path
        sample_times : tf.Tensor, optional
            Time samples for channel synthesis. If not provided, defaults to a single
            snapshot at t=0, i.e., tf.range(1, dtype=tf.float32). Shape: [T].
        normalize : bool, optional
            Channel normalization flag (default: False).
        num_subcarriers : int, optional
            Number of OFDM subcarriers (default: 2048).
        subcarrier_spacing : float, optional
            Subcarrier spacing [Hz] (default: 15000).
            
        Returns
        -------
        tuple
            h_freq : tf.Tensor
                OFDM channel frequency response.
                Shape: [B, UT, rx_ant, BS, tx_ant, T, num_subcarriers]
            rt_connected_bs : tf.Tensor
                Boolean tensor indicating if a link is dominated by RT paths.
                Shape: [B, UT, BS]
        """
        # Input validation
        if user is None:
            raise ValidationError("user cannot be None")
        
        if sample_times is None:
            # Default to a single snapshot at t=0
            sample_times = tf.range(1, dtype=tf.float32)

        try:
            # Store current user for debugging
            self._current_user = user
            
            # Load + preprocess from user
            rt_params_np, los_flag_np = self.prepare_from_user(user)
            # print("✓ Step 1: Load and preprocess RT data completed")
            
            # Check if rt_params_np is None (power_sum was empty)
            if rt_params_np is None:
                print("DEBUG: Returning zero channel due to empty power_sum")
                # Return zero channel with appropriate shape
                batch_size = 1
                num_ut = 1
                num_rx_ant = 2  # Based on ut_array configuration
                num_bs = len(self._scenario.bs_loc[0]) if hasattr(self._scenario, 'bs_loc') else 1
                num_tx_ant = 1  # Based on bs_array configuration
                num_ofdm_symbols = len(sample_times)
                
                h_freq_zero = tf.zeros([batch_size, num_ut, num_rx_ant, num_bs, num_tx_ant, 
                                       num_ofdm_symbols, num_subcarriers], dtype=tf.complex64)
                rt_connected_bs_zero = tf.zeros([batch_size, num_ut, num_bs], dtype=tf.bool)
                return h_freq_zero, rt_connected_bs_zero

            # Prepare internal LSP generator
            lsp_generator = self._get_lsp_generator()
            # print("✓ Step 2: LSP generator prepared")
            
            # Step 4-8: Generate clusters using preprocessor
            clusters, rt_connected_bs = self.preprocessor(lsp_generator, rt_params_np, los_flag_np)
            # print("✓ Step 3: Cluster generation (Steps 4-8) completed")
            
            # Get LSP for ray building
            lsp = lsp_generator()
            # print("✓ Step 4: LSP generation completed")
            
            # Convert los_flag_np to tensor for ray builder
            # The shape needs to be [B, BS, UT] to match other tensors
            los_mask_tensor = tf.constant(los_flag_np, dtype=tf.bool)
            los_mask_tensor = tf.transpose(los_mask_tensor, [0, 2, 1]) 
            # print("✓ Step 5: LoS mask tensor preparation completed")

            # Step 9-12: Generate rays using ray builder
            rays, ray_powers, kappa, phases = self.ray_builder(clusters, lsp, los_mask_tensor)
            # print("✓ Step 6: Ray generation (Steps 9-12) completed")
            
            # Step 13: Generate final channel coefficients
            h_final, delays_final = self.synthesizer(
                rays, ray_powers, kappa, phases,
                sample_times
            )
            # print("✓ Step 7: Channel synthesis (Step 13) completed")

            # Setup OFDM parameters
            self.setup_ofdm_parameters(num_subcarriers, subcarrier_spacing)
            # print("✓ Step 8: OFDM parameters setup completed")
            
            # Transform input format
            a, tau = self.permute_hybrid_outputs(h_final, delays_final)
            # print("✓ Step 9: Output format transformation completed")
            
            # Generate OFDM channel response
            h_freq = cir_to_ofdm_channel(
                frequencies=self.frequencies,
                a=a,
                tau=tau,
                normalize=normalize
            )
            # print("✓ Step 10: OFDM channel response generation completed")
            
            return h_freq, rt_connected_bs
            
        except Exception as e:
            if isinstance(e, HybridClusterError):
                raise
            else:
                raise HybridClusterError(f"Unexpected error in hybrid cluster generation: {str(e)}") from e

    def generate_clusters_only(self, rt_params_np, los_flag_np):
        """
        Generate only clusters (Steps 4-8)
        
        Returns
        -------
        tuple
            (clusters, rt_connected_bs)
        """
        lsp_generator = self._get_lsp_generator()
        return self.preprocessor(lsp_generator, rt_params_np, los_flag_np)

    def generate_rays_only(self, clusters, los_flag_np):
        """
        Generate only rays (Steps 9-12)
        
        Parameters
        ----------
        clusters : dict
            Cluster parameters from preprocessor.
        los_flag_np : np.ndarray
            LoS/NLoS flags [B, UT, BS].

        Returns
        -------
        tuple
            (rays, ray_powers, kappa, phases)
        """
        lsp = self._get_lsp_generator()()
        los_mask_tensor = tf.constant(los_flag_np, dtype=tf.bool)
        los_mask_tensor = tf.transpose(los_mask_tensor, [0, 2, 1])
        return self.ray_builder(clusters, lsp, los_mask_tensor)

    def synthesize_channel_only(self, rays, ray_powers, kappa, phases, 
                               sample_times):
        """
        Synthesize channel only (Step 13)
        
        Returns
        -------
        tuple
            (h_final, delays_final)
        """
        return self.synthesizer(rays, ray_powers, kappa, phases,
                               sample_times)

    # =============================================================================
    # OFDM Channel Generation Methods
    # =============================================================================

    def setup_ofdm_parameters(self, num_subcarriers: int = 2048, subcarrier_spacing: float = 15000.0):
        """
        Setup OFDM parameters
        
        Parameters
        ----------
        num_subcarriers : int, optional
            Number of OFDM subcarriers (default: 2048)
        subcarrier_spacing : float, optional
            Subcarrier spacing [Hz] (default: 15000)
        """
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        
        # Calculate subcarrier frequencies
        self.frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

    def permute_hybrid_outputs(self, h_final: tf.Tensor, delays_final: tf.Tensor):
        """
        Transform HybridClusterGenerator outputs to cir_to_ofdm_channel input format
        
        Parameters
        ----------
        h_final : tf.Tensor
            HybridClusterGenerator channel coefficients [B, BS, UT, N*M, rx_ant, tx_ant, T]
        delays_final : tf.Tensor
            HybridClusterGenerator delay times [B, BS, UT, N*M, rx_ant, tx_ant]
            
        Returns
        -------
        tuple
            - a: Path coefficients [B, UT, rx_ant, BS, tx_ant, N*M, T]
            - tau: Path delays [B, UT, rx_ant, BS, tx_ant, N*M]
        """
        # h_final: [B, BS, UT, N*M, rx_ant, tx_ant, T]
        # BS = Tx, UT = Rx
        # → [B, UT, rx_ant, BS, tx_ant, N*M, T]
        a = tf.transpose(h_final, [0, 2, 4, 1, 5, 3, 6])
        
        # delays_final: [B, BS, UT, N*M, rx_ant, tx_ant]
        # → [B, UT, rx_ant, BS, tx_ant, N*M]
        tau = tf.transpose(delays_final, [0, 2, 4, 1, 5, 3])
        
        return a, tau

    def create_ofdm_channel_from_outputs(self, h_final: tf.Tensor, 
                                        delays_final: tf.Tensor,
                                        num_subcarriers: int = 2048,
                                        subcarrier_spacing: float = 15000.0,
                                        normalize: bool = False):
        """
        Generate OFDM channel response from HybridClusterGenerator outputs
        
        Parameters
        ----------
        h_final : tf.Tensor
            HybridClusterGenerator channel coefficients [B, BS, UT, N*M, rx_ant, tx_ant, T]
        delays_final : tf.Tensor
            HybridClusterGenerator delay times [B, BS, UT, N*M, rx_ant, tx_ant]
        num_subcarriers : int, optional
            Number of OFDM subcarriers (default: 2048)
        subcarrier_spacing : float, optional
            Subcarrier spacing [Hz] (default: 15000)
        normalize : bool, optional
            Channel normalization flag (default: False)
            
        Returns
        -------
        tf.Tensor
            OFDM channel response [B, UT, rx_ant, BS, tx_ant, T, num_subcarriers]
        """
        # 1. Generate subcarrier frequencies
        frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
        
        # 2. Transform input format
        a, tau = self.permute_hybrid_outputs(h_final, delays_final)
        
        # 3. Generate OFDM channel response
        h_freq = cir_to_ofdm_channel(
            frequencies=frequencies,
            a=a,
            tau=tau,
            normalize=normalize
        )
        
        return h_freq
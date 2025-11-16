# hybrid_cluster/preprocessor.py
from sionna.phy.channel.tr38901.rays import RaysGenerator
from .constants import (
    POWER_THRESHOLD_DB, INVALID_DELAY, RAD_TO_DEG
)
from .exceptions import RayTracingError, ValidationError, ParameterError
from sionna.phy import config
from sionna.phy.utils import log10
from sionna.phy.channel.utils import deg_2_rad, rad_2_deg, wrap_angle_0_360
import tensorflow as tf
import numpy as np
import sys

class HybridClusterPreprocessor(RaysGenerator):
    """
    TR 38.901 §8.4 Steps 4-8: RT+RC cluster generation
    
    This class handles the preprocessing steps for hybrid cluster generation:
    - Step 4: Ray-tracing input preparation
    - Step 5: RC delay generation (TR38901 8.4-1 to 8.4-3)
    - Step 6: RC power generation (TR38901 8.4-4 to 8.4-8)
    - Step 7: RC angle generation (TR38901 8.4-9 to 8.4-18)
    - Step 8: RT+RC cluster combination
    
    Parameters
    ----------
    scenario : Scenario
        Sionna scenario object containing topology and parameters
    """
    
    def __init__(self, scenario):
        """
        Initialize the preprocessor with scenario configuration.
        
        Parameters
        ----------
        scenario : Scenario
            Sionna scenario object containing topology and parameters
        """
        super().__init__(scenario)
        self._scenario = scenario

    def _update_scenario_los(self, los_flag_np):
        """
        Update the scenario's LoS status and dependent properties.
        
        Parameters
        ----------
        los_flag_np : np.ndarray
            LoS/NLoS flags array [B, UT, BS]
            
        Notes
        -----
        This method updates the scenario's internal LoS status and recomputes
        dependent properties like LSP parameters and pathloss.
        """
        los_flag_transposed = np.transpose(los_flag_np, (0, 2, 1))
        los_flag_tf = tf.convert_to_tensor(los_flag_transposed, dtype=tf.bool)

        indoor_mask = tf.expand_dims(self._scenario.indoor, axis=1)
        is_outdoor = tf.logical_not(indoor_mask)
        self._scenario._los = tf.logical_and(los_flag_tf, is_outdoor)

        self._scenario._compute_lsp_log_mean_std()
        self._scenario._compute_pathloss_basic()
        
        if hasattr(self, 'lsp_generator') and self.lsp_generator is not None:
             self.lsp_generator.topology_updated_callback()

    def _prepare_rt_inputs(self, rt_params_np, valid_paths_mask_np):
        """
        Prepare ray-tracing inputs for TensorFlow processing.
        
        Parameters
        ----------
        rt_params_np : dict[str, np.ndarray]
            Ray-tracing parameters dictionary containing:
            - 'tau': delay times
            - 'phi_r', 'phi_t': azimuth angles
            - 'theta_r', 'theta_t': zenith angles
            - 'a_re', 'a_im': complex amplitudes
        valid_paths_mask_np : np.ndarray
            Boolean mask indicating valid ray-tracing paths
            
        Returns
        -------
        dict[str, tf.Tensor]
            Processed ray-tracing parameters ready for TensorFlow operations
            
        Raises
        ------
        ValidationError
            If required keys are missing from rt_params_np
        """
        # Input validation
        if not isinstance(rt_params_np, dict):
            raise ValidationError("rt_params_np must be a dictionary")
        
        required_keys = ['tau', 'phi_r', 'phi_t', 'theta_r', 'theta_t', 'a_re', 'a_im']
        missing_keys = [key for key in required_keys if key not in rt_params_np]
        if missing_keys:
            raise ValidationError(f"Missing required keys in rt_params_np: {missing_keys}")
        
        mask = tf.convert_to_tensor(valid_paths_mask_np, dtype=tf.bool)

        rt_params_masked = {}
        for key, value in rt_params_np.items():
            tensor = tf.convert_to_tensor(value, dtype=self.rdtype)
            if key in ['a_re', 'a_im']:
                mask_expanded = mask[:, :, tf.newaxis, :, tf.newaxis, :]
                rt_params_masked[key] = tf.where(mask_expanded, tensor, 0.0)
            elif key == 'tau':
                rt_params_masked[key] = tf.where(mask, tensor, INVALID_DELAY)
            else:
                rt_params_masked[key] = tf.where(mask, tensor, 0.0)

        a_re = rt_params_masked['a_re']
        a_im = rt_params_masked['a_im']
        rt_powers = tf.reduce_sum(a_re**2 + a_im**2, axis=(2, 4))

        rt_params_tf = {'powers': rt_powers}
        for key, value in rt_params_masked.items():
            if key not in ['a_re', 'a_im']:
                rt_params_tf[key] = value

        for key, tensor in rt_params_tf.items():
            perm = [0, 2, 1] + list(range(3, tensor.ndim))
            tensor = tf.transpose(tensor, perm)
            rt_params_tf[key] = tensor        
            
        return rt_params_tf

    def _generate_rc_delays(self, lsp, rt_delays):
        """
        Step 5: Generate RC delays (TR38901 8.4-1 to 8.4-3)
        
        Parameters
        ----------
        lsp : LSP
            Large Scale Parameters object containing delay spread
        rt_delays : tf.Tensor
            Ray-tracing delay times [B, BS, UT, P]
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            - tau_RC: RC delay times [B, BS, UT, N]
            - tau_RC_prime: RC delay times before correction [B, BS, UT, N]
            
        Notes
        -----
        This method implements TR 38.901 equations 8.4-1 to 8.4-3 for
        generating RC (Random Cluster) delays based on RT delays and K-factor.
        """
        delay_spread = lsp.ds

        r_tau = tf.reshape(self._scenario.get_param("rTau"),
                       [-1, self._scenario.num_bs, self._scenario.num_ut])
        
        valid_mask = rt_delays > INVALID_DELAY        
        L_RT = tf.reduce_sum(tf.cast(valid_mask, self.rdtype), axis=-1)
        
        valid_delays = tf.where(valid_mask, rt_delays, 0.0)
        sum_of_valid_delays = tf.reduce_sum(valid_delays, axis=-1)
        avg_tau_RT = tf.math.divide_no_nan(sum_of_valid_delays, L_RT)

        L_RC = tf.cast(self._scenario.num_clusters_nlos, self.rdtype)

        # Important: Multiply r_tau by delay_spread for unit consistency
        mu_t_corrected = r_tau * delay_spread + (L_RT / (L_RC + 1.0)) * (r_tau * delay_spread - avg_tau_RT)
        mu_tau = tf.maximum(mu_t_corrected, avg_tau_RT)

        mu_tau_expanded = tf.expand_dims(mu_tau, axis=-1)
        num_clusters_to_gen = self._scenario.num_clusters_nlos
        
        if num_clusters_to_gen == 0:
            raise ParameterError("Number of RC clusters is 0 - check scenario configuration")
        
        x = config.tf_rng.uniform(shape=tf.concat([tf.shape(mu_tau), [num_clusters_to_gen]], axis=0),
                                  minval=1e-5, maxval=1.0, dtype=self.rdtype)
        # 8.4-1
        tau_RC_prime = -mu_tau_expanded * tf.math.log(x)
        # 8.4-3
        k_db = 10.0 * log10(lsp.k_factor)
        c_tau = (0.7705 - 0.0433 * k_db + 
                0.0002 * tf.square(k_db) + 0.000017 * tf.pow(k_db, 3.))
        c_tau = tf.where(self._scenario.los, c_tau, tf.ones_like(c_tau))
        tau_RC = tau_RC_prime - tf.reduce_min(tau_RC_prime, axis=-1, keepdims=True)
        # 8.4-2
        tau_RC = tau_RC / tf.expand_dims(c_tau, axis=-1)
        tau_RC = tf.sort(tau_RC, axis=-1)
        return tau_RC, tau_RC_prime

    def _generate_rc_powers(self, lsp, rt_delays, rt_powers, tau_RC_prime):
        """
        Step 6: Generate RC powers (TR38901 8.4-4 to 8.4-8)
        
        Parameters
        ----------
        lsp : LSP
            Large Scale Parameters object containing delay spread and K-factor
        rt_delays : tf.Tensor
            Ray-tracing delay times [B, BS, UT, P]
        rt_powers : tf.Tensor
            Ray-tracing powers [B, BS, UT, P]
        tau_RC_prime : tf.Tensor
            RC delay times before correction [B, BS, UT, N]
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            - rc_powers: RC powers [B, BS, UT, N]
            - rt_powers_scaled: Scaled RT powers [B, BS, UT, P]
            - los_power_boost: LoS power boost factor [B, BS, UT, 1]
            
        Notes
        -----
        This method implements TR 38.901 equations 8.4-4 to 8.4-8 for
        generating RC powers and scaling RT powers based on K-factor.
        """
        valid_mask = rt_delays > INVALID_DELAY
        r_tau = tf.reshape(self._scenario.get_param("rTau"),
                       [-1, self._scenario.num_bs, self._scenario.num_ut, 1])
        sigma_sf = tf.reshape(self._scenario.get_param("zeta"),
                              [-1, self._scenario.num_bs, self._scenario.num_ut, 1])
        delay_spread = tf.expand_dims(lsp.ds, axis=-1)
        k_factor = tf.expand_dims(lsp.k_factor, axis=-1)
        
        Z_RT = config.tf_rng.normal(shape=tf.shape(rt_delays), stddev=sigma_sf, dtype=self.rdtype)
        Z_RC = config.tf_rng.normal(shape=tf.shape(tau_RC_prime), stddev=sigma_sf, dtype=self.rdtype)
        # 8.4-5
        V_RT = tf.exp(-rt_delays * (r_tau - 1.0) / (r_tau * delay_spread)) * tf.pow(10.0, -Z_RT / 10.0)
        V_RT = tf.where(valid_mask, V_RT, 0.0)

        # 8.4-4
        V_RC = tf.exp(-tau_RC_prime * (r_tau - 1.0) / (r_tau * delay_spread)) * tf.pow(10.0, -Z_RC / 10.0)

        A = tf.where(tf.expand_dims(self._scenario.los, axis=-1), k_factor, tf.zeros_like(k_factor))

        total_V_RC = tf.reduce_sum(V_RC, axis=-1, keepdims=True)
        total_V_RT = tf.reduce_sum(V_RT, axis=-1, keepdims=True)
        # 8.4-6
        denom = (A + 1.0) * (total_V_RT + total_V_RC)
        P_RC_virtual = tf.math.divide_no_nan(V_RC , denom)
        
        # 8.4-7
        P_RT_virtual = tf.math.divide_no_nan(V_RT , denom)

        los_power_boost = tf.math.divide_no_nan(A, (A + 1.0))
        
        num_rt_clusters = tf.shape(rt_delays)[-1]
        los_mask = tf.one_hot(indices=0, depth=num_rt_clusters, dtype=self.rdtype)
        los_mask = tf.reshape(los_mask, [1, 1, 1, -1])

        los_boost_term = tf.expand_dims(tf.cast(self._scenario.los, self.rdtype), -1) * los_power_boost * los_mask
        P_RT_virtual += los_boost_term
        # 8.4-8
        total_P_RT_virtual = tf.reduce_sum(P_RT_virtual, axis=-1, keepdims=True)
        
        # Use only valid RT powers for scaling calculation
        valid_rt_powers = tf.where(valid_mask, rt_powers, 0.0)
        total_P_RT_real = tf.reduce_sum(valid_rt_powers, axis=-1, keepdims=True)  
        scaling_factor = tf.math.divide_no_nan(total_P_RT_real, total_P_RT_virtual)
        P_RC_real = scaling_factor * P_RC_virtual
        
        return P_RC_real
        
    def _generate_rc_angles(self, lsp, rt_params, rc_powers):
        """Step 7: Generate cluster angles (TR38901 8.4-9 to 8.4-18)"""
        shape_rc = tf.shape(rc_powers)
        num_rt_clusters = tf.shape(rt_params['powers'])[-1]

        rt_powers = rt_params['powers']
        rt_aoa_rad = rt_params['phi_r']
        rt_aod_rad = rt_params['phi_t']
        rt_zoa_rad = rt_params['theta_r']
        rt_zod_rad = rt_params['theta_t']

        k_factor_db = 10.0 * log10(lsp.k_factor)

        all_powers = tf.concat([rt_powers, rc_powers], axis=-1)
        powers_sum = tf.reduce_sum(all_powers, axis=-1, keepdims=True)
        powers_regularized = tf.math.divide_no_nan(all_powers, powers_sum)
        max_power = tf.reduce_max(powers_regularized, axis=-1, keepdims=True)
        powers_ratio = tf.clip_by_value(tf.math.divide_no_nan(powers_regularized[..., num_rt_clusters:], max_power), 1e-12, 1.0)

        # Azimuth Angles
        c_phi_nlos = self._scenario.get_param("CPhiNLoS")
        # 8.4-11
        c_phi_los = c_phi_nlos * (1.1035 - 0.028 * k_factor_db - 
                                 0.002 * tf.square(k_factor_db) + 0.0001 * tf.pow(k_factor_db, 3.))
        c_phi = tf.where(self._scenario.los, c_phi_los, c_phi_nlos)
        c_phi = tf.expand_dims(c_phi, axis=-1)

        # AOA
        asa_rad = deg_2_rad(lsp.asa)
        # 8.4-10
        phi_prime_aoa = (2. *tf.expand_dims(asa_rad, -1) / 1.4) * tf.sqrt(-tf.math.log(powers_ratio)) / c_phi
        # 8.4-13
        complex_phasors_aoa = tf.complex(rt_powers, 0.) * tf.exp(tf.complex(0.,rt_aoa_rad))
        phi_center_aoa = tf.math.angle(tf.reduce_sum(complex_phasors_aoa, axis=-1, keepdims=True))
        # 8.4-12
        X_n_aoa = tf.cast(tf.random.uniform(shape=shape_rc, minval=0, maxval=2, dtype=tf.int32) * 2 - 1, self.rdtype)
        Y_n_aoa = tf.random.normal(shape = shape_rc, mean = 0.0, stddev = tf.expand_dims(asa_rad, -1)/7.0, dtype=self.rdtype)
        rc_aoa_rad = X_n_aoa * phi_prime_aoa + Y_n_aoa + phi_center_aoa
        rc_aoa_deg = rad_2_deg(rc_aoa_rad)
        rc_aoa_deg_wrapped = wrap_angle_0_360(rc_aoa_deg)
        rc_aoa_deg_wrapped = tf.where(tf.math.greater(rc_aoa_deg_wrapped, 180.), rc_aoa_deg_wrapped-360., rc_aoa_deg_wrapped)

        # AOD
        asd_rad = deg_2_rad(lsp.asd)
        phi_prime_aod = (2. *tf.expand_dims(asd_rad, -1) / 1.4) * tf.sqrt(-tf.math.log(powers_ratio))
        complex_phasors_aod = tf.complex(rt_powers, 0.) * tf.exp(tf.complex(0.,rt_aod_rad))
        phi_center_aod = tf.math.angle(tf.reduce_sum(complex_phasors_aod, axis=-1, keepdims=True))
        X_n_aod = tf.cast(tf.random.uniform(shape=shape_rc, minval=0, maxval=2, dtype=tf.int32) * 2 - 1, self.rdtype)
        Y_n_aod = tf.random.normal(shape = shape_rc, mean = 0.0, stddev = tf.expand_dims(asd_rad, -1)/7.0, dtype=self.rdtype)
        rc_aod_rad = X_n_aod * phi_prime_aod + Y_n_aod + phi_center_aod
        rc_aod_deg = rad_2_deg(rc_aod_rad)
        rc_aod_deg_wrapped = wrap_angle_0_360(rc_aod_deg)
        rc_aod_deg_wrapped = tf.where(tf.math.greater(rc_aod_deg_wrapped, 180.), rc_aod_deg_wrapped-360., rc_aod_deg_wrapped)

        # Zenith Angles
        c_theta_nlos = self._scenario.get_param("CThetaNLoS")
        # 8.4-15
        c_theta_los = c_theta_nlos * (1.3086 + 0.0339 * k_factor_db - 
                                     0.0077 * tf.square(k_factor_db) + 0.0002 * tf.pow(k_factor_db, 3.))
        c_theta = tf.where(self._scenario.los, c_theta_los, c_theta_nlos)
        c_theta = tf.expand_dims(c_theta, axis=-1)

        # ZOA
        zsa_rad = deg_2_rad(lsp.zsa)
        # 8.4-14
        theta_prime_zoa = -tf.expand_dims(zsa_rad, -1) *tf.math.log(powers_ratio) / c_theta
        # 8.4-17
        complex_phasors_zoa = tf.complex(rt_powers, 0.) * tf.exp(tf.complex(0., rt_zoa_rad))
        theta_center_zoa = tf.math.angle(tf.reduce_sum(complex_phasors_zoa, axis=-1, keepdims=True))

        theta_bar_zoa = tf.where(tf.expand_dims(self._scenario.indoor, -1), deg_2_rad(tf.constant(90.0, dtype=self.rdtype)), theta_center_zoa)

        # 8.4-16
        X_n_zoa = tf.cast(tf.random.uniform(shape=shape_rc, minval=0, maxval=2, dtype=tf.int32) * 2 - 1, self.rdtype)
        Y_n_zoa = tf.random.normal(shape=shape_rc, mean=0.0, stddev=tf.expand_dims(zsa_rad, -1)/7.0, dtype=self.rdtype)
        rc_zoa_rad = X_n_zoa * theta_prime_zoa + Y_n_zoa + theta_bar_zoa
        rc_zoa_deg = rad_2_deg(rc_zoa_rad)
        rc_zoa_deg_wrapped = wrap_angle_0_360(rc_zoa_deg)
        rc_zoa_deg_wrapped = tf.where(tf.math.greater(rc_zoa_deg_wrapped, 180.), rc_zoa_deg_wrapped-360., rc_zoa_deg_wrapped)

        # ZOD
        zsd_rad = deg_2_rad(lsp.zsd)
        theta_prime_zod = -tf.expand_dims(zsd_rad, -1) *tf.math.log(powers_ratio) / c_theta
        complex_phasors_zod = tf.complex(rt_powers, 0.) * tf.exp(tf.complex(0., rt_zod_rad))
        theta_center_zod = tf.math.angle(tf.reduce_sum(complex_phasors_zod, axis=-1, keepdims=True))

        mu_offset_zod = deg_2_rad(self._scenario.zod_offset)
        # 8.4-18
        X_n_zod = tf.cast(tf.random.uniform(shape=shape_rc, minval=0, maxval=2, dtype=tf.int32)*2-1, self.rdtype)
        Y_n_zod = tf.random.normal(shape=shape_rc, mean=0.0, stddev=tf.expand_dims(zsd_rad, -1)/7.0, dtype=self.rdtype)   
        rc_zod_rad = X_n_zod * theta_prime_zod + Y_n_zod + theta_center_zod + tf.expand_dims(mu_offset_zod, -1)
        rc_zod_deg = rad_2_deg(rc_zod_rad)
        rc_zod_deg_wrapped = wrap_angle_0_360(rc_zod_deg)
        rc_zod_deg_wrapped = tf.where(tf.math.greater(rc_zod_deg_wrapped, 180.), rc_zod_deg_wrapped-360., rc_zod_deg_wrapped)

        return (rc_aoa_deg_wrapped, rc_aod_deg_wrapped, rc_zoa_deg_wrapped, rc_zod_deg_wrapped)
            
    def __call__(self, lsp_generator, rt_params_np, los_flag_np):
        """
        Execute Steps 4-8: Complete cluster generation pipeline.
        
        This method orchestrates the entire preprocessing pipeline:
        - Step 4: Ray-tracing input preparation and LoS status update
        - Step 5: RC delay generation
        - Step 6: RC power generation  
        - Step 7: RC angle generation
        - Step 8: RT+RC cluster combination
        
        Parameters
        ----------
        lsp_generator : LSPGenerator
            Sionna LSPGenerator instance for LSP sampling
        rt_params_np : dict[str, np.ndarray]
            Ray-tracer output parameters containing:
            - 'tau': delay times [B, UT, BS, P]
            - 'phi_r', 'phi_t': azimuth angles [B, UT, BS, P]
            - 'theta_r', 'theta_t': zenith angles [B, UT, BS, P]
            - 'a_re', 'a_im': complex amplitudes [B, UT, BS, rx_ant, tx_ant, P]
        los_flag_np : np.ndarray
            LoS/NLoS flags [B, UT, BS] (True=LoS, False=NLoS)
            
        Returns
        -------
        tuple[dict[str, tf.Tensor], list[int]]
            - clusters: Combined RT+RC cluster parameters
                - 'tau': delay times [B, BS, UT, N] (seconds)
                - 'powers': powers [B, BS, UT, N] (linear scale)
                - 'phi_r', 'phi_t': azimuth angles [B, BS, UT, N] (degrees)
                - 'theta_r', 'theta_t': zenith angles [B, BS, UT, N] (degrees)
            - rt_connected_bs: List of BS indices with valid RT paths
            
        Raises
        ------
        RayTracingError
            If no valid ray-tracing paths are found
        """

        self.lsp_generator = lsp_generator                 # For later callback

        self._update_scenario_los(los_flag_np)             # Step-4

        lsp = self.lsp_generator()                         # LSP sampling



        # ── Step-4 : RT valid path mask
        valid_paths_mask_np = rt_params_np["tau"] > INVALID_DELAY   # (B,UT,BS,P)
        if not np.any(valid_paths_mask_np):
            raise RayTracingError("No valid ray-tracing paths found in input data")

        # Collect BS indices with RT paths (for debugging/statistics)
        rt_connected_bs = [
            bs for bs in range(rt_params_np["tau"].shape[2])
            if np.any(valid_paths_mask_np[0, 0, bs, :])
        ]


        # ── Step-4 : numpy → tf conversion + RT tensor rearrangement
        rt_params = self._prepare_rt_inputs(rt_params_np, valid_paths_mask_np)
        rt_delays  = rt_params["tau"]     # [B,BS,UT,L_RT]
        rt_powers  = rt_params["powers"]  # Same shape

        # ── Step-5 : RC delays
        rc_delays, rc_delays_unscaled = self._generate_rc_delays(lsp, rt_delays)

        # ── Step-6 : RC powers
        rc_powers = self._generate_rc_powers(
            lsp, rt_delays, rt_powers, rc_delays_unscaled
        )

        # ── Step-7 : RC angles
        rc_phi_r, rc_phi_t, rc_theta_r, rc_theta_t = self._generate_rc_angles(
            lsp, rt_params, rc_powers
        )

        # ── Step-8 : Combine RT + RC clusters
        rt_phi_r_deg   = rt_params["phi_r"]   * RAD_TO_DEG
        rt_phi_t_deg   = rt_params["phi_t"]   * RAD_TO_DEG
        rt_theta_r_deg = rt_params["theta_r"] * RAD_TO_DEG
        rt_theta_t_deg = rt_params["theta_t"] * RAD_TO_DEG

        clusters = {
            "tau"     : tf.concat([rt_delays,  rc_delays ], axis=-1),
            "powers"  : tf.concat([rt_powers,  rc_powers ], axis=-1),
            "phi_r"   : tf.concat([rt_phi_r_deg,   rc_phi_r   ], axis=-1),
            "phi_t"   : tf.concat([rt_phi_t_deg,   rc_phi_t   ], axis=-1),
            "theta_r" : tf.concat([rt_theta_r_deg, rc_theta_r ], axis=-1),
            "theta_t" : tf.concat([rt_theta_t_deg, rc_theta_t ], axis=-1),
        }

        # ── Step-8 end : Power threshold cutoff
        max_p   = tf.reduce_max(clusters["powers"], axis=-1, keepdims=True)
        thresh  = max_p * tf.pow(10., POWER_THRESHOLD_DB/10.0)
        weak_m  = clusters["powers"] < thresh          # bool mask

        for k, t in clusters.items():                  # tau is INVALID_DELAY, others are 0
            clusters[k] = tf.where(
                weak_m,
                tf.constant(INVALID_DELAY, self.rdtype) if k == "tau" else tf.constant(0.0, self.rdtype),
                t
            )

        return clusters, rt_connected_bs

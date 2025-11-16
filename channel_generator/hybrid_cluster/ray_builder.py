# hybrid_cluster/ray_builder.py
from sionna.phy.channel.tr38901.rays import RaysGenerator
from .constants import (
    RAY_OFFSETS_DEG, INVALID_DELAY,
    SPEED_OF_LIGHT, PI, RAD_TO_DEG
)
from .exceptions import ValidationError, ParameterError
import tensorflow as tf
import numpy as np
from sionna.phy.channel.utils import wrap_angle_0_360

class HybridClusterRayBuilder(RaysGenerator):
    """
    TR 38.901 §8.4 Steps 9-12: Intra-cluster ray generation
    
    This class handles the ray generation steps for hybrid cluster generation:
    - Step 9: Generate intra-cluster rays (TR38901 8.4-19 to 8.4-21)
    - Step 10: Generate ray powers
    - Step 11: Generate XPR (Cross Polarization Ratio) (TR38901 8.4-26)
    - Step 12: Generate initial phases
    
    Parameters
    ----------
    scenario : Scenario
        Sionna scenario object containing topology and parameters
    """
    
    def __init__(self, scenario):
        """
        Initialize the ray builder with scenario configuration.
        
        Parameters
        ----------
        scenario : Scenario
            Sionna scenario object containing topology and parameters
        """
        super().__init__(scenario)
        self._scenario = scenario

    def _generate_intra_cluster_rays(self, clusters, lsp, los_mask):
        """
        Step 9: Generate intra-cluster rays (TR38901 8.4-19 to 8.4-21)
        
        Parameters
        ----------
        clusters : dict[str, tf.Tensor]
            Cluster parameters containing:
            - 'tau': delay times [B, BS, UT, N]
            - 'powers': powers [B, BS, UT, N]
            - 'phi_r', 'phi_t': azimuth angles [B, BS, UT, N] (degrees)
            - 'theta_r', 'theta_t': zenith angles [B, BS, UT, N] (degrees)
        lsp : LSP
            Large Scale Parameters object containing angle spreads
        los_mask : tf.Tensor
            Boolean LoS mask with shape `[B, BS, UT]`.
            
        Returns
        -------
        dict[str, tf.Tensor]
            Ray parameters for each cluster:
            - 'tau': relative delay times [B, BS, UT, N, M]
            - 'phi_r': ray azimuth angles of arrival [B, BS, UT, N, M] (degrees)
            - 'phi_t': ray azimuth angles of departure [B, BS, UT, N, M] (degrees)
            - 'theta_r': ray zenith angles of arrival [B, BS, UT, N, M] (degrees)
            - 'theta_t': ray zenith angles of departure [B, BS, UT, N, M] (degrees)
            
        Notes
        -----
        This method implements TR 38.901 equations 8.4-19 to 8.4-21 for
        generating intra-cluster rays with proper angle offsets and wrapping.
        """
        # Input validation
        if not isinstance(clusters, dict):
            raise ValidationError("clusters must be a dictionary")
        
        required_keys = ['tau', 'powers', 'phi_r', 'phi_t', 'theta_r', 'theta_t']
        missing_keys = [key for key in required_keys if key not in clusters]
        if missing_keys:
            raise ValidationError(f"Missing required keys in clusters: {missing_keys}")
        
        # Check only valid clusters
        valid_cluster_mask = tf.logical_and(
            clusters['tau'] > INVALID_DELAY,  # Valid delay time
            clusters['powers'] > 0.0  # Non-zero power
        )

        M = self._scenario.rays_per_cluster
        if M <= 0:
            raise ParameterError("rays_per_cluster must be greater than 0")
        if M > len(RAY_OFFSETS_DEG):
            raise ParameterError(f"rays_per_cluster ({M}) exceeds available ray offsets ({len(RAY_OFFSETS_DEG)})")
        
        alpha_m = tf.reshape(RAY_OFFSETS_DEG[:M], [1,1,1,1,M])

        # For LoS links, the ray angle offsets of the LoS cluster (first cluster)
        # must be zero, so that the LoS ray direction is identical to the 
        # cluster direction. The other rays of this cluster will have zero power.
        if tf.reduce_any(los_mask):
            _B, _BS, _UT, N = tf.unstack(tf.shape(clusters['tau']))
            
            # Mask for the first cluster (n=0)
            first_cluster_mask = tf.one_hot(0, N, on_value=True, off_value=False, dtype=tf.bool)
            first_cluster_mask = tf.reshape(first_cluster_mask, [1, 1, 1, N, 1])

            # Mask for LoS clusters
            los_cluster_mask = tf.expand_dims(tf.expand_dims(los_mask, -1), -1) & first_cluster_mask

            # For LoS clusters, ray angle offsets should be zero
            alpha_m_los = tf.zeros_like(alpha_m)
            alpha_m = tf.where(los_cluster_mask, alpha_m_los, alpha_m)

        C_ASA = tf.expand_dims(lsp.asa, -1)
        C_ASD = tf.expand_dims(lsp.asd, -1)
        C_ZSA = tf.expand_dims(lsp.zsa, -1)
        C_ZSD = (3./8.)*tf.math.pow(tf.constant(10.,
                self.rdtype),
                self._scenario.lsp_log_mean[:,:,:,6])
        C_ZSD = tf.expand_dims(C_ZSD, -1)
        
        # Important: Convert cluster angles from radian to degree
        phi_r_deg = clusters['phi_r'] * RAD_TO_DEG
        phi_t_deg = clusters['phi_t'] * RAD_TO_DEG
        theta_r_deg = clusters['theta_r'] * RAD_TO_DEG
        theta_t_deg = clusters['theta_t'] * RAD_TO_DEG
        
        def _add_offset(center_deg, spread_deg):
            center_exp = tf.expand_dims(center_deg, -1)
            spread_deg = tf.expand_dims(spread_deg, -1)
            return center_exp + spread_deg * alpha_m
        # 8.4-19~21
        phi_rays_aoa   = _add_offset(phi_r_deg,   C_ASA) 
        phi_rays_aod   = _add_offset(phi_t_deg,   C_ASD)
        theta_rays_zoa = _add_offset(theta_r_deg, C_ZSA) 
        theta_rays_zod = _add_offset(theta_t_deg, C_ZSD) 
        
        def _wrap180(x): 
            x = wrap_angle_0_360(x)
            return tf.where(x > 180. , x-360., x)

        phi_rays_aoa   = _wrap180(phi_rays_aoa)
        phi_rays_aod   = _wrap180(phi_rays_aod)

        def _wrap180pos(x):
            x = wrap_angle_0_360(x)
            return tf.where(x > 180., 360.-x, x)

        theta_rays_zoa = _wrap180pos(theta_rays_zoa)
        theta_rays_zod = _wrap180pos(theta_rays_zod)
        
        # Calculate tau correctly: get from clusters['tau'] and copy for each ray
        cluster_tau = clusters['tau']  # [B, BS, UT, N]
        tau = tf.expand_dims(cluster_tau, -1)  # [B, BS, UT, N, 1]
        tau = tf.broadcast_to(tau, tf.shape(phi_rays_aoa))  # [B, BS, UT, N, M]
        
        # Set rays of invalid clusters to 0
        valid_mask_expanded = tf.expand_dims(valid_cluster_mask, -1)  # [B,BS,UT,N,1]
        
        phi_rays_aoa = tf.where(valid_mask_expanded, phi_rays_aoa, 0.0)
        phi_rays_aod = tf.where(valid_mask_expanded, phi_rays_aod, 0.0)  
        theta_rays_zoa = tf.where(valid_mask_expanded, theta_rays_zoa, 0.0)
        theta_rays_zod = tf.where(valid_mask_expanded, theta_rays_zod, 0.0)
        tau = tf.where(valid_mask_expanded, tau, 0.0)
      
        return {
            "tau"     : tau,
            "phi_r"   : phi_rays_aoa,
            "phi_t"   : phi_rays_aod,
            "theta_r" : theta_rays_zoa,
            "theta_t" : theta_rays_zod,
        }

    def _generate_ray_powers(self, clusters, rays, los_mask):
        """
        Step 10: Generate ray powers
        
        Parameters
        ----------
        clusters : dict[str, tf.Tensor]
            Cluster parameters containing 'powers' [B, BS, UT, N]
        rays : dict[str, tf.Tensor]
            Ray parameters from _generate_intra_cluster_rays
        los_mask : tf.Tensor
            Boolean LoS mask with shape `[B, BS, UT]`.
            
        Returns
        -------
        tf.Tensor
            Ray powers [B, BS, UT, N, M] (linear scale)
            
        Notes
        -----
        Ray powers are calculated by dividing cluster powers equally
        among all rays within each cluster.
        """
        M = self._scenario.rays_per_cluster
        P_cluster = tf.expand_dims(clusters["powers"], -1)
        
        # Default: distribute power equally among rays for NLoS
        P_ray_nlos = P_cluster / tf.cast(M, self.rdtype)
        P_ray = tf.broadcast_to(P_ray_nlos, tf.shape(rays['tau']))

        # For LoS links, concentrate all power of the first cluster into the first ray
        if tf.reduce_any(los_mask):
            _B, _BS, _UT, N, M_dyn = tf.unstack(tf.shape(rays['tau']))

            # Mask for the first cluster (n=0)
            first_cluster_mask = tf.one_hot(0, N, on_value=True, off_value=False, dtype=tf.bool)
            first_cluster_mask = tf.reshape(first_cluster_mask, [1, 1, 1, N, 1])

            # Mask for LoS clusters
            los_cluster_mask = tf.expand_dims(tf.expand_dims(los_mask, -1), -1) & first_cluster_mask

            # Power distribution for LoS cluster: [P_cluster, 0, 0, ...]
            los_power_dist = tf.one_hot(0, M_dyn, dtype=self.rdtype)
            los_power_dist = tf.reshape(los_power_dist, [1,1,1,1,M_dyn])
            P_ray_los = P_cluster * los_power_dist

            P_ray = tf.where(los_cluster_mask, P_ray_los, P_ray)
        
        # Explicitly set ray powers of invalid clusters to 0 
        valid_cluster_mask = tf.logical_and(
            clusters['tau'] > INVALID_DELAY,  # Valid delay time
            clusters['powers'] > 0.0  # Non-zero power
        )
        valid_mask_expanded = tf.expand_dims(valid_cluster_mask, -1)  # [B,BS,UT,N,1]
        P_ray = tf.where(valid_mask_expanded, P_ray, 0.0)
        
        return P_ray

    def _generate_xpr(self, rays, los_mask):
        """
        Step 11: Generate XPR (Cross Polarization Ratio) (TR38901 8.4-26)
        
        Parameters
        ----------
        rays : dict[str, tf.Tensor]
            Ray parameters from _generate_intra_cluster_rays
        los_mask : tf.Tensor
            Boolean LoS mask with shape `[B, BS, UT]`.
            
        Returns
        -------
        tf.Tensor
            XPR values [B, BS, UT, N, M] (linear scale)
            
        Notes
        -----
        This method implements TR 38.901 equation 8.4-26 for generating
        cross polarization ratios for each ray.
        """
        B, BS, UT, N, M = tf.unstack(tf.shape(rays["phi_r"]))

        mu_xpr = self._scenario.get_param("muXPR")
        sigma_xpr = self._scenario.get_param("sigmaXPR")

        mu_xpr    = tf.expand_dims(tf.expand_dims(mu_xpr,    3), 4)
        sigma_xpr = tf.expand_dims(tf.expand_dims(sigma_xpr, 3), 4)

        x_dB = tf.random.normal(shape=[B, BS, UT, N, M],
                                mean=mu_xpr,
                                stddev=sigma_xpr,
                                dtype=self.rdtype)

        kappa = tf.pow(tf.constant(10.0, self.rdtype), x_dB / 10.0)
        
        # For LoS links, override the kappa of the first ray of the first cluster
        # to be very high (effectively infinite XPR).
        if tf.reduce_any(los_mask):
            
            # Mask for the first cluster (n=0)
            first_cluster_mask = tf.one_hot(0, N, on_value=True, off_value=False, dtype=tf.bool)
            first_cluster_mask = tf.reshape(first_cluster_mask, [1,1,1,N,1])

            # Mask for the first ray (m=0)
            first_ray_mask = tf.one_hot(0, M, on_value=True, off_value=False, dtype=tf.bool)
            first_ray_mask = tf.reshape(first_ray_mask, [1,1,1,1,M])

            # Combine masks for the single LoS ray
            los_mask_exp = tf.expand_dims(tf.expand_dims(los_mask, -1), -1)
            los_ray_mask = los_mask_exp & first_cluster_mask & first_ray_mask

            # Set kappa for the LoS ray to a large value
            kappa = tf.where(los_ray_mask, tf.constant(1000.0, self.rdtype), kappa)
            
        return kappa   

    def _generate_initial_phases(self, rays, los_mask):
        """
        Step 12: Generate initial phases with LOS correction
        
        Parameters
        ----------
        rays : dict[str, tf.Tensor]
            Ray parameters from _generate_intra_cluster_rays
        los_mask : tf.Tensor
            Boolean LoS mask with shape `[B, BS, UT]`.
            
        Returns
        -------
        tf.Tensor
            Initial phases [B, BS, UT, N, M] (radians)
            
        Notes
        -----
        Initial phases are generated randomly for each ray and include
        LoS correction for line-of-sight links.
        """
        rdtype   = self.rdtype
        shape_phi = tf.shape(rays["phi_r"])
        B, BS, UT, N, M = tf.unstack(shape_phi)
        phases = tf.random.uniform(
                    shape=tf.concat([shape_phi, [4]], axis=0),
                    minval=-np.pi, maxval=np.pi, dtype=rdtype)

        if not tf.reduce_any(los_mask):
            return phases

        d3d     = tf.cast(self._scenario.distance_3d, rdtype)
        lambda0 = tf.constant(SPEED_OF_LIGHT, rdtype) / self._scenario.carrier_frequency
        phi_los = -2.0 * PI * d3d / lambda0
        
        # Create a specific mask for the first ray of the first cluster of LoS links
        # Mask for the first cluster (n=0)
        first_cluster_mask = tf.one_hot(0, N, on_value=True, off_value=False, dtype=tf.bool)
        first_cluster_mask = tf.reshape(first_cluster_mask, [1,1,1,N,1])

        # Mask for the first ray (m=0)
        first_ray_mask = tf.one_hot(0, M, on_value=True, off_value=False, dtype=tf.bool)
        first_ray_mask = tf.reshape(first_ray_mask, [1,1,1,1,M])

        # Combine masks for the single LoS ray
        los_mask_exp = tf.expand_dims(tf.expand_dims(los_mask, -1), -1)
        los_ray_mask = los_mask_exp & first_cluster_mask & first_ray_mask

        # Expand mask for the 4 phase components
        los_ray_mask_4d = tf.expand_dims(los_ray_mask, -1)

        # Reshape phi_los for broadcasting
        phi_los = phi_los[..., tf.newaxis, tf.newaxis, tf.newaxis]
        phi_los_broadcasted = tf.broadcast_to(phi_los, tf.shape(phases))

        # Apply LoS phase correction only to F_theta_theta (idx 0) and F_phi_phi (idx 3)
        # for the single LoS ray.
        theta_theta_mask = los_ray_mask_4d & tf.equal(tf.range(4), 0)
        phases = tf.where(theta_theta_mask, phi_los_broadcasted, phases)
        
        phi_phi_mask = los_ray_mask_4d & tf.equal(tf.range(4), 3)
        phases = tf.where(phi_phi_mask, phi_los_broadcasted, phases)

        return phases

    def __call__(self, clusters, lsp, los_mask):
        """
        Execute Steps 9-12: Complete ray generation pipeline.
        
        This method orchestrates the entire ray generation pipeline:
        - Step 9: Generate intra-cluster rays
        - Step 10: Generate ray powers
        - Step 11: Generate XPR (Cross Polarization Ratio)
        - Step 12: Generate initial phases with LOS correction
        
        Parameters
        ----------
        clusters : dict[str, tf.Tensor]
            Cluster parameters from preprocessor containing:
            - 'tau': delay times [B, BS, UT, N]
            - 'powers': powers [B, BS, UT, N]
            - 'phi_r', 'phi_t': azimuth angles [B, BS, UT, N] (degrees)
            - 'theta_r', 'theta_t': zenith angles [B, BS, UT, N] (degrees)
        lsp : LSPSample
            Large Scale Parameters sample containing ASA, ZSA, etc.
        los_mask : tf.Tensor
            Boolean LoS mask with shape `[B, BS, UT]`.
            
        Returns
        -------
        tuple[dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]
            - rays: Ray parameters for each cluster
                - 'tau_rel': relative delay times [B, BS, UT, N, M]
                - 'phi_r', 'phi_t': ray azimuth angles [B, BS, UT, N, M] (degrees)
                - 'theta_r', 'theta_t': ray zenith angles [B, BS, UT, N, M] (degrees)
            - p_ray: Ray powers [B, BS, UT, N, M] (linear scale)
            - kappa: XPR values [B, BS, UT, N, M] (linear scale)
            - phi: Initial phases [B, BS, UT, N, M, 4] (radians)
        """
        # Step-9 : intra-cluster ray geometry
        rays = self._generate_intra_cluster_rays(clusters, lsp, los_mask)

        # Step-10 : ray powers
        p_ray = self._generate_ray_powers(clusters, rays, los_mask)

        # Step-11 : XPR
        kappa = self._generate_xpr(rays, los_mask)

        # Step-12 : initial phases (with LOS correction)
        phi   = self._generate_initial_phases(rays, los_mask)

        return rays, p_ray, kappa, phi

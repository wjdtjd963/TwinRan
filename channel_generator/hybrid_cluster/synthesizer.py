# hybrid_cluster/synthesizer.py
from sionna.phy.channel.tr38901.rays import RaysGenerator
from .constants import (
    SPEED_OF_LIGHT, PI, DEG_TO_RAD
)
from .exceptions import ValidationError, ParameterError
import tensorflow as tf
import numpy as np

class HybridChannelSynthesizer(RaysGenerator):
    """
    TR 38.901 §8.4 Step 13: LOS/NLOS channel synthesis
    
    This class handles the final channel synthesis step for hybrid cluster generation:
    - Step 13: LOS/NLOS synthesis to generate channel coefficients and delays
    
    Parameters
    ----------
    scenario : Scenario
        Sionna scenario object containing topology and parameters
    """
    
    def __init__(self, scenario):
        """
        Initialize the channel synthesizer with scenario configuration.
        
        Parameters
        ----------
        scenario : Scenario
            Sionna scenario object containing topology and parameters
        """
        super().__init__(scenario)
        self._scenario = scenario

    def _unit_sphere_vector(self, theta, phi):
        """
        Calculate unit sphere vector from spherical coordinates.
        
        Parameters
        ----------
        theta : tf.Tensor
            Zenith angle (radians)
        phi : tf.Tensor
            Azimuth angle (radians)
            
        Returns
        -------
        tf.Tensor
            Unit sphere vector [..., 3]
        """
        rho_hat = tf.stack([tf.sin(theta)*tf.cos(phi),
                           tf.sin(theta)*tf.sin(phi), 
                           tf.cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)
    
    def _forward_rotation_matrix(self, orientations):
        """
        Calculate forward composite rotation matrix (TR38901 7.1-4).
        
        Parameters
        ----------
        orientations : tf.Tensor
            Orientation angles [..., 3] (alpha, beta, gamma)
            
        Returns
        -------
        tf.Tensor
            Forward rotation matrix [..., 3, 3]
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = tf.stack([tf.cos(a)*tf.cos(b),
            tf.cos(a)*tf.sin(b)*tf.sin(c)-tf.sin(a)*tf.cos(c),
            tf.cos(a)*tf.sin(b)*tf.cos(c)+tf.sin(a)*tf.sin(c)], axis=-1)

        row_2 = tf.stack([tf.sin(a)*tf.cos(b),
            tf.sin(a)*tf.sin(b)*tf.sin(c)+tf.cos(a)*tf.cos(c),
            tf.sin(a)*tf.sin(b)*tf.cos(c)-tf.cos(a)*tf.sin(c)], axis=-1)

        row_3 = tf.stack([-tf.sin(b),
            tf.cos(b)*tf.sin(c),
            tf.cos(b)*tf.cos(c)], axis=-1)

        rot_mat = tf.stack([row_1, row_2, row_3], axis=-2)
        return rot_mat
    
    def _rot_pos(self, orientations, positions):
        """Rotate positions according to orientations"""
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)
    
    def _reverse_rotation_matrix(self, orientations):
        """Reverse composite rotation matrix"""
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv
    
    def _gcs_to_lcs(self, orientations, theta, phi):
        """
        Convert angles from Global Coordinate System (GCS) to Local Coordinate System (LCS).
        
        Parameters
        ----------
        orientations : tf.Tensor
            Orientation angles [..., 3] (alpha, beta, gamma)
        theta : tf.Tensor
            Zenith angle in GCS (radians)
        phi : tf.Tensor
            Azimuth angle in GCS (radians)
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            - theta_prime: Zenith angle in LCS (radians)
            - phi_prime: Azimuth angle in LCS (radians)
            
        Notes
        -----
        This method implements TR 38.901 equations 7.1-7 and 7.1-8.
        """
        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = tf.matmul(rot_inv, rho_hat)
        
        v1 = tf.constant([0,0,1], self.rdtype)
        v1 = tf.reshape(v1, [1]*(rot_rho.shape.rank-1)+[3])
        v2 = tf.constant([1+0j,1j,0], self.cdtype)
        v2 = tf.reshape(v2, [1]*(rot_rho.shape.rank-1)+[3])
        
        z = tf.matmul(v1, rot_rho)
        z = tf.clip_by_value(z, tf.constant(-1., self.rdtype),
                             tf.constant(1., self.rdtype))
        theta_prime = tf.acos(z)
        phi_prime = tf.math.angle((tf.matmul(v2, tf.cast(rot_rho, self.cdtype))))
        
        theta_prime = tf.squeeze(theta_prime, axis=[-2, -1])
        phi_prime = tf.squeeze(phi_prime, axis=[-2, -1])
        
        return theta_prime, phi_prime
    
    def _compute_psi(self, orientations, theta, phi):
        """
        Compute displacement angle Psi for LCS-GCS field transformation (TR38901 7.1-15).
        
        Parameters
        ----------
        orientations : tf.Tensor
            Orientation angles [..., 3] (alpha, beta, gamma)
        theta : tf.Tensor
            Zenith angle (radians)
        phi : tf.Tensor
            Azimuth angle (radians)
            
        Returns
        -------
        tf.Tensor
            Displacement angle Psi (radians)
        """
        a = orientations[...,0]
        b = orientations[...,1] 
        c = orientations[...,2]
        
        real = tf.sin(c)*tf.cos(theta)*tf.sin(phi-a)
        real += tf.cos(c)*(tf.cos(b)*tf.sin(theta)-tf.sin(b)*tf.cos(theta)*tf.cos(phi-a))
        imag = tf.sin(c)*tf.cos(phi-a) + tf.sin(b)*tf.cos(c)*tf.sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi
    
    def _l2g_response(self, f_prime, orientations, theta, phi):
        """
        Transform field components from Local Coordinate System (LCS) to Global Coordinate System (GCS).
        
        Parameters
        ----------
        f_prime : tf.Tensor
            Field components in LCS [..., 2]
        orientations : tf.Tensor
            Orientation angles [..., 3] (alpha, beta, gamma)
        theta : tf.Tensor
            Zenith angle (radians)
        phi : tf.Tensor
            Azimuth angle (radians)
            
        Returns
        -------
        tf.Tensor
            Field components in GCS [..., 2]
            
        Notes
        -----
        This method implements TR 38.901 equation 7.1-11.
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([tf.cos(psi), -tf.sin(psi)], axis=-1)
        row2 = tf.stack([tf.sin(psi), tf.cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f
                        
    def _step_13_phase_matrix(self, phi, kappa):
        """
        Step 13: Generate phase matrix for 2x2 polarization.
        
        Parameters
        ----------
        phi : tf.Tensor
            Initial phases [..., 4] (radians)
        kappa : tf.Tensor
            XPR values [..., 1] (linear scale)
            
        Returns
        -------
        tf.Tensor
            Phase matrix [..., 2, 2] (complex)
        """
        phi_complex = tf.cast(phi, self.cdtype)
        kappa_complex = tf.cast(kappa, self.cdtype)
        j_complex = tf.constant(1j, self.cdtype)
        
        F_theta_theta = tf.exp(j_complex * phi_complex[..., 0])
        F_theta_phi   = tf.exp(j_complex * phi_complex[..., 1]) / tf.sqrt(kappa_complex)
        F_phi_theta   = tf.exp(j_complex * phi_complex[..., 2]) / tf.sqrt(kappa_complex)
        F_phi_phi     = tf.exp(j_complex * phi_complex[..., 3])
        
        h_row1 = tf.stack([F_theta_theta, F_theta_phi], axis=-1)
        h_row2 = tf.stack([F_phi_theta, F_phi_phi], axis=-1)
        h_phase = tf.stack([h_row1, h_row2], axis=-2)
        
        return h_phase
    
    def _step_13_array_offsets(self, aoa, aod, zoa, zod):
        """Step 13: Antenna array offset phases"""
        if self._scenario.direction == "downlink":
            tx_array = self._scenario.bs_array
            rx_array = self._scenario.ut_array
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        else:
            tx_array = self._scenario.ut_array
            rx_array = self._scenario.bs_array
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations
        
        lambda_0 = self._scenario.lambda_0
        
        r_hat_rx_raw = self._unit_sphere_vector(zoa, aoa)
        r_hat_tx_raw = self._unit_sphere_vector(zod, aod)
        r_hat_rx = tf.squeeze(r_hat_rx_raw, axis=-1)
        r_hat_tx = tf.squeeze(r_hat_tx_raw, axis=-1)
        
        tx_ant_pos_lcs = tx_array.ant_pos
        rx_ant_pos_lcs = rx_array.ant_pos
        
        if self._scenario.direction == "downlink":
            tx_orientations_exp = tf.expand_dims(tx_orientations, 2)
            rx_orientations_exp = tf.expand_dims(rx_orientations, 2)
        else:
            tx_orientations_exp = tf.expand_dims(tx_orientations, 2)
            rx_orientations_exp = tf.expand_dims(rx_orientations, 2)
        
        tx_ant_pos_lcs_exp = tf.reshape(tx_ant_pos_lcs, [1, 1] + list(tx_ant_pos_lcs.shape) + [1])
        rx_ant_pos_lcs_exp = tf.reshape(rx_ant_pos_lcs, [1, 1] + list(rx_ant_pos_lcs.shape) + [1])
        
        tx_ant_pos_gcs = self._rot_pos(tx_orientations_exp, tx_ant_pos_lcs_exp)
        rx_ant_pos_gcs = self._rot_pos(rx_orientations_exp, rx_ant_pos_lcs_exp)
        
        tx_ant_pos_gcs = tf.squeeze(tx_ant_pos_gcs, axis=-1)
        rx_ant_pos_gcs = tf.squeeze(rx_ant_pos_gcs, axis=-1)
        
        if self._scenario.direction == "downlink":
            tx_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(tx_ant_pos_gcs, 2), 3), 4)
            rx_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(rx_ant_pos_gcs, 1), 3), 4)
        else:
            tx_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(tx_ant_pos_gcs, 1), 3), 4)
            rx_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(rx_ant_pos_gcs, 2), 3), 4)
            
        r_hat_rx_exp = tf.expand_dims(r_hat_rx, -2)
        r_hat_tx_exp = tf.expand_dims(r_hat_tx, -2)
        
        dot_prod_rx = tf.reduce_sum(r_hat_rx_exp * rx_positions, axis=-1)
        dot_prod_tx = tf.reduce_sum(r_hat_tx_exp * tx_positions, axis=-1)

        antenna_path_difference = tf.expand_dims(dot_prod_rx, -1) + tf.expand_dims(dot_prod_tx, -2)
        
        phase = 2 * np.pi / lambda_0 * antenna_path_difference
        
        h_array = tf.exp(tf.complex(tf.constant(0., self.rdtype), phase))

        return h_array, antenna_path_difference
    
    def _step_13_doppler_matrix(self, aoa, zoa, sample_times):
        """Step 13: Doppler matrix calculation"""
        lambda_0 = self._scenario.lambda_0
        velocities = self._scenario.ut_velocities
        
        r_hat_rx_raw = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = tf.squeeze(r_hat_rx_raw, axis=-1)
        
        v_bar = tf.expand_dims(velocities, 1)
        v_bar = tf.expand_dims(tf.expand_dims(v_bar, -2), -2)
        
        v_dot_r = tf.reduce_sum(r_hat_rx * v_bar, axis=-1)
        
        doppler_freq = v_dot_r / lambda_0
        doppler_freq = tf.expand_dims(doppler_freq, -1)
        
        time_expanded = tf.reshape(sample_times, [1, 1, 1, 1, 1, -1])
        
        exponent = 2 * np.pi * doppler_freq * time_expanded
        
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        
        return h_doppler
    
    def _step_13_field_matrix(self, aoa, aod, zoa, zod, h_phase):
        """Step 13: Field matrix calculation with complete LCS transformation"""
        if self._scenario.direction == "downlink":
            tx_array = self._scenario.bs_array
            rx_array = self._scenario.ut_array  
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        else:
            tx_array = self._scenario.ut_array
            rx_array = self._scenario.bs_array
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations
            
        B, BS, UT, N, M = tf.unstack(tf.shape(aoa))
            
        tx_orientations_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(tx_orientations, 2), 3), 4)
        tx_orientations_exp = tf.broadcast_to(tx_orientations_exp, [B, BS, UT, N, M, 3])
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations_exp, zod, aod)
        
        rx_orientations_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(rx_orientations, 1), 3), 4)
        rx_orientations_exp = tf.broadcast_to(rx_orientations_exp, [B, BS, UT, N, M, 3])
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations_exp, zoa, aoa)
        
        f_tx_pol1_prime = tf.stack(tx_array._ant_pol1.field(zod_prime, aod_prime), axis=-1)
        f_rx_pol1_prime = tf.stack(rx_array._ant_pol1.field(zoa_prime, aoa_prime), axis=-1)
        
        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations_exp, zod, aod)
        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations_exp, zoa, aoa)
        
        f_tx_pol1 = tf.squeeze(f_tx_pol1, axis=-1)
        f_rx_pol1 = tf.squeeze(f_rx_pol1, axis=-1)
        
        f_tx_array = tf.tile(tf.expand_dims(f_tx_pol1, -2), [1,1,1,1,1,tx_array.num_ant,1])
        f_rx_array = tf.tile(tf.expand_dims(f_rx_pol1, -2), [1,1,1,1,1,rx_array.num_ant,1])
        
        h_phase_exp = tf.expand_dims(h_phase, -3)
        f_tx_exp = tf.expand_dims(f_tx_array, -1)
        pol_tx = tf.squeeze(tf.matmul(h_phase_exp, tf.complex(f_tx_exp, 0.0)), axis=-1)
        
        pol_tx_exp = tf.expand_dims(pol_tx, -3)
        f_rx_exp = tf.expand_dims(tf.complex(f_rx_array, 0.0), -2)
        
        h_field = tf.reduce_sum(f_rx_exp * pol_tx_exp, axis=-1)
        
        return h_field
    
    def _step_13(self, rays, p_ray, kappa, phi, sample_times):
        """
        Step 13: Calculate channel coefficients for all paths.
        
        This method computes the channel coefficients and delays for all paths
        (both LoS and NLoS) provided by the RayBuilder.
        
        Parameters
        ----------
        rays : dict[str, tf.Tensor]
            Ray parameters containing angles and delays
        p_ray : tf.Tensor
            Ray powers [B, BS, UT, N, M] (linear scale)
        kappa : tf.Tensor
            XPR values [B, BS, UT, N, M] (linear scale)
        phi : tf.Tensor
            Initial phases [B, BS, UT, N, M, 4] (radians)
        sample_times : tf.Tensor
            Time samples [T] (seconds)
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            - h: Channel coefficients [B, BS, UT, N*M, rx_ant, tx_ant, T]
            - delays: Delays [B, BS, UT, N*M, rx_ant, tx_ant] (seconds)
        """
        aoa = rays["phi_r"] * DEG_TO_RAD
        aod = rays["phi_t"] * DEG_TO_RAD
        zoa = rays["theta_r"] * DEG_TO_RAD
        zod = rays["theta_t"] * DEG_TO_RAD
        
        h_phase = self._step_13_phase_matrix(phi, kappa)
        h_field = self._step_13_field_matrix(aoa, aod, zoa, zod, h_phase)
        h_array, antenna_path_difference = self._step_13_array_offsets(aoa, aod, zoa, zod)
        h_doppler = self._step_13_doppler_matrix(aoa, zoa, sample_times)
        
        # Combine all components
        h_field_array = h_field * h_array
        h_field_array = tf.expand_dims(h_field_array, -1)
        h_doppler_exp = tf.expand_dims(tf.expand_dims(h_doppler, -2), -2)
        h_full = h_field_array * h_doppler_exp
        
        # Apply power scaling
        power_scaling = tf.sqrt(p_ray)
        power_complex = tf.complex(power_scaling, tf.constant(0., self.rdtype))
        power_shape = tf.concat([tf.shape(power_complex), [1, 1, 1]], 0)
        power_reshaped = tf.reshape(power_complex, power_shape)
        h = h_full * power_reshaped
        
        delays = rays["tau"] # Shape: [B, BS, UT, N, M]
        delays_expanded = tf.expand_dims(tf.expand_dims(delays, -1), -1)
        delays = delays_expanded - antenna_path_difference / tf.constant(SPEED_OF_LIGHT, self.rdtype)
        
        # Reshape to flatten the cluster (N) and ray (M) dimensions into a
        # single 'paths' dimension.
        h_shape = tf.shape(h)
        B, BS, UT, N, M = h_shape[0], h_shape[1], h_shape[2], h_shape[3], h_shape[4]
        
        num_paths = N * M
        
        # New shape for h: [B, BS, UT, num_paths, rx_ant, tx_ant, T]
        remaining_h_shape = h_shape[5:]
        h_reshaped = tf.reshape(h, tf.concat([[B, BS, UT, num_paths], remaining_h_shape], axis=0))
        
        # New shape for delays: [B, BS, UT, num_paths, rx_ant, tx_ant]
        remaining_delays_shape = tf.shape(delays)[5:]
        delays_reshaped = tf.reshape(delays, tf.concat([[B, BS, UT, num_paths], remaining_delays_shape], axis=0))

        return h_reshaped, delays_reshaped
    
    def generate_channel_coefficients(self, rays, p_ray, kappa, phi, 
                                   sample_times):
        """
        Generate final channel coefficients (TR38901 Step 13).
        
        Parameters
        ----------
        rays : dict[str, tf.Tensor]
            Ray parameters from ray builder
        p_ray : tf.Tensor
            Ray powers [B, BS, UT, N, M] (linear scale)
        kappa : tf.Tensor
            XPR values [B, BS, UT, N, M] (linear scale)
        phi : tf.Tensor
            Initial phases [B, BS, UT, N, M, 4] (radians)
        sample_times : tf.Tensor
            Time samples [T] (seconds)
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            - h_base: Final channel coefficients [B, BS, UT, N*M, rx_ant, tx_ant, T]
            - delays: Final delays [B, BS, UT, N*M, rx_ant, tx_ant] (seconds)
        """
        h_base, delays = self._step_13(rays, p_ray, kappa, phi, sample_times)
        
       
        return h_base, delays
        # ──────────────────────────────────────────────────────────────
    def __call__(self,
                 rays, p_ray, kappa, phi,            # RayBuilder output
                 sample_times):                      # 1-D tf.Tensor [T]
        """
        Execute Step 13: Complete channel synthesis pipeline.
        
        This method orchestrates the entire channel synthesis pipeline by
        calculating channel coefficients for all provided rays.
        
        Parameters
        ----------
        rays : dict[str, tf.Tensor]
            Ray parameters from ray builder
        p_ray : tf.Tensor 
            Ray powers [B, BS, UT, N, M] (linear scale)
        kappa : tf.Tensor
            XPR values [B, BS, UT, N, M] (linear scale)
        phi : tf.Tensor
            Initial phases [B, BS, UT, N, M, 4] (radians)
        sample_times : tf.Tensor
            Time samples [T] (seconds)
            
        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            - h: Final channel coefficients [B, BS, UT, N*M, rx_ant, tx_ant, T]
            - delays: Final delays [B, BS, UT, N*M, rx_ant, tx_ant] (seconds)
            
        Raises
        ------
        ValidationError
            If input validation fails
        """
        # Input validation
        if not isinstance(rays, dict):
            raise ValidationError("rays must be a dictionary")
        
        required_ray_keys = ['tau', 'phi_r', 'phi_t', 'theta_r', 'theta_t']
        missing_ray_keys = [key for key in required_ray_keys if key not in rays]
        if missing_ray_keys:
            raise ValidationError(f"Missing required keys in rays: {missing_ray_keys}")
        
        if sample_times is None or tf.rank(sample_times) != 1:
            raise ValidationError("sample_times must be a 1D tensor")
        
        h, delays = self.generate_channel_coefficients(
            rays, p_ray, kappa, phi,
            sample_times
        )
        return h, delays


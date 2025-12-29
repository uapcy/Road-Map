# phase_3_advanced_tomography.py - UPDATED FOR AUTOMATIC DEPTH CALIBRATION
# NOW WITH VECTORIZED FUNCTIONS AND DIAGNOSTICS

import numpy as np
import cvxpy as cp
from scipy import signal
from numpy.linalg import inv
from phase_3_utilities import _calculate_seismic_wavelength

def _calculate_image_sharpness(tomogram):
    """
    Calculates image sharpness (Entropy). 
    Lower Entropy = Sharper Image (better focus).
    We use negative entropy so we can maximize the score.
    """
    magnitude = np.abs(tomogram)
    total_energy = np.sum(magnitude) + 1e-9
    prob_dist = magnitude / total_energy
    
    # Shannon Entropy: -Sum(p * log(p))
    # We want to minimize entropy, so we return 1/Entropy or negative
    entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    
    return -entropy # Higher is better

def perform_velocity_autofocus(Y_matrix, sub_ap_centers, radar_params, 
                               fixed_freq_hz, 
                               velocity_range=(300.0, 6000.0), 
                               steps=20):
    """
    Performs 'Velocity Spectrum Analysis' (Autofocus) to find the correct depth scale.
    Sweeps through material velocities (e.g. 555 for concrete, 3000 for rock).
    The velocity that produces the sharpest image is the correct one.
    
    Args:
        Y_matrix: Displacement history
        sub_ap_centers: Sub-aperture centers
        radar_params: Metadata
        fixed_freq_hz: The optimal frequency found in the previous step
        velocity_range: Range of m/s to test
        steps: Number of velocities to test
        
    Returns:
        best_velocity: The velocity that maximizes image sharpness.
    """
    print(f"\n--- Starting Phase 3c: Automatic Depth Calibration (Velocity Autofocus) ---", flush=True)
    print(f"    Sweeping {velocity_range[0]} m/s to {velocity_range[1]} m/s at {fixed_freq_hz:.1f} Hz...", flush=True)
    
    velocities = np.linspace(velocity_range[0], velocity_range[1], steps)
    sharpness_scores = []
    
    # Geometry Prep
    num_pixels, num_looks = Y_matrix.shape
    z_vec_low_res = np.linspace(0, 50, 32)
    
    az_res = radar_params.get('azimuth_resolution_m', 1.0)
    center_idx = sub_ap_centers[num_looks // 2]
    b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0)
    
    slant_range = radar_params.get('slant_range_m', 10000)
    inc_angle = radar_params.get('incidence_angle_rad', 0.5)
    
    # Iterate Velocities
    for vel in velocities:
        # 1. Calculate Wavelength
        wavelength = vel / fixed_freq_hz
        
        # 2. Construct A Matrix
        denom = wavelength * slant_range * np.sin(inc_angle)
        if abs(denom) < 1e-9: denom = 1.0
        kz_vec = (4 * np.pi * b_perp_vec) / denom
        
        A_base = np.exp(1j * np.outer(kz_vec, z_vec_low_res))
        
        # 3. Fast Inversion
        tomogram_slice = np.abs(np.sum(Y_matrix[:, :, np.newaxis] * A_base.conj()[np.newaxis, :, :], axis=1))
        
        # 4. Measure Sharpness
        score = _calculate_image_sharpness(tomogram_slice)
        sharpness_scores.append(score)
        # print(f"    Vel: {vel:.0f} m/s -> Sharpness: {score:.4f}")
        
    # Find winner
    best_idx = np.argmax(sharpness_scores)
    best_vel = velocities[best_idx]
    
    # Identification
    material = "Unknown"
    if 300 <= best_vel <= 600: material = "Reinforced Concrete"
    elif 1400 <= best_vel <= 1600: material = "Water/Wet Soil"
    elif 2500 <= best_vel <= 3500: material = "Limestone/Brick"
    elif 4000 <= best_vel <= 6000: material = "Granite/Steel"
    
    print(f"    ✅ Optimal Velocity Found: {best_vel:.0f} m/s", flush=True)
    print(f"    🔍 Material ID: {material}", flush=True)
    
    return best_vel

# --- NEW VECTORIZED VELOCITY AUTOFOCUS FUNCTION ---
def perform_velocity_autofocus_vectorized(Y_matrix, sub_ap_centers, radar_params, 
                                         fixed_freq_hz, 
                                         velocity_range=(300.0, 6000.0), 
                                         steps=20):
    """
    VECTORIZED version of velocity autofocus for better performance.
    Processes all velocities in parallel using broadcasting.
    
    Args:
        Y_matrix: Displacement history [Pixels x Looks]
        sub_ap_centers: Sub-aperture centers
        radar_params: Metadata
        fixed_freq_hz: The optimal frequency found in the previous step
        velocity_range: Range of m/s to test
        steps: Number of velocities to test
        
    Returns:
        best_velocity: The velocity that maximizes image sharpness.
        sharpness_scores: Array of sharpness scores for all tested velocities
        velocity_map: 2D array of sharpness scores [velocities x pixels] (optional)
    """
    print(f"\n--- Starting Phase 3c: VECTORIZED Velocity Autofocus ---", flush=True)
    print(f"    Sweeping {velocity_range[0]} m/s to {velocity_range[1]} m/s at {fixed_freq_hz:.1f} Hz...", flush=True)
    
    # Generate velocity array
    velocities = np.linspace(velocity_range[0], velocity_range[1], steps)
    
    # Geometry parameters
    num_pixels, num_looks = Y_matrix.shape
    z_vec_low_res = np.linspace(0, 50, 32)  # Low res depth for speed
    
    az_res = radar_params.get('azimuth_resolution_m', 1.0)
    center_idx = sub_ap_centers[num_looks // 2]
    b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0)
    
    slant_range = radar_params.get('slant_range_m', 10000)
    inc_angle = radar_params.get('incidence_angle_rad', 0.5)
    
    # Pre-calculate constants that don't depend on velocity
    sin_inc = np.sin(inc_angle)
    if abs(sin_inc) < 1e-9: sin_inc = 1.0
    denominator_constant = slant_range * sin_inc
    
    # --- VECTORIZED CALCULATION ---
    # Calculate wavelengths for all velocities at once
    wavelengths = velocities / fixed_freq_hz  # [steps]
    
    # Calculate kz vectors for all velocities [steps x looks]
    # kz = (4 * pi * b_perp) / (wavelength * slant_range * sin(inc))
    kz_vectors = (4 * np.pi * b_perp_vec) / (wavelengths[:, np.newaxis] * denominator_constant)  # [steps x looks]
    
    # Calculate phase matrix for all velocities and depths [steps x looks x depths]
    phase_matrix = np.exp(1j * kz_vectors[:, :, np.newaxis] * z_vec_low_res[np.newaxis, np.newaxis, :])  # [steps x looks x depths]
    
    # Reshape Y_matrix for broadcasting [steps x pixels x looks]
    # We need to repeat Y_matrix for each velocity step
    Y_reshaped = Y_matrix.T[np.newaxis, :, :]  # [1 x looks x pixels]
    Y_reshaped = np.repeat(Y_reshaped, steps, axis=0)  # [steps x looks x pixels]
    Y_reshaped = Y_reshaped.transpose(0, 2, 1)  # [steps x pixels x looks]
    
    # Calculate tomograms for all velocities [steps x pixels x depths]
    # Using einsum for efficient computation: sum over looks dimension
    # tomograms = sum_looks(Y * conj(A)) -> [steps x pixels x depths]
    tomograms = np.einsum('spi,sid->spd', 
                          Y_reshaped,                   # [steps x pixels x looks]
                          np.conj(phase_matrix),        # [steps x looks x depths]
                          optimize=True)               # [steps x pixels x depths]
    
    # Calculate magnitudes [steps x pixels x depths]
    tomogram_magnitudes = np.abs(tomograms)
    
    # Calculate sharpness scores for each velocity [steps]
    sharpness_scores = np.zeros(steps)
    
    for v_idx in range(steps):
        # Get tomogram for this velocity [pixels x depths]
        vel_tomogram = tomogram_magnitudes[v_idx]
        
        # Calculate sharpness
        magnitude = np.abs(vel_tomogram)
        total_energy = np.sum(magnitude) + 1e-9
        prob_dist = magnitude / total_energy
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
        sharpness_scores[v_idx] = -entropy  # Higher is better
    
    # Find best velocity
    best_idx = np.argmax(sharpness_scores)
    best_vel = velocities[best_idx]
    
    # Material identification
    material = "Unknown"
    if 300 <= best_vel <= 600: material = "Reinforced Concrete"
    elif 1400 <= best_vel <= 1600: material = "Water/Wet Soil"
    elif 2500 <= best_vel <= 3500: material = "Limestone/Brick"
    elif 4000 <= best_vel <= 6000: material = "Granite/Steel"
    
    print(f"    ✅ Vectorized Autofocus Complete", flush=True)
    print(f"    ✅ Optimal Velocity Found: {best_vel:.0f} m/s", flush=True)
    print(f"    🔍 Material ID: {material}", flush=True)
    print(f"    ⚡ Processing Time: Vectorized over {steps} velocities", flush=True)
    
    return best_vel, sharpness_scores, tomogram_magnitudes

# --- NEW DIAGNOSTICS FUNCTION ---
def compute_autofocus_diagnostics(velocities, sharpness_scores, best_vel, Y_matrix_shape=None):
    """
    Compute quality metrics for velocity autofocus results.
    
    Args:
        velocities: Array of tested velocities
        sharpness_scores: Array of corresponding sharpness scores
        best_vel: Selected best velocity
        Y_matrix_shape: Shape of input data for SNR calculation (optional)
        
    Returns:
        diagnostics: Dictionary with quality metrics
    """
    if len(sharpness_scores) == 0 or len(velocities) == 0:
        return {"error": "No data for diagnostics"}
    
    diagnostics = {}
    
    # 1. Basic statistics
    max_score = np.max(sharpness_scores)
    min_score = np.min(sharpness_scores)
    mean_score = np.mean(sharpness_scores)
    std_score = np.std(sharpness_scores)
    
    diagnostics['sharpness_max'] = float(max_score)
    diagnostics['sharpness_min'] = float(min_score)
    diagnostics['sharpness_mean'] = float(mean_score)
    diagnostics['sharpness_std'] = float(std_score)
    
    # 2. Signal-to-Noise Ratio of the sharpness peak
    if std_score > 0:
        peak_snr = (max_score - mean_score) / std_score
        diagnostics['peak_snr_db'] = float(10 * np.log10(peak_snr + 1e-9))
    else:
        diagnostics['peak_snr_db'] = 0.0
    
    # 3. Find peak width at half maximum
    half_max = (max_score + min_score) / 2
    above_half = sharpness_scores > half_max
    
    if np.any(above_half):
        # Find first and last indices above half max
        indices = np.where(above_half)[0]
        first_idx = indices[0]
        last_idx = indices[-1]
        
        # Convert to velocity range
        velocity_range_half_max = velocities[last_idx] - velocities[first_idx]
        relative_width = velocity_range_half_max / (velocities[-1] - velocities[0])
        
        diagnostics['half_max_width_mps'] = float(velocity_range_half_max)
        diagnostics['half_max_relative_width'] = float(relative_width)
        
        # Quality assessment based on peak width
        if relative_width < 0.1:
            diagnostics['peak_quality'] = "Excellent (Sharp Peak)"
        elif relative_width < 0.25:
            diagnostics['peak_quality'] = "Good (Well-Defined)"
        elif relative_width < 0.5:
            diagnostics['peak_quality'] = "Fair (Broad Peak)"
        else:
            diagnostics['peak_quality'] = "Poor (No Distinct Peak)"
    else:
        diagnostics['half_max_width_mps'] = float(velocities[-1] - velocities[0])
        diagnostics['half_max_relative_width'] = 1.0
        diagnostics['peak_quality'] = "Poor (No Clear Peak)"
    
    # 4. Confidence score (0-100)
    if std_score > 0:
        # Based on peak SNR and narrowness
        peak_prominence = (max_score - mean_score) / std_score
        if 'half_max_relative_width' in diagnostics:
            narrowness = 1.0 - diagnostics['half_max_relative_width']
        else:
            narrowness = 0.5
        
        confidence = 100 * peak_prominence * narrowness / (peak_prominence * narrowness + 1)
        diagnostics['confidence_score'] = float(np.clip(confidence, 0, 100))
    else:
        diagnostics['confidence_score'] = 0.0
    
    # 5. Check if best velocity is at boundary (problematic)
    vel_tolerance = 0.05 * (velocities[-1] - velocities[0])  # 5% tolerance
    at_lower_bound = abs(best_vel - velocities[0]) < vel_tolerance
    at_upper_bound = abs(best_vel - velocities[-1]) < vel_tolerance
    
    diagnostics['at_lower_bound'] = bool(at_lower_bound)
    diagnostics['at_upper_bound'] = bool(at_upper_bound)
    
    if at_lower_bound or at_upper_bound:
        diagnostics['boundary_warning'] = "Best velocity at search boundary - consider expanding range"
    else:
        diagnostics['boundary_warning'] = "None"
    
    # 6. Material consistency check
    material_suggestions = []
    if 300 <= best_vel <= 600: 
        material_suggestions.append("Reinforced Concrete (300-600 m/s)")
    if 1400 <= best_vel <= 1600: 
        material_suggestions.append("Water/Wet Soil (1400-1600 m/s)")
    if 2500 <= best_vel <= 3500: 
        material_suggestions.append("Limestone/Brick (2500-3500 m/s)")
    if 4000 <= best_vel <= 6000: 
        material_suggestions.append("Granite/Steel (4000-6000 m/s)")
    
    if not material_suggestions:
        material_suggestions.append(f"Unusual Velocity ({best_vel:.0f} m/s)")
    
    diagnostics['material_suggestions'] = material_suggestions
    
    # 7. Data quality indicators (if Y_matrix shape provided)
    if Y_matrix_shape is not None:
        num_pixels, num_looks = Y_matrix_shape
        diagnostics['num_pixels'] = int(num_pixels)
        diagnostics['num_looks'] = int(num_looks)
        diagnostics['data_density'] = float(num_pixels * num_looks / 1000)  # Thousands of samples
    
    return diagnostics

# --- NEW ENHANCED FUNCTION WITH DIAGNOSTICS ---
def perform_velocity_autofocus_with_diagnostics(Y_matrix, sub_ap_centers, radar_params, 
                                               fixed_freq_hz, 
                                               velocity_range=(300.0, 6000.0), 
                                               steps=20, 
                                               return_diagnostics=False,
                                               use_vectorized=True):
    """
    Enhanced version of velocity autofocus with optional diagnostics.
    
    Args:
        Y_matrix: Displacement history [Pixels x Looks]
        sub_ap_centers: Sub-aperture centers
        radar_params: Metadata
        fixed_freq_hz: The optimal frequency found in the previous step
        velocity_range: Range of m/s to test
        steps: Number of velocities to test
        return_diagnostics: If True, returns diagnostics dictionary
        use_vectorized: If True, uses vectorized implementation (faster)
        
    Returns:
        best_velocity: The velocity that maximizes image sharpness.
        diagnostics: Dictionary with quality metrics (if return_diagnostics=True)
    """
    print(f"\n--- Starting Phase 3c: Enhanced Velocity Autofocus with Diagnostics ---", flush=True)
    
    # Choose implementation
    if use_vectorized:
        print(f"    Using VECTORIZED implementation...", flush=True)
        best_vel, sharpness_scores, tomogram_magnitudes = perform_velocity_autofocus_vectorized(
            Y_matrix, sub_ap_centers, radar_params, fixed_freq_hz, velocity_range, steps
        )
    else:
        print(f"    Using STANDARD implementation...", flush=True)
        # Use standard implementation and collect scores
        velocities = np.linspace(velocity_range[0], velocity_range[1], steps)
        sharpness_scores = []
        
        # Geometry Prep
        num_pixels, num_looks = Y_matrix.shape
        z_vec_low_res = np.linspace(0, 50, 32)
        
        az_res = radar_params.get('azimuth_resolution_m', 1.0)
        center_idx = sub_ap_centers[num_looks // 2]
        b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0)
        
        slant_range = radar_params.get('slant_range_m', 10000)
        inc_angle = radar_params.get('incidence_angle_rad', 0.5)
        
        # Iterate Velocities
        for vel in velocities:
            # 1. Calculate Wavelength
            wavelength = vel / fixed_freq_hz
            
            # 2. Construct A Matrix
            denom = wavelength * slant_range * np.sin(inc_angle)
            if abs(denom) < 1e-9: denom = 1.0
            kz_vec = (4 * np.pi * b_perp_vec) / denom
            
            A_base = np.exp(1j * np.outer(kz_vec, z_vec_low_res))
            
            # 3. Fast Inversion
            tomogram_slice = np.abs(np.sum(Y_matrix[:, :, np.newaxis] * A_base.conj()[np.newaxis, :, :], axis=1))
            
            # 4. Measure Sharpness
            score = _calculate_image_sharpness(tomogram_slice)
            sharpness_scores.append(score)
        
        # Find best velocity
        best_idx = np.argmax(sharpness_scores)
        best_vel = velocities[best_idx]
        sharpness_scores = np.array(sharpness_scores)
    
    # Material identification
    material = "Unknown"
    if 300 <= best_vel <= 600: material = "Reinforced Concrete"
    elif 1400 <= best_vel <= 1600: material = "Water/Wet Soil"
    elif 2500 <= best_vel <= 3500: material = "Limestone/Brick"
    elif 4000 <= best_vel <= 6000: material = "Granite/Steel"
    
    print(f"    ✅ Optimal Velocity Found: {best_vel:.0f} m/s", flush=True)
    print(f"    🔍 Material ID: {material}", flush=True)
    
    if return_diagnostics:
        print(f"    📊 Computing diagnostics...", flush=True)
        # Generate velocities array for diagnostics
        velocities = np.linspace(velocity_range[0], velocity_range[1], steps)
        
        # Compute diagnostics
        diagnostics = compute_autofocus_diagnostics(
            velocities, sharpness_scores, best_vel, Y_matrix.shape
        )
        
        # Print key diagnostics
        print(f"    📈 Peak Quality: {diagnostics.get('peak_quality', 'Unknown')}")
        print(f"    🎯 Confidence Score: {diagnostics.get('confidence_score', 0):.1f}/100")
        if 'peak_snr_db' in diagnostics:
            print(f"    📶 Peak SNR: {diagnostics['peak_snr_db']:.1f} dB")
        
        if diagnostics.get('at_lower_bound', False) or diagnostics.get('at_upper_bound', False):
            print(f"    ⚠️  {diagnostics.get('boundary_warning', '')}")
        
        return best_vel, diagnostics
    else:
        return best_vel

# --- Original Advanced Functions (Kept for compatibility) ---

def _calculate_robustness_score(Y_pixel, A_layered, h_value, epsilon):
    if h_value is None or h_value.size == 0: return 0.0
    residual_vector = Y_pixel - A_layered @ h_value
    E_res = np.linalg.norm(residual_vector, ord=2)
    E_worst = 1.5 * epsilon
    if E_res <= epsilon: return 10.0 
    if E_res >= E_worst: return 0.0 
    score = 10.0 * (E_worst - E_res) / (E_worst - epsilon)
    return max(0, min(10.0, score))

def _focus_vsa(Y, sub_ap_centers, radar_params, seismic_wavelength_range):
    """ Legacy VSA function """
    return np.zeros_like(Y), np.zeros(10), np.zeros(10)

def _focus_layered_inversion(Y, sub_ap_centers, radar_params, layer_velocities):
    """ Legacy Layered function """
    return np.zeros_like(Y), np.zeros(10)
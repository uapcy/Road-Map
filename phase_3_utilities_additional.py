"""
phase_3_utilities.py
Utility functions for 3D tomography processing.
All new features that don't modify existing code.
"""

import numpy as np
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Try importing optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization functions will be limited.")

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

TOMOGRAPHY_PRESETS = {
    'building': {
        'physics_model': 'seismic',
        'final_method': 'cs',
        'smoothness_weight': 0.1,
        'z_min': -10,
        'z_max': 50,
        'epsilon': 0.05,
        'seismic_velocity_ms': 555.0,
        'vibration_frequency_hz': 50.0,
        'target_type': 'building'
    },
    'geology': {
        'physics_model': 'seismic',
        'final_method': 'capon',
        'smoothness_weight': 0.3,
        'z_min': 0,
        'z_max': 200,
        'epsilon': 0.1,
        'seismic_velocity_ms': 3000.0,
        'vibration_frequency_hz': 20.0,
        'target_type': 'geology'
    },
    'orbital_sar': {
        'physics_model': 'orbital',
        'final_method': 'capon',
        'smoothness_weight': 0.05,
        'z_min': -50,
        'z_max': 100,
        'epsilon': 0.2,
        'seismic_velocity_ms': 1500.0,
        'vibration_frequency_hz': 50.0,
        'target_type': 'geology'
    },
    'bridge': {
        'physics_model': 'seismic',
        'final_method': 'beamforming',
        'smoothness_weight': 0.05,
        'z_min': -20,
        'z_max': 30,
        'epsilon': 0.15,
        'seismic_velocity_ms': 5000.0,
        'vibration_frequency_hz': 100.0,
        'target_type': 'bridge'
    }
}

def get_preset_config(preset_name: str, **kwargs) -> Dict[str, Any]:
    """
    Get configuration for a specific application.
    
    Args:
        preset_name: Name of preset ('building', 'geology', 'orbital_sar', 'bridge')
        **kwargs: Override parameters
        
    Returns:
        dict: Configuration dictionary
    """
    if preset_name not in TOMOGRAPHY_PRESETS:
        available = list(TOMOGRAPHY_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    config = TOMOGRAPHY_PRESETS[preset_name].copy()
    config.update(kwargs)
    return config

def save_preset_config(preset_name: str, config: Dict[str, Any], 
                      filepath: Optional[str] = None) -> str:
    """
    Save a preset configuration to JSON file.
    
    Args:
        preset_name: Name of the preset
        config: Configuration dictionary
        filepath: Path to save file (None = preset_name.json)
        
    Returns:
        str: Path to saved file
    """
    if filepath is None:
        filepath = f"{preset_name}_config.json"
    
    with open(filepath, 'w') as f:
        json.dump({preset_name: config}, f, indent=4)
    
    return filepath

def load_preset_config(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load preset configuration from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        dict: Loaded configurations
    """
    with open(filepath, 'r') as f:
        return json.load(f)

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def validate_tomography_params(radar_params: Dict[str, Any], 
                              physics_model: str) -> Tuple[bool, List[str]]:
    """
    Validate all required parameters before processing.
    
    Args:
        radar_params: Radar parameters dictionary
        physics_model: 'seismic' or 'orbital'
        
    Returns:
        tuple: (is_valid, list_of_warnings)
    """
    warnings = []
    
    if physics_model == 'orbital':
        required = ['center_frequency', 'col_sample_spacing']
        missing = []
        for param in required:
            if param not in radar_params:
                missing.append(param)
        
        if missing:
            return False, [f"Missing orbital parameters: {missing}"]
    
    # Additional validations
    if 'effective_prf' not in radar_params and 'prf_stats' not in radar_params:
        warnings.append("PRF information not found. Using default value.")
    
    if 'azimuth_spacing_m' not in radar_params:
        warnings.append("Azimuth spacing not specified. Using default 1.0 m.")
    
    return True, warnings

def validate_orbital_params(radar_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and extract orbital parameters with defaults.
    
    Args:
        radar_params: Radar parameters dictionary
        
    Returns:
        dict: Validated orbital parameters
    """
    orbital_params = {}
    
    # Required parameters
    required = ['center_frequency', 'col_sample_spacing']
    for param in required:
        if param in radar_params:
            orbital_params[param] = radar_params[param]
        else:
            raise ValueError(f"Missing required orbital parameter: {param}")
    
    # Optional parameters with defaults
    if 'slant_range_m' in radar_params:
        orbital_params['slant_range_m'] = radar_params['slant_range_m']
    else:
        orbital_params['slant_range_m'] = 600000.0  # Default 600 km
        warnings.warn(f"Using default slant range: {orbital_params['slant_range_m']} m")
    
    if 'incidence_angle_rad' in radar_params:
        orbital_params['incidence_angle_rad'] = radar_params['incidence_angle_rad']
    elif 'incidence_angle' in radar_params:
        orbital_params['incidence_angle_rad'] = np.radians(radar_params['incidence_angle'])
    else:
        orbital_params['incidence_angle_rad'] = np.radians(40.09)  # Capella default
        warnings.warn(f"Using default incidence angle: {np.degrees(orbital_params['incidence_angle_rad']):.2f}°")
    
    return orbital_params

# =============================================================================
# MODEL SELECTION GUIDANCE
# =============================================================================

def recommend_model(Y_matrix_processed: np.ndarray, 
                   sub_ap_centers: np.ndarray,
                   radar_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recommend physics model and inversion method based on input characteristics.
    
    Args:
        Y_matrix_processed: [Num_Pixels x Num_Looks] processed data
        sub_ap_centers: Sub-aperture centers
        radar_params: Optional radar parameters for context
        
    Returns:
        dict: Recommended settings with explanations
    """
    num_looks = len(sub_ap_centers)
    num_pixels = Y_matrix_processed.shape[0]
    
    # Analyze coherence (signal quality)
    if num_pixels > 1 and num_looks > 1:
        try:
            corr_matrix = np.corrcoef(Y_matrix_processed.T)
            coherence = np.mean(np.abs(corr_matrix))
        except:
            coherence = 0.5  # Default if calculation fails
    else:
        coherence = 0.5  # Default for small datasets
    
    # Analyze energy distribution
    energy = np.sum(np.abs(Y_matrix_processed)**2)
    avg_energy = energy / (num_pixels * num_looks) if (num_pixels * num_looks) > 0 else 0
    
    # Make recommendations
    recommendations = {
        'coherence_score': float(coherence),
        'avg_energy': float(avg_energy),
        'num_looks': num_looks,
        'num_pixels': num_pixels
    }
    
    # Method recommendation
    if coherence > 0.7 and num_looks > 8:
        recommendations['final_method'] = 'cs'
        recommendations['method_reason'] = 'High coherence and sufficient looks for CS'
        recommendations['method_confidence'] = 'high'
    elif coherence > 0.5:
        recommendations['final_method'] = 'capon'
        recommendations['method_reason'] = 'Moderate coherence suitable for Capon'
        recommendations['method_confidence'] = 'medium'
    else:
        recommendations['final_method'] = 'beamforming'
        recommendations['method_reason'] = 'Low coherence requires robust beamforming'
        recommendations['method_confidence'] = 'high'
    
    # Physics model suggestion
    if radar_params is not None:
        # Check if orbital parameters are available
        has_orbital_params = all(p in radar_params for p in ['center_frequency', 'col_sample_spacing'])
        
        if has_orbital_params and coherence > 0.7:
            recommendations['physics_model'] = 'orbital'
            recommendations['physics_reason'] = 'Orbital parameters available and high coherence'
        else:
            recommendations['physics_model'] = 'seismic'
            if not has_orbital_params:
                recommendations['physics_reason'] = 'Orbital parameters not available'
            else:
                recommendations['physics_reason'] = 'Lower coherence suggests seismic/vibration'
    else:
        recommendations['physics_model'] = 'seismic'
        recommendations['physics_reason'] = 'No radar parameters provided for orbital model'
    
    # Target type suggestion based on context
    if avg_energy > 1.0:
        recommendations['target_type'] = 'building'
    elif avg_energy > 0.1:
        recommendations['target_type'] = 'geology'
    else:
        recommendations['target_type'] = 'bridge'
    
    return recommendations

# =============================================================================
# ADAPTIVE PARAMETER TUNING
# =============================================================================

def auto_tune_epsilon(Y: np.ndarray, A: np.ndarray, 
                     method: str = 'median') -> float:
    """
    Automatically determine epsilon for CS based on noise level.
    
    Args:
        Y: Measurement vector [Looks]
        A: Steering matrix [Looks x Depth]
        method: Tuning method ('median', 'svd', 'percentile')
        
    Returns:
        float: Optimal epsilon value
    """
    if method == 'median':
        # Simple median-based estimation
        noise_estimate = np.median(np.abs(Y)) * np.sqrt(len(Y))
    
    elif method == 'svd':
        # SVD-based noise estimation
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        if len(s) > 0:
            noise_estimate = np.median(np.abs(Y)) * np.sqrt(len(Y)) * (s[-1] / s[0])
        else:
            noise_estimate = np.median(np.abs(Y)) * np.sqrt(len(Y))
    
    elif method == 'percentile':
        # Percentile-based robust estimation
        abs_Y = np.abs(Y)
        noise_estimate = np.percentile(abs_Y, 25) * np.sqrt(len(Y))
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median', 'svd', or 'percentile'")
    
    # Heuristic: epsilon = noise_level * sqrt(2 * log(n))
    n = len(Y)
    epsilon = noise_estimate * np.sqrt(2 * np.log(n))
    
    # Ensure reasonable bounds
    epsilon = max(epsilon, 0.01)  # Minimum threshold
    epsilon = min(epsilon, 1.0)   # Maximum threshold
    
    return float(epsilon)

def auto_tune_damping_coeff(Y_matrix: np.ndarray, 
                           condition_threshold: float = 1e6) -> float:
    """
    Auto-tune damping coefficient based on data conditioning.
    
    Args:
        Y_matrix: Input data matrix
        condition_threshold: Threshold for considering matrix ill-conditioned
        
    Returns:
        float: Recommended damping coefficient
    """
    if Y_matrix.ndim != 2:
        return 0.1  # Default
    
    # Estimate condition from covariance
    try:
        R = Y_matrix @ Y_matrix.conj().T
        cond_number = np.linalg.cond(R)
        
        # Heuristic damping based on condition number
        if cond_number > condition_threshold:
            damping = 0.5  # High damping for ill-conditioned
        elif cond_number > 1e4:
            damping = 0.2  # Medium damping
        elif cond_number > 1e2:
            damping = 0.1  # Low damping
        else:
            damping = 0.05  # Very low damping for well-conditioned
    except:
        damping = 0.1  # Default if calculation fails
    
    return damping

# =============================================================================
# DIAGNOSTIC OUTPUTS
# =============================================================================

def compute_diagnostics(Y: np.ndarray, A: np.ndarray, 
                       x_est: np.ndarray) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        Y: Original measurement vector
        A: Steering matrix
        x_est: Estimated reflectivity profile
        
    Returns:
        dict: Diagnostic metrics
    """
    residuals = Y - A @ x_est
    
    # Root Mean Square Error
    rmse = float(np.sqrt(np.mean(np.abs(residuals)**2)))
    
    # Signal-to-Noise Ratio (dB)
    signal_power = float(np.mean(np.abs(Y)**2))
    noise_power = float(np.mean(np.abs(residuals)**2))
    
    if noise_power > 0:
        snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_db = float('inf')
    
    # Sparsity metrics
    abs_x = np.abs(x_est)
    if len(abs_x) > 0:
        max_val = np.max(abs_x)
        sparsity = float(np.sum(abs_x > 0.1 * max_val) / len(x_est))
    else:
        sparsity = 0.0
    
    # Condition number of A (numerical stability)
    try:
        cond_number = float(np.linalg.cond(A))
    except:
        cond_number = float('inf')
    
    # Reconstruction energy ratio
    recon_energy = float(np.sum(np.abs(A @ x_est)**2))
    orig_energy = float(np.sum(np.abs(Y)**2))
    
    if orig_energy > 0:
        energy_ratio = float(recon_energy / orig_energy)
    else:
        energy_ratio = 0.0
    
    return {
        'rmse': rmse,
        'snr_db': snr_db,
        'sparsity': sparsity,
        'condition_number': cond_number,
        'energy_ratio': energy_ratio,
        'residual_norm': float(np.linalg.norm(residuals)),
        'signal_power': signal_power,
        'noise_power': noise_power
    }

def compute_batch_diagnostics(diagnostics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate diagnostics from multiple pixels.
    
    Args:
        diagnostics_list: List of per-pixel diagnostics
        
    Returns:
        dict: Aggregated statistics
    """
    if not diagnostics_list:
        return {}
    
    aggregated = {
        'num_pixels': len(diagnostics_list),
        'average_rmse': np.mean([d.get('rmse', 0) for d in diagnostics_list]),
        'average_snr_db': np.mean([d.get('snr_db', 0) for d in diagnostics_list if np.isfinite(d.get('snr_db', 0))]),
        'average_sparsity': np.mean([d.get('sparsity', 0) for d in diagnostics_list]),
        'median_rmse': np.median([d.get('rmse', 0) for d in diagnostics_list]),
        'std_rmse': np.std([d.get('rmse', 0) for d in diagnostics_list]),
        'min_snr_db': np.min([d.get('snr_db', 0) for d in diagnostics_list if np.isfinite(d.get('snr_db', 0))]),
        'max_snr_db': np.max([d.get('snr_db', 0) for d in diagnostics_list if np.isfinite(d.get('snr_db', 0))])
    }
    
    return aggregated

# =============================================================================
# VISUALIZATION HELPER
# =============================================================================

def plot_tomogram(tomogram: np.ndarray, z_vec: np.ndarray, 
                 pixel_positions: Optional[np.ndarray] = None,
                 title: str = "Tomogram", figsize: Tuple[int, int] = (12, 8),
                 cmap: str = 'viridis', save_path: Optional[str] = None,
                 show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Visualize tomographic results.
    
    Args:
        tomogram: [Num_Pixels x Num_Z] tomogram amplitude
        z_vec: Depth vector
        pixel_positions: Optional pixel positions for x-axis
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure (None = don't save)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib.figure.Figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Matplotlib not available for visualization.")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Prepare x-axis
    if pixel_positions is None:
        pixel_positions = np.arange(tomogram.shape[0])
    
    # Plot 1: Amplitude
    im1 = axes[0].imshow(np.abs(tomogram).T, aspect='auto',
                        extent=[pixel_positions[0], pixel_positions[-1], 
                                z_vec[-1], z_vec[0]],
                        cmap=cmap, origin='lower')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_xlabel('Pixel Position')
    axes[0].set_title(f'{title} - Amplitude')
    plt.colorbar(im1, ax=axes[0], label='Amplitude')
    
    # Plot 2: Phase (if complex)
    if np.iscomplexobj(tomogram):
        im2 = axes[1].imshow(np.angle(tomogram).T, aspect='auto',
                            extent=[pixel_positions[0], pixel_positions[-1], 
                                    z_vec[-1], z_vec[0]],
                            cmap='hsv', vmin=-np.pi, vmax=np.pi, origin='lower')
        axes[1].set_ylabel('Depth (m)')
        axes[1].set_xlabel('Pixel Position')
        axes[1].set_title('Phase')
        plt.colorbar(im2, ax=axes[1], label='Phase (rad)')
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Real-valued tomogram\n(no phase information)',
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig

def plot_diagnostics(diagnostics: Dict[str, Any], 
                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot diagnostic metrics.
    
    Args:
        diagnostics: Diagnostic dictionary
        save_path: Path to save figure
        
    Returns:
        matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    metrics_to_plot = ['rmse', 'snr_db', 'sparsity', 'energy_ratio']
    available_metrics = [m for m in metrics_to_plot if m in diagnostics]
    
    if not available_metrics:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        value = diagnostics[metric]
        
        if isinstance(value, (int, float)):
            ax.bar([0], [value])
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric}: {value:.4f}')
            ax.set_xticks([])
        elif isinstance(value, list) and len(value) > 0:
            ax.hist(value, bins=20, alpha=0.7)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric} Distribution')
    
    # Hide unused subplots
    for i in range(len(available_metrics), 4):
        axes[i].axis('off')
    
    plt.suptitle('Tomography Diagnostics', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# =============================================================================
# MEMORY MANAGEMENT (Simple version)
# =============================================================================

def estimate_batch_size(data_shape: Tuple[int, int, int], 
                       dtype: np.dtype = np.complex64,
                       available_memory_gb: float = 2.0) -> int:
    """
    Estimate optimal batch size based on available memory.
    
    Args:
        data_shape: Shape of 3D data (rows, cols, looks)
        dtype: Data type
        available_memory_gb: Available memory in GB
        
    Returns:
        int: Recommended batch size (number of columns)
    """
    rows, cols, looks = data_shape
    
    # Element size in bytes
    if dtype == np.complex64:
        element_size = 8
    elif dtype == np.float32:
        element_size = 4
    elif dtype == np.float64:
        element_size = 8
    elif dtype == np.complex128:
        element_size = 16
    else:
        element_size = 8  # Default
    
    # Memory for one column
    column_memory = rows * looks * element_size
    
    # Memory for output (estimate depth similar to rows)
    output_memory = rows * rows * element_size
    
    # Total per column with overhead
    total_per_column = (column_memory + output_memory) * 1.5
    
    # Convert available memory to bytes
    available_bytes = available_memory_gb * 1024**3
    
    # Calculate batch size
    batch_size = max(1, int(available_bytes // total_per_column))
    batch_size = min(batch_size, cols)
    batch_size = max(batch_size, min(10, cols))
    
    return batch_size

def process_in_batches(data_3d: np.ndarray, 
                      processing_func: callable,
                      batch_size: Optional[int] = None,
                      **kwargs) -> List[Any]:
    """
    Simple batch processing wrapper.
    
    Args:
        data_3d: 3D input data [rows, cols, looks]
        processing_func: Function to apply to each batch
        batch_size: Columns per batch (None = auto)
        **kwargs: Arguments for processing_func
        
    Returns:
        list: Results from each batch
    """
    rows, cols, looks = data_3d.shape
    
    if batch_size is None:
        batch_size = estimate_batch_size(data_3d.shape, data_3d.dtype)
    
    results = []
    
    for batch_start in range(0, cols, batch_size):
        batch_end = min(batch_start + batch_size, cols)
        batch_data = data_3d[:, batch_start:batch_end, :]
        
        print(f"Processing columns {batch_start}-{batch_end-1} "
              f"({batch_data.shape[1]} columns)...")
        
        batch_result = processing_func(batch_data, **kwargs)
        results.append(batch_result)
    
    return results

# =============================================================================
# EXPORT/SAVE UTILITIES
# =============================================================================

def save_tomogram_results(tomogram: np.ndarray, z_vec: np.ndarray,
                         radar_params: Dict[str, Any],
                         output_path: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save tomogram results to NPZ file with metadata.
    
    Args:
        tomogram: Tomogram data
        z_vec: Depth vector
        radar_params: Radar parameters
        output_path: Output file path
        metadata: Additional metadata
        
    Returns:
        str: Path to saved file
    """
    save_dict = {
        'tomogram': tomogram,
        'z_vec': z_vec,
        'radar_params': radar_params,
        'save_timestamp': np.datetime64('now')
    }
    
    if metadata:
        save_dict.update(metadata)
    
    np.savez_compressed(output_path, **save_dict)
    return output_path

def load_tomogram_results(filepath: str) -> Dict[str, Any]:
    """
    Load tomogram results from NPZ file.
    
    Args:
        filepath: Path to NPZ file
        
    Returns:
        dict: Loaded data
    """
    with np.load(filepath, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}
    
    return result

# =============================================================================
# QUICK TEST/EXAMPLE
# =============================================================================

def example_usage():
    """Example usage of utility functions."""
    print("Testing tomography utilities...")
    
    # Create synthetic data
    Y = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
    A = np.random.randn(50, 30) + 1j * np.random.randn(50, 30)
    x_est = np.random.randn(30) + 1j * np.random.randn(30)
    
    # Test diagnostics
    diag = compute_diagnostics(Y[:, 0], A, x_est)
    print(f"Diagnostics: {diag}")
    
    # Test auto-tuning
    epsilon = auto_tune_epsilon(Y[:, 0], A)
    print(f"Auto-tuned epsilon: {epsilon:.4f}")
    
    # Test model recommendation
    rec = recommend_model(Y, np.arange(50))
    print(f"Recommendations: {rec}")
    
    print("Example complete.")

if __name__ == "__main__":
    example_usage()
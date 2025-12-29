# Quick_Signal_Evaluator.py
# A standalone diagnostic tool to evaluate SAR signal quality before processing.
# Combines Static Image Analysis and Dynamic Sub-Aperture (Doppler) Checks.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fft, ifft, ifftshift
from scipy.stats import entropy
import sys
import os

# Import your existing infrastructure
try:
    from Ext_Data import get_external_data_paths
    from data_loader import load_mlc_data, parse_radar_parameters
    # We will reimplement a lightweight sub-aperture generator here to ensure it runs standalone
    # without circular dependency issues, but using the exact same math as Phase 2.
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure this script is in the same folder as your other files.\n{e}")
    sys.exit(1)

# --- CONFIGURATION ---
SAMPLE_SIZE = 2048       # Number of pixels to grab for the "Virtual Probe" line
NUM_LOOKS = 96           # Number of time-steps (Doppler frames) to generate
OVERLAP = 0.85           # Overlap factor (same as main processor)
CENTER_CROP_SIZE = 512   # For 2D Static checks

def calculate_dynamic_range(magnitude_data):
    """Measures the contrast between bright reflectors and the noise floor."""
    # Avoid log(0)
    valid_data = magnitude_data[magnitude_data > 0]
    if len(valid_data) == 0: return 0.0
    
    # Top 1% (Signal) vs Bottom 10% (Noise Floor)
    peak_signal = np.percentile(valid_data, 99.9)
    noise_floor = np.percentile(valid_data, 10)
    
    if noise_floor == 0: return 0.0
    
    # Dynamic Range in dB
    dr_db = 20 * np.log10(peak_signal / noise_floor)
    return dr_db

def calculate_focus_score(complex_patch):
    """
    Checks if the image is blurry by analyzing high-frequency content in FFT space.
    A focused SAR image has energy spread across frequencies. A blurry one is clustered in low frequencies.
    """
    f_transform = fftshift(fft2(complex_patch))
    magnitude_spectrum = np.abs(f_transform)
    
    # Calculate energy in the center (Low Freq) vs Outer edges (High Freq)
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    
    total_energy = np.sum(magnitude_spectrum)
    center_energy = np.sum(magnitude_spectrum[cy-radius:cy+radius, cx-radius:cx+radius])
    high_freq_energy = total_energy - center_energy
    
    # Ratio of high frequency content
    focus_ratio = high_freq_energy / total_energy
    return focus_ratio

def generate_diagnostic_sub_apertures(tomographic_line, num_looks, overlap):
    """
    Splits a single line of pixels into time-steps (Doppler/Sub-apertures).
    This creates the 'Cine Loop' for stability analysis.
    """
    # FFT to frequency domain
    spectrum = fftshift(fft(tomographic_line))
    total_bandwidth = len(spectrum)
    
    # Calculate look bandwidth
    look_bw = int(total_bandwidth / (num_looks * (1 - overlap) + overlap))
    step = int(look_bw * (1 - overlap))
    
    stack = []
    
    for i in range(num_looks):
        start = i * step
        end = start + look_bw
        
        if end > total_bandwidth: break
        
        # Extract sub-band
        sub_spectrum = spectrum[start:end]
        
        # Apply windowing (Hanning) to reduce sidelobes
        window = np.hanning(len(sub_spectrum))
        windowed_spectrum = sub_spectrum * window
        
        # Pad back to full size for resolution preservation
        padded = np.zeros(total_bandwidth, dtype=np.complex64)
        # Center the sub-spectrum in the pad
        pad_start = (total_bandwidth - len(sub_spectrum)) // 2
        padded[pad_start : pad_start+len(sub_spectrum)] = windowed_spectrum
        
        # IFFT back to spatial domain -> This is one "Frame" of our movie
        img = ifft(ifftshift(padded))
        stack.append(img)
        
    return np.array(stack) # Shape: (Num_Looks, Num_Pixels)

def analyze_temporal_stability(stack):
    """
    Evaluates the 'Heartbeat' of the signal.
    We compare Frame 1 to Frame 2, Frame 2 to Frame 3, etc.
    """
    # 1. Coherence (Correlation between adjacent looks)
    # Coherence = |E[x * y*]| / sqrt(E[|x|^2] * E[|y|^2])
    
    coherences = []
    phase_diffs = []
    
    for i in range(len(stack) - 1):
        frame_a = stack[i]
        frame_b = stack[i+1]
        
        # Simple correlation coefficient for the whole line
        num = np.abs(np.sum(frame_a * np.conj(frame_b)))
        den = np.sqrt(np.sum(np.abs(frame_a)**2) * np.sum(np.abs(frame_b)**2))
        
        if den > 0:
            coherences.append(num / den)
            
        # Phase stability check
        # We look at the phase difference of strong reflectors only
        strong_mask = np.abs(frame_a) > np.percentile(np.abs(frame_a), 90)
        if np.any(strong_mask):
            diff = np.angle(frame_a[strong_mask] * np.conj(frame_b[strong_mask]))
            # A stable signal should have a constant phase shift (due to geometry), variance should be low
            phase_std = np.std(diff)
            phase_diffs.append(phase_std)

    avg_coherence = np.mean(coherences) if coherences else 0.0
    avg_phase_instability = np.mean(phase_diffs) if phase_diffs else 999.0
    
    return avg_coherence, avg_phase_instability

def main():
    print("\n--- 🏥 SAR Signal Health Check (Evaluator) ---")
    
    # 1. Get Data
    paths = get_external_data_paths()
    if not paths: return
    
    params = parse_radar_parameters(paths['json_file'])
    print(f"\nLoading image: {paths['tiff_file']}...")
    complex_data = load_mlc_data(paths['tiff_file'], params)
    
    if complex_data is None:
        print("Error: Failed to load data.")
        return

    rows, cols = complex_data.shape
    print(f"Image Dimensions: {rows} x {cols}")
    
    # --- PART A: STATIC IMAGE CHECKS ---
    print("\n[Test A]: Analyzing Static Image Properties...")
    
    # Take a center crop for speed
    cr, cc = rows // 2, cols // 2
    crop_size = min(rows, cols, CENTER_CROP_SIZE)
    patch = complex_data[cr-crop_size//2 : cr+crop_size//2, cc-crop_size//2 : cc+crop_size//2]
    
    # 1. Dynamic Range
    dr_db = calculate_dynamic_range(np.abs(patch))
    
    # 2. Focus
    focus_score = calculate_focus_score(patch)
    
    # --- PART B: DYNAMIC (ULTRASOUND) CHECKS ---
    print("[Test B]: performing 'Virtual Probe' Analysis (Sub-Apertures)...")
    
    # Select a test line (Virtual Probe)
    # We pick the center column of our crop
    probe_line = patch[:, crop_size//2]
    
    # Generate the "Movie" (96 frames)
    stack = generate_diagnostic_sub_apertures(probe_line, NUM_LOOKS, OVERLAP)
    
    # Analyze the stability of the movie
    coherence, phase_instability = analyze_temporal_stability(stack)
    
    # --- REPORT CARD ---
    print("\n" + "="*40)
    print("      SIGNAL EVALUATION REPORT      ")
    print("="*40)
    
    print(f"1. DYNAMIC RANGE (Contrast):   {dr_db:.1f} dB")
    if dr_db > 30: print("   -> VERDICT: EXCELLENT (Strong targets visible)")
    elif dr_db > 15: print("   -> VERDICT: GOOD (Usable)")
    else: print("   -> VERDICT: POOR (Image looks flat/noisy)")
        
    print("-" * 40)
    
    print(f"2. FOCUS SCORE (Sharpness):    {focus_score:.3f}")
    if focus_score > 0.4: print("   -> VERDICT: SHARP (Good high-freq content)")
    elif focus_score > 0.2: print("   -> VERDICT: ACCEPTABLE")
    else: print("   -> VERDICT: BLURRY (Possible atmospheric distortion)")
        
    print("-" * 40)
    
    print(f"3. TEMPORAL COHERENCE:         {coherence:.3f} (0.0 - 1.0)")
    print("   (Similarity between Doppler frames)")
    if coherence > 0.8: print("   -> VERDICT: EXCELLENT (Stable reflectors)")
    elif coherence > 0.5: print("   -> VERDICT: GOOD (Standard SAR)")
    elif coherence > 0.3: print("   -> VERDICT: WEAK (Tomography will be difficult)")
    else: print("   -> VERDICT: FAILED (Random noise, Tomography impossible)")
        
    print("-" * 40)
    
    print(f"4. PHASE STABILITY (Jitter):   {phase_instability:.3f} rad")
    if phase_instability < 0.5: print("   -> VERDICT: STABLE (Good for Interferometry/Tomo)")
    elif phase_instability < 1.0: print("   -> VERDICT: SHAKY (Results may be noisy)")
    else: print("   -> VERDICT: UNSTABLE (Phase scrambling detected)")
    
    print("="*40)
    
    # Final Recommendation
    score = 0
    if dr_db > 15: score += 1
    if focus_score > 0.2: score += 1
    if coherence > 0.4: score += 1
    if phase_instability < 1.0: score += 1
    
    print("\nFINAL RECOMMENDATION:")
    if score == 4:
        print("✅ GREEN LIGHT: Data is perfect for 3D Tomography.")
    elif score >= 2:
        print("⚠️  YELLOW LIGHT: Proceed, but expect some noise. Use 'VelocitySpectrum' mode.")
    else:
        print("❌ RED LIGHT: Data quality is too low. Check file or select a different area.")

    # Optional Visualization
    print("\nDisplaying diagnostic plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: The Image Patch
    ax1.imshow(np.log1p(np.abs(patch)), cmap='gray')
    ax1.set_title("Static Image Sample")
    ax1.axis('off')
    
    # Plot 2: The "Ultrasound" Stack (Look magnitude vs Pixel)
    # We plot the stack as an image: Y-axis = Pixel Depth, X-axis = Time (Look)
    stack_mag = np.abs(stack).T # Transpose so Depth is Y
    ax2.imshow(np.log1p(stack_mag), aspect='auto', cmap='jet')
    ax2.set_title(f"Virtual Probe Data ({NUM_LOOKS} Time Steps)")
    ax2.set_xlabel("Time (Look Index)")
    ax2.set_ylabel("Depth (Pixel Index)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
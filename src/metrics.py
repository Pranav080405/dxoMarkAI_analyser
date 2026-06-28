import cv2
import numpy as np

def analyze_sharpness(opencv_img):
    """Calculates acutance based on the variance of the Laplacian."""
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Map raw variance to a normalized scale (0 to 100) for education
    sharpness_score = min(100.0, max(0.0, (laplacian_var / 500.0) * 100))
    return sharpness_score, laplacian_var

def analyze_noise_snr(opencv_img):
    """Samples the center quadrant to compute the Signal-to-Noise Ratio (SNR)."""
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Extract center 10% patch as a reference uniform area
    patch_h, patch_w = int(h * 0.1), int(w * 0.1)
    roi = gray[h//2 - patch_h:h//2 + patch_h, w//2 - patch_w:w//2 + patch_w]
    
    mean_signal = np.mean(roi)
    std_noise = np.std(roi)
    
    if std_noise == 0:
        return 100.0, float('inf')
    
    # Calculate SNR in Decibels
    snr_db = 20 * np.log10(mean_signal / std_noise)
    
    # Map SNR to a 0-100 score (Typical consumer sensor ranges from 15dB to 45dB)
    noise_score = min(100.0, max(0.0, ((snr_db - 15) / 30) * 100))
    return noise_score, snr_db

def analyze_exposure_and_dynamic_range(opencv_img):
    """Evaluates exposure balance by inspecting highlight clipping and shadow compression."""
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    
    clipped_highlights = np.sum(gray >= 254) / total_pixels
    crushed_shadows = np.sum(gray <= 2) / total_pixels
    
    mean_brightness = np.mean(gray)
    
    # Perfect exposure hits mid-tone ~120 without clipping
    exposure_penalty = abs(mean_brightness - 122) / 122
    clipping_penalty = (clipped_highlights + crushed_shadows) * 2
    
    exposure_score = max(0.0, min(100.0, (1.0 - (exposure_penalty + clipping_penalty)) * 100))
    return exposure_score, mean_brightness, clipped_highlights
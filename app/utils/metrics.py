import random

def generate_synthetic_metrics(model_type="cnn"):
    """
    Generate synthetic metrics for steganography evaluation.
    
    Args:
        model_type (str): Either "cnn" or "steganogan"
    
    Returns:
        dict: Dictionary containing the synthetic metrics
    """
    # Base ranges for CNN (slightly lower performance)
    if model_type.lower() == "cnn":
        accuracy_range = (92.5, 96.8)  # Percentage
        rs_bpp_range = (0.3, 0.7)      # Bits per pixel
        psnr_range = (28.0, 33.5)      # dB
        ssim_range = (0.82, 0.91)      # 0-1 scale
    # Better ranges for SteganoGAN
    else:  # steganogan
        accuracy_range = (95.0, 98.5)  # Percentage
        rs_bpp_range = (0.5, 0.9)      # Bits per pixel
        psnr_range = (31.0, 36.5)      # dB
        ssim_range = (0.87, 0.95)      # 0-1 scale
    
    # Generate random values within the specified ranges
    accuracy = round(random.uniform(*accuracy_range), 2)
    rs_bpp = round(random.uniform(*rs_bpp_range), 3)
    psnr = round(random.uniform(*psnr_range), 2)
    ssim = round(random.uniform(*ssim_range), 4)
    
    return {
        "accuracy": accuracy,
        "rs_bpp": rs_bpp,
        "psnr": psnr,
        "ssim": ssim
    }
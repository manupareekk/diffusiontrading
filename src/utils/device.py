"""
Device selection utilities.

Automatically detects the best available acceleration device
(CUDA for NVIDIA, MPS for Mac Silicon, or CPU).
"""

import torch

def get_device(device_str: str = "auto") -> str:
    """
    Get the best available device or the requested one.

    Args:
        device_str: Desired device ("auto", "cuda", "mps", "cpu")

    Returns:
        Device string identifier
    """
    if device_str != "auto":
        return device_str

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_torch_device(device_str: str = "auto") -> torch.device:
    """
    Get the best available PyTorch device object.
    """
    return torch.device(get_device(device_str))

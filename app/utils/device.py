"""
Device detection and configuration utilities.
"""

import platform
from typing import Dict, Any, Optional


def detect_optimal_device() -> Dict[str, Any]:
    """
    Detect the optimal device configuration for the current system.
    
    Returns:
        Dict containing device information and recommendations.
    """
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "apple_silicon": False,
        "torch_available": False,
        "mlx_available": False,
        "recommended_backend": "torch"
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["apple_silicon"] = True
        info["recommended_backend"] = "mlx"  # Prefer MLX on Apple Silicon
    
    # Check PyTorch availability and MPS support
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        
        # Check MPS (Metal Performance Shaders) availability
        if hasattr(torch.backends, 'mps'):
            info["mps_available"] = torch.backends.mps.is_available()
            if info["mps_available"]:
                info["mps_built"] = torch.backends.mps.is_built()
        else:
            info["mps_available"] = False
            info["mps_built"] = False
        
        # Check CUDA availability
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
    
    except ImportError:
        info["torch_available"] = False
    
    # Check MLX availability
    try:
        import mlx.core as mx
        info["mlx_available"] = True
        info["mlx_version"] = getattr(mx, '__version__', 'unknown')
        
        # If MLX is available on Apple Silicon, prefer it
        if info["apple_silicon"]:
            info["recommended_backend"] = "mlx"
    
    except ImportError:
        info["mlx_available"] = False
        # Fallback to torch on Apple Silicon if MLX not available
        if info["apple_silicon"]:
            info["recommended_backend"] = "torch"
    
    return info


def get_optimal_torch_device() -> str:
    """
    Get the optimal PyTorch device string.
    
    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    try:
        import torch
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        # Check CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Fallback to CPU
        return "cpu"
    
    except ImportError:
        return "cpu"


def get_memory_info() -> Optional[Dict[str, Any]]:
    """
    Get system memory information.
    
    Returns:
        Dict with memory info or None if unavailable.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent_used": memory.percent,
            "free": memory.free
        }
    except ImportError:
        return None


def validate_device_compatibility(backend: str) -> bool:
    """
    Validate if the specified backend is compatible with the current device.
    
    Args:
        backend: Backend name ("mlx", "torch")
    
    Returns:
        True if compatible, False otherwise.
    """
    device_info = detect_optimal_device()
    
    if backend == "mlx":
        return device_info["apple_silicon"] and device_info["mlx_available"]
    
    elif backend == "torch":
        return device_info["torch_available"]
    
    return False

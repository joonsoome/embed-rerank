"""
🚀 Apple Silicon Device Detection: Unleashing the Power of Apple's Chips

This module is your gateway to detecting and optimizing for Apple Silicon's
incredible capabilities. We automatically detect M-series chips, configure
MLX acceleration, and ensure your AI workloads run at maximum speed.

🧠 Apple Silicon Magic:
- Unified Memory Architecture detection
- MLX framework compatibility checking
- Metal Performance Shaders optimization
- Automatic backend selection for peak performance

Welcome to the future of on-device AI, powered by Apple's revolutionary silicon!
"""

import platform
from typing import Any, Dict, Optional


def detect_optimal_device() -> Dict[str, Any]:
    """
    🔍 Detect Optimal Device: Unleashing Apple Silicon Potential

    This function is like a digital sommelier for Apple Silicon - it tastes
    your hardware configuration and recommends the perfect AI backend pairing.

    When it detects Apple Silicon (M1/M2/M3/M4), it practically jumps with
    excitement and immediately suggests MLX for that unified memory magic!

    Returns:
        Dict with device capabilities and Apple Silicon optimization recommendations
    """
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "apple_silicon": False,
        "torch_available": False,
        "mlx_available": False,
        "recommended_backend": "torch",
    }

    # 🎯 Apple Silicon Detection: The moment of truth!
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["apple_silicon"] = True
        info["recommended_backend"] = "mlx"  # 🚀 MLX is the way on Apple Silicon!
        info["silicon_generation"] = "M-series"  # Could be M1, M2, M3, or M4+
        info["unified_memory"] = True

    # 🔥 PyTorch Capability Detection
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__

        # 🎭 Metal Performance Shaders: Apple's GPU acceleration magic
        if hasattr(torch.backends, 'mps'):
            info["mps_available"] = torch.backends.mps.is_available()
            if info["mps_available"]:
                info["mps_built"] = torch.backends.mps.is_built()
                info["apple_gpu_acceleration"] = True
        else:
            info["mps_available"] = False
            info["mps_built"] = False

        # 🟢 CUDA Detection (for the non-Apple folks)
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda

    except ImportError:
        info["torch_available"] = False

    # 🌟 MLX Framework Detection: The Apple Silicon Crown Jewel
    try:
        import mlx.core as mx

        info["mlx_available"] = True
        info["mlx_version"] = getattr(mx, '__version__', 'unknown')
        info["apple_mlx_magic"] = True

        # 🎉 If MLX is available on Apple Silicon, this is PEAK performance!
        if info["apple_silicon"]:
            info["recommended_backend"] = "mlx"
            info["performance_tier"] = "🚀 LUDICROUS SPEED"

    except ImportError:
        info["mlx_available"] = False
        # 😢 Fallback to PyTorch MPS on Apple Silicon
        if info["apple_silicon"]:
            info["recommended_backend"] = "torch"
            info["performance_tier"] = "⚡ Fast (but MLX would be faster!)"

    return info


def get_optimal_torch_device() -> str:
    """
    🎯 Optimal PyTorch Device Selection: Finding Your Silicon Sweet Spot

    This function is like a personal trainer for your PyTorch models - it finds
    the best device to flex those neural network muscles. On Apple Silicon,
    it enthusiastically points to MPS (Metal Performance Shaders) for that
    GPU-accelerated goodness!

    Returns:
        Device string optimized for your hardware ("mps", "cuda", or "cpu")
    """
    try:
        import torch

        # 🍎 Apple Silicon MPS: The golden path for M-series chips
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # 🚀 Apple GPU acceleration FTW!

        # 🟢 NVIDIA CUDA: For the non-Apple folks
        if torch.cuda.is_available():
            return "cuda"

        # 💻 CPU Fallback: Still respectable performance
        return "cpu"

    except ImportError:
        return "cpu"


def get_memory_info() -> Optional[Dict[str, Any]]:
    """
    📊 Memory Information: Understanding Your Apple Silicon Memory Pool

    Apple Silicon's unified memory architecture is a thing of beauty - CPU and
    GPU share the same memory pool, eliminating costly data transfers. This
    function helps you understand and monitor that magical unified memory.

    Returns:
        Dict with memory information or None if psutil unavailable
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent_used": memory.percent,
            "free": memory.free,
            "unified_memory": True,  # 🍎 Apple Silicon has unified memory!
            "memory_type": "Unified Memory" if platform.machine() == "arm64" else "Traditional RAM",
        }
    except ImportError:
        return None


def validate_device_compatibility(backend: str) -> bool:
    """
    ✅ Device Compatibility Validation: Ensuring Perfect Harmony

    This function is like a compatibility matchmaker - it ensures your chosen
    backend can dance beautifully with your hardware. MLX + Apple Silicon?
    Perfect match! PyTorch + anything? Pretty solid relationship!

    Args:
        backend: Your desired backend ("mlx" for Apple Silicon magic, "torch" for versatility)

    Returns:
        True if it's a match made in silicon heaven, False if you need to reconsider
    """
    device_info = detect_optimal_device()

    if backend == "mlx":
        # 🍎 MLX requires Apple Silicon - no exceptions, but oh so worth it!
        return device_info["apple_silicon"] and device_info["mlx_available"]

    elif backend == "torch":
        # 🔥 PyTorch is more flexible - works almost everywhere
        return device_info["torch_available"]

    return False

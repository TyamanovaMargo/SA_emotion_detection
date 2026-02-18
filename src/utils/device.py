"""Device detection and management utilities."""

import torch
import gc
import os
from typing import Literal, Optional

DeviceType = Literal["cuda", "cpu", "mps"]


def get_optimal_device(prefer_gpu: bool = True, gpu_index: int = 0) -> str:
    """
    Detect and return the optimal device for computation.
    
    Priority:
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback
    
    Args:
        prefer_gpu: If False, always return CPU
        gpu_index: Preferred GPU index when multiple GPUs available
        
    Returns:
        Device string: "cuda:0", "cuda:1", "mps", or "cpu"
    """
    if not prefer_gpu:
        return "cpu"
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_index < gpu_count:
            return f"cuda:{gpu_index}"
        return "cuda:0"
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def get_device_from_env(env_var: str = "DEVICE", default: str = "auto") -> str:
    """
    Get device from environment variable with auto-detection fallback.
    
    Args:
        env_var: Environment variable name
        default: Default value if env var not set ("auto" for auto-detection)
        
    Returns:
        Device string
    """
    device = os.getenv(env_var, default)
    
    if device == "auto":
        return get_optimal_device()
    
    return device


def setup_gpu_memory(device: str, memory_fraction: float = 0.9):
    """
    Configure GPU memory settings for optimal performance.
    
    Args:
        device: Device string ("cuda", "cuda:0", "cuda:1", "mps", "cpu")
        memory_fraction: Fraction of GPU memory to use (0.0-1.0)
    """
    if device.startswith("cuda"):
        # Parse GPU index from device string
        gpu_idx = 0
        if ":" in device:
            gpu_idx = int(device.split(":")[1])
        
        # Set memory fraction per GPU
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_idx)
        
        # Enable TF32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
    
    elif device == "mps":
        # MPS-specific optimizations
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def clear_gpu_memory(device: str = "cuda"):
    """
    Aggressively clear GPU memory.
    
    Call this between model stages to free memory for the next model.
    
    Args:
        device: Device string
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def get_batch_size_for_device(device: str, base_batch_size: int = 8) -> int:
    """
    Get optimal batch size based on device capabilities.
    
    Args:
        device: Device string (e.g. "cuda:0", "cuda:1", "cpu")
        base_batch_size: Base batch size for GPU
        
    Returns:
        Optimal batch size
    """
    if device == "cpu":
        return 1  # CPU processes one at a time
    
    elif device.startswith("cuda"):
        # Parse GPU index
        gpu_idx = 0
        if ":" in device:
            gpu_idx = int(device.split(":")[1])
        
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            gpu_memory_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9
            if gpu_memory_gb >= 16:
                return base_batch_size * 2
            elif gpu_memory_gb >= 8:
                return base_batch_size
            else:
                return max(1, base_batch_size // 2)
    
    elif device == "mps":
        # Apple Silicon - moderate batch size
        return max(1, base_batch_size // 2)
    
    return 1


def print_device_info():
    """Print information about available compute devices."""
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    
    # CPU
    print(f"CPU: Available")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: Available")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print(f"CUDA: Not available")
    
    # MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon): Available")
    else:
        print(f"MPS (Apple Silicon): Not available")
    
    # Selected device
    optimal = get_optimal_device()
    print(f"\nOptimal Device: {optimal}")
    print("=" * 60)

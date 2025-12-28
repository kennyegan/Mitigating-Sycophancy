import torch

# Check if MPS (Apple Silicon GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Success: PyTorch is using Apple Silicon GPU (MPS).")
elif torch.cuda.is_available():
    print("Success: PyTorch is using CUDA (NVIDIA GPU).") # Rare on Mac
else:
    print("PyTorch is running on CPU.")
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())      # → True
print("CUDA versión (runtime):", torch.version.cuda)      # → 12.6
print("GPU:", torch.cuda.get_device_name(0))

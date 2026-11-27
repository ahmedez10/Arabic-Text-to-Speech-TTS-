import torch

cuda_ok = torch.cuda.is_available()
print("PyTorch version:", torch.__version__)
print("Is CUDA available?", cuda_ok)
if cuda_ok:
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU (no CUDA GPU detected)")

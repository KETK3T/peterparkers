import torch

print("Cuda available:", torch.cuda.is_available(),
      "build:", torch.version.cuda,
      "gpu:", torch.cuda.get_device_name(0)
      )
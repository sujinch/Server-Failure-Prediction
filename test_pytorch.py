import torch

print("Version de PyTorch :", torch.__version__)

print("CUDA disponible :", torch.cuda.is_available())

x = torch.tensor([1.0, 2.0, 3.0])

print("Notre premier tenseur :", x)
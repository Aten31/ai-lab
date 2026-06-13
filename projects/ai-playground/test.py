import torch

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

x = torch.tensor([1,2,3])
print("Tensor:", x)

print("Tensor × 2:", x*2)
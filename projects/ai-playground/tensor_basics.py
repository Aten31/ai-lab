import torch 

a  = torch.tensor([1,2,3,4])
b  = torch.tensor([4,5,6,7])

print("Tensor A:",a)
print("Tensor B:",b)

print('Addition:')
print(a+b)

print("Multiplication:")
print(a*b)

print("Dot Product:")
print(torch.dot(a,b))

matrix = torch.tensor([
    [1,2],
    [3,4]
])

print("Matrix:")
print(matrix)

print("Matrix Shape: ")
print(matrix.shape)


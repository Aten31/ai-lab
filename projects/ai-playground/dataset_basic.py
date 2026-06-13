import torch

# pretend this is student data
students = torch.tensor([
    [80,20],
    [65,15],
    [90,30],
    [50,5]
])

print("Dataset:")
print(students)

print("\nShape:")
print(students.shape)

print("\nFirst student:")
print(students[0])

print("\nSecond column:")
print(students[:,1])

print("\nAverage score:")
print(torch.mean(students.float(),dim=0))
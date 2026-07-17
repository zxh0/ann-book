import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + 1) # [2 3 4]
print(a * 2) # [2 4 6]
print(a + b) # [5 7 9]
print(a - b) # [-3 -3 -3]
print(a * b) # [ 4 10 18]
print(a @ b) # 32

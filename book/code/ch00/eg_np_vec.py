import numpy as np

vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])

print(vec_a + 1)     # [2 3 4]
print(vec_a * 2)     # [2 4 6]
print(vec_a + vec_b) # [5 7 9]
print(vec_a - vec_b) # [-3 -3 -3]
print(vec_a * vec_b) # [ 4 10 18]
print(vec_a @ vec_b) # 32

def weighted_sum(x: list[list[float]], 
                 w: list[list[float]], 
                 k: int) -> float:
    total = 0.0
    for i in range(k):
        for j in range(k):
            total += x[i][j] * w[i][j]
    return total


x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
w = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
k = 3
print(weighted_sum(x, w, k)) # 5.0

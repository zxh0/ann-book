def normalize(x: list[float]) -> list[float]:
    total = sum(x)
    return [v / total for v in x]

x = [0.496, 0.254, 0.131, 0.067, 0.034, 0.018]
print([round(x, 3) for x in normalize([0.496, 0.254, 0.131, 0.067, 0.0, 0.0, 0.0])])
print([round(x, 3) for x in normalize([0.496, 0.254, 0.131, 0.0, 0.0, 0.0])])
print([round(x, 3) for x in normalize([0.496, 0.254, 0.0,  0.0, 0.0, 0.0])])
print([round(x, 3) for x in normalize([0.496, 0.0, 0.0,  0.0, 0.0, 0.0])])

# [0.523, 0.268, 0.138, 0.071, 0.0, 0.0, 0.0]
# [0.563, 0.288, 0.149, 0.0, 0.0, 0.0]
# [0.661, 0.339, 0.0, 0.0, 0.0, 0.0]
# [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

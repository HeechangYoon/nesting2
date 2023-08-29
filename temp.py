import numpy as np

# 큰 배열과 작은 배열 생성
large_array = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
])

small_array = np.array([
    [1, 1],
    [1, 1]
])

# 작은 배열을 큰 배열에 밀착시키기
large_rows, large_cols = large_array.shape
small_rows, small_cols = small_array.shape

for r in range(large_rows - small_rows + 1):
    for c in range(large_cols - small_cols + 1):
        if np.all(large_array[r:r + small_rows, c:c + small_cols] == 0):
            large_array[r:r + small_rows, c:c + small_cols] = small_array
            break
    else:
        continue
    break

print(large_array)
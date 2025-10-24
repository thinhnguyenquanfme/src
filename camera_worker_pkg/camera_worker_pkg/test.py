import numpy as np

grid = np.mgrid[0:2, 0:3]
gridT = grid.T
zero = np.zeros((2,2,3))

print(grid)
print(gridT)

print(zero)
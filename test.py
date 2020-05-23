import numpy as np

i0 = 0
i1 = 0
for i in range(100000):
    x = np.random.randint(2)
    if x == 0:
        i0 += 1
    else:
        i1 += 1
print(i0, i1)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

axes_stats = {1: {0: (0.78125, 0.7517729997634888), 1: (0, 0), 2: (0, 0)},
              2: {0: (0.6305555701255798, 0.5723472237586975), 1: (0.7376760840415955, 0.6863157749176025), 2: (0, 0)},
              3: {0: (0.625, 0.6131805181503296), 1: (0.7145161032676697, 0.6602686643600464),
                  2: (0.6465908885002136, 0.5638148784637451)}}

complexity = []
ordering = []
acc_dz = []
f1_dz = []
for comp, v1 in axes_stats.items():
    for ord, v2 in v1.items():
        if v2 != (0, 0):
            complexity.append(comp)
            ordering.append(ord)
            acc_dz.append(v2[0])
            f1_dz.append(v2[1])

z3 = np.zeros(6)
dx = np.ones(6) * 0.9
dy = np.ones(6) * 0.9
# dz = [1,2,3,4,5,6,7,8,9,10]

ax1.bar3d(complexity, ordering, z3, dx, dy, acc_dz)
ax1.set_xticks([1, 2, 3])
ax1.set_yticks([0, 1, 2])

ax1.set_xlabel('complexity')
ax1.set_ylabel('ordering')
ax1.set_zlabel('Accuracy')
plt.tight_layout()
plt.show()

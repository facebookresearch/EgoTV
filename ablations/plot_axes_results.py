import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

# style.use('ggplot')

fig = plt.figure()

# for baselines
axes_stats_baseline = {1: {0: (0.78125, 0.7517729997634888), 1: (0, 0), 2: (0, 0)},
              2: {0: (0.6305555701255798, 0.5723472237586975), 1: (0.7376760840415955, 0.6863157749176025), 2: (0, 0)},
              3: {0: (0.625, 0.6131805181503296), 1: (0.7145161032676697, 0.6602686643600464),
                  2: (0.6465908885002136, 0.5638148784637451)}}

# for nesy
axes_stats_nesy = {1: {0: (0.9230769276618958, 0.9181286692619324), 1: (0, 0), 2: (0, 0)},
              2: {0: (0.8223684430122375, 0.7988077402114868), 1: (0.780168354511261, 0.7502656579017639), 2: (0, 0)},
              3: {0: (0.8072289228439331, 0.7837838530540466), 1: (0.7915443181991577, 0.7672131061553955), 2: (0.8272395133972168, 0.8029196262359619)}}


def plot_3d(ax, stats, title):
    complexity = []
    ordering = []
    acc_dz = []
    f1_dz = []
    for comp, v1 in stats.items():
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

    ax.bar3d(complexity, ordering, z3, dx, dy, acc_dz)
    ax.set_xticks([1, 2, 3])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlabel('complexity')
    ax.set_ylabel('ordering')
    ax.set_zlabel('Accuracy')
    ax.title.set_text(title)

ax = fig.add_subplot(121, projection='3d')
plot_3d(ax, axes_stats_baseline, title='Baselines')
ax = fig.add_subplot(122, projection='3d')
plot_3d(ax, axes_stats_nesy, title='NeSy')

# plt.tight_layout()
plt.savefig('complexity-ordering-comparison.png')
# plt.show()

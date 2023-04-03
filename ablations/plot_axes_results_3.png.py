# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
# for baselines
axes_stats_baseline = {1: {0: (0.78125, 0.7517729997634888), 1: (0, 0), 2: (0, 0)},
              2: {0: (0.6305555701255798, 0.5723472237586975), 1: (0.7376760840415955, 0.6863157749176025), 2: (0, 0)},
              3: {0: (0.625, 0.6131805181503296), 1: (0.7145161032676697, 0.6602686643600464),
                  2: (0.6465908885002136, 0.5638148784637451)}}

# for nesy
axes_stats_nesy = {1: {0: (0.9230769276618958, 0.9181286692619324), 1: (0, 0), 2: (0, 0)},
              2: {0: (0.8223684430122375, 0.7988077402114868), 1: (0.780168354511261, 0.7502656579017639), 2: (0, 0)},
              3: {0: (0.8072289228439331, 0.7837838530540466), 1: (0.7915443181991577, 0.7672131061553955), 2: (0.8272395133972168, 0.8029196262359619)}}

# plt.style.use("classic")
fig = plt.figure(figsize=(6,3))
plt.rcParams['font.size'] = 13

def plot_complexity(stats):
    x = []
    value = []
    for k1, v1 in stats.items():
        x.append(k1)
        val_mean = []
        for k2, v2 in v1.items():
            acc, f1 = v2
            if acc != 0:
                val_mean.append(acc)
        value.append(mean(val_mean))
    return x, value

ax = fig.add_subplot(121)
x, value1 = plot_complexity(axes_stats_baseline)
_, value2 = plot_complexity(axes_stats_nesy)
ax.plot(x, value1, label='Baselines')
ax.plot(x, value2, label='NeSy')
ax.set_xlabel("complexity")
ax.set_ylabel("accuracy")
ax.set_xticks([1, 2, 3])
ax.set_yticks(np.arange(4, 11, 2) * 0.1)
ax.legend()

ax = fig.add_subplot(122)
axes_stats_baseline1 = {0: {}, 1: {}, 2: {}}
for k1, v1 in axes_stats_baseline.items():
    for k2, v2 in axes_stats_baseline[k1].items():
        axes_stats_baseline1[k2][k1] = v2
axes_stats_nesy1 = {0: {}, 1: {}, 2: {}}
for k1, v1 in axes_stats_nesy.items():
    for k2, v2 in axes_stats_nesy[k1].items():
        axes_stats_nesy1[k2][k1] = v2

x, value1 = plot_complexity(axes_stats_baseline1)
_, value2 = plot_complexity(axes_stats_nesy1)
ax.plot(x, value1, label='Baselines')
ax.plot(x, value2, label='NeSy')
ax.set_xlabel("ordering")
ax.set_ylabel("accuracy")
ax.set_xticks([0, 1, 2])
ax.set_yticks(np.arange(4, 11, 2) * 0.1)
ax.legend()

plt.tight_layout()
plt.show()

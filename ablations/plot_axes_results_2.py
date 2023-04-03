# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

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

def plot_scatter(ax, stats, title):
    x = []
    y = []
    value = []
    for k1, v1 in stats.items():
        for k2, v2 in v1.items():
            x.append(k1)
            y.append(k2)
            acc, f1 = v2
            value.append(acc)
    value = value * 10
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=(value[i]**6)*1000, c=colors[(x[i]-1)], marker=shapes[y[i]])
    ax.set_xlabel("complexity")
    ax.set_ylabel("ordering")
    ax.set_xticks([1, 2, 3])
    ax.set_yticks([0, 1, 2])
    ax.title.set_text(title)

colors = ['red', 'blue', 'green', 'yellow', 'purple']
# shapes = ['o', 'v', ',']
shapes = ['o', 'o', 'o']
# plt.style.use("classic")
# fig = plt.figure(figsize=(6,3))
# plt.rcParams['font.size'] = 13
# ax = fig.add_subplot(121)
# plot_scatter(ax, axes_stats_baseline, title='Baselines')
# ax = fig.add_subplot(122)
# plot_scatter(ax, axes_stats_nesy, title='NeSy')
# plt.tight_layout()
# plt.show()

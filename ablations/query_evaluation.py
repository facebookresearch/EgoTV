# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
plt.rcParams['font.size'] = 15

Split = ['train', 'sub-goal', 'verb-noun', 'context-verb-noun', 'context-goal', 'abstraction']

state_dict = {'heat': 86.6, 'cool': 83.2, 'clean': 82.0, 'slice': 40.0}
plt.bar(state_dict.keys(), state_dict.values())
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("sub-goal", fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title("StateQuery Accuracy", fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

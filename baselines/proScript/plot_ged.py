import matplotlib.pyplot as plt
import numpy as np
fig1 = plt.figure(figsize=(4, 4))
x= np.arange(1, 13, 1)

ged_sub_goal = [3.43, 1.31, 0.35, 0.20, 0.11, 0.05, 0.05, 0.04, 0.04, 0.04, 0.037, 0.035]
ged_verb_noun = [2.83, 0.97, 0.25, 0.19, 0.09, 0.075, 0.068, 0.07, 0.057, 0.037, 0.031, 0.025]
ged_context_verb_noun = [2.50, 0.98, 0.15, 0.03, 0.01, 0.0, 0.0, 0.0, 0.007, 0.0, 0.0, 0.0]
ged_context_goal = [2.70, 1.20, 0.18, 0.08, 0.05, 0.040, 0.035, 0.027, 0.025, 0.015, 0.0, 0.0]
plt.plot(x, ged_sub_goal, label='sub-goal composition')
plt.plot(x, ged_verb_noun, label='verb-noun composition')
plt.plot(x, ged_context_verb_noun, label='context-verb-noun composition')
plt.plot(x, ged_context_goal, label='context-goal composition')
plt.xlabel('Epoch')
plt.ylabel('Test GED')
plt.tight_layout()
plt.legend()
plt.show()
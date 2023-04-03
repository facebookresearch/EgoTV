# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# fig = plt.figure(figsize=(6,3))
# Define the confusion matrix
confusion_matrix = np.array([[104, 0, 0, 0, 3, 0],
                             [0, 86, 0, 0, 9, 0],
                             [1, 0, 171, 0, 8, 0],
                             [0, 1, 1, 98, 16, 0],
                             [0, 0, 1, 1, 150, 0],
                             [5, 13, 9, 83, 76, 0]])

# Define the class names
classes = ['heat', 'cool', 'clean',
           'slice', 'put', 'other']

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams['font.size'] = 18
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
disp = disp.plot(cmap=plt.cm.Blues, ax=ax)

# Set the axis labels
ax.set_xlabel('Predicted', fontsize=23)
ax.set_ylabel('True', fontsize=23)
# ax.set_title('Confusion Matrix', fontsize=16)
ax.set_xticklabels(['StateQuery\n(obj, hot)', 'StateQuery\n(obj, cold)', 'StateQuery\n(obj, clean)',
                    'RelationQuery\n(obj,knife,slice)', 'RelationQuery\n(obj,recep,in)', 'None'], ha='left')
# ax.invert_yaxis()
ax.xaxis.tick_top()
# plt.gca().yaxis.label.set_visible(False)
# Rotate the xtick labels at a 45 degree angle
plt.xticks(rotation=45)
plt.yticks(fontsize=20)
plt.xticks(fontsize=18)
# plt.imshow(np.random.random(15).reshape((5,3)))
# cb = plt.colorbar()
# cb.remove()
disp.im_.colorbar.remove()
plt.tight_layout()
# Show the plot
plt.savefig('confusion_mat.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('classic')
plt.style.use("bmh")
fig1 = plt.figure(figsize=(14, 14))
x = np.arange(0, 12, 2)


def plot_bar(metric):
    plt.bar(x - 0.7, resnet_bert_no, width, label="resnet | bert")
    plt.bar(x - 0.5, resnet_bert_attention, width, label="resnet | bert | attention")
    plt.bar(x - 0.3, resnet_glove_no, width, label="resnet | glove")
    plt.bar(x - 0.1, resnet_glove_attention, width, label="resnet | glove | attention")
    plt.bar(x + 0.1, i3d_bert_no, width, label="i3d | bert")
    plt.bar(x + 0.3, i3d_bert_attention, width, label="i3d | bert | attention")
    plt.bar(x + 0.5, i3d_glove_no, width, label="i3d | glove")
    plt.bar(x + 0.7, i3d_glove_attention, width, label="i3d | glove | attention")
    plt.xticks(x, ['validation', 'sub-goal', 'verb-noun', 'context verb-noun', 'context goal', 'abstraction'])
    plt.ylabel(metric)
    plt.tight_layout()

plt.rcParams['font.size'] = 15
plt.subplot(2, 1, 1)
resnet_bert_no = [96.2, 59.6, 84.3, 91.1, 81.0, 62.7]
resnet_glove_no = [95.7, 62.0, 84.9, 92.2, 83.1, 62.0]
resnet_bert_attention = [86.0, 53.45, 78.1, 85.4, 84.0, 56.0]
resnet_glove_attention = [95.7, 56.4, 86.0, 95.0, 93.0, 66.1]
i3d_bert_no = np.array([[98.6, 58.7, 79.3, 88.1, 83.7, 61.6],
                        [98.8, 56.3, 81.0, 90.2, 85.2, 62.9]]).mean(axis=0)
i3d_glove_no = [99.2, 61.0, 81.0, 89.1, 83.3, 61.3]
i3d_bert_attention = [90.6, 59.4, 79.1, 88.4, 84.5, 64.0]
i3d_glove_attention = [90.5, 52.8, 77.4, 88.9, 85.2, 60.2]
width = 0.2
# plot data in grouped manner of bar type
plot_bar(metric="Accuracy")
plt.legend(title="Models",
           fontsize=12,
           title_fontsize=16,
           ncol=5,
           bbox_to_anchor=(0.5, 0.7, 0.5, 0.5))

plt.subplot(2, 1, 2)
resnet_bert_no = [96.0, 48.0, 83.6, 91.1, 80.0, 50.3]
resnet_glove_no = [95.7, 61.7, 85.1, 92.2, 82.4, 53.3]
resnet_bert_attention = [86.4, 37.0, 77.0, 86.0, 84.3, 39.0]
resnet_glove_attention = [95.7, 42.0, 86.1, 95.0, 93.0, 58.3]
i3d_bert_no = np.array([[98.6, 49.0, 78.7, 88.3, 83.1, 47.5],
                        [98.8, 44.5, 80.8, 90.5, 85.1, 46.9]]).mean(axis=0)
i3d_glove_no = [99.2, 59.1, 80.2, 89.3, 83.1, 50.5]
i3d_bert_attention = [90.3, 48.5, 77.3, 88.3, 84.2, 49.5]
i3d_glove_attention = [90.2, 34.5, 77.1, 89.3, 85.4, 50.0]
plot_bar(metric="F1")
plt.savefig('metric comparison')

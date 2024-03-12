import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

# 示例数据
y_true = []
y_pred = []
for i in range(1, 16):
    y_true.append(0)
for i in range(16, 21):
    y_true.append(1)
for i in range(21, 100):
    y_true.append(2)


for i in range(1, 13):
    y_pred.append(0)
for i in range(13, 18):
    y_pred.append(1)
for i in range(18, 100):
    y_pred.append(2)


# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 使用seaborn绘制热力图
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['弱', '中', '强'], yticklabels=['弱', '中', '强'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')


# plt.savefig('confusion_matrix.png')

# 显式调用 plt.show() 来显示图形
plt.show()

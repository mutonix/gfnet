import numpy as np
from itertools import cycle
import torch.utils.data
import torch
from torch.autograd import Variable
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

n_class = 9
use_gpu = True

def softmax(x):
    return np.exp(x) / np.exp(x).sum(-1, keepdims=True)

def onehot(x, n_class=n_class):
    return np.eye(n_class)[x].tolist()

def plot_roc_auc(arch, y_preds, y_test,y_score):
    # print(y_preds)
    # print(y_preds.shape)
    # print(y_test.shape)
    # print(y_score.shape)
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'pink', 'purple', 'brown'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(arch+'_figure.jpg')

def val_last(val_loader, model, criterion, optimizer, arch):
    model.eval()
    y = []
    pred = []
    scores = []
    # Iterate over data.
    with torch.no_grad():
        for inputs, labels in val_loader:
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            # 找到概率最大的下标
            _, preds = torch.max(outputs.data, 1)
            for i in preds.cpu():
                # print('preds ', i.shape)  # []
                pred.append(onehot(i))

            for i in labels.cpu():
                y.append(onehot(i))
            for i in outputs.data.cpu():
            #     print('outputs, i.shape', i.shape)  # 9
            #     print('softmax(i).shape',softmax(i).shape)  # 9
                scores.append(softmax(i).tolist())


            # pred.append(np.array((preds.cpu().numpy())))
            # y.append(np.array((labels.cpu().numpy())))
            # scores.append(np.array((outputs.data.cpu().numpy())))
    print(np.array(scores).shape)
    plot_roc_auc(arch,y_preds= np.array(pred),y_score=np.array(scores),y_test=np.array(y))
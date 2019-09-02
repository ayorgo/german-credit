import numpy as np
import pandas as pd
import requests
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     GridSearchCV)
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score


def plot_categorical(series, axes=None):
    """ A shortcut for plotting categorical features
    """

    if not axes:
        _, axes = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 3))

    (series.value_counts(normalize=True)
           .sort_index()
           .plot(ax=axes, kind='bar', rot='horizontal', title=series.name))


def downsample(x, y):
    """ Randomly downsample the majority class in binary classification dataset
    """

    if (y == 0).sum() > (y == 1).sum():
        majority_class = 0
        minority_class = 1
    else:
        majority_class = 1
        minority_class = 0

    x_majority = x[y == majority_class]
    x_minority = x[y == minority_class]

    downsampled_size = len(x_minority)
    x_majority_downsampled = resample(x_majority,
                                      replace=True,
                                      n_samples=downsampled_size)

    x_downsampled = pd.concat([x_majority_downsampled,
                               x_minority],
                              ignore_index=True)
    y_downsampled = pd.concat([pd.Series([majority_class]*downsampled_size),
                               pd.Series([minority_class]*downsampled_size)],
                              ignore_index=True)

    return x_downsampled, y_downsampled


def train(x, y, estimator, params, iter_no):
    """ Training bootstrap utilising stratified K-fold cross validation
    """

    for _ in range(iter_no):
        x_downsampled, y_downsampled = downsample(x, y)

        size = int(round(len(x_downsampled)*0.3))
        x_train, x_test, y_train, y_test = train_test_split(x_downsampled,
                                                            y_downsampled,
                                                            test_size=size)

        grid_srch = GridSearchCV(estimator=estimator,
                                 param_grid=params,
                                 cv=StratifiedKFold(n_splits=5),
                                 scoring='roc_auc',
                                 refit='roc_auc',
                                 n_jobs=-1)
        grid_srch.fit(x_train, y_train)

        y_pred_proba = grid_srch.predict_proba(x_test)[:, 1]

        yield {'estimator': grid_srch,
               'y_test': y_test,
               'y_pred_proba': y_pred_proba}


def calculate_metrics(models, threshold):
    """ Calculate performance metrics
    """

    def prc(num):
        return round(100*num, 1)

    fpr_roc = np.linspace(0, 1, 100)

    for model in models:
        metric = dict()

        conf_m = confusion_matrix(model['y_test'],
                                  model['y_pred_proba'] >= threshold,
                                  labels=[0, 1])
        metric['tn'], metric['fp'], metric['fn'], metric['tp'] = conf_m.ravel()

        metric['tpr'] = prc(metric['tp']/(metric['tp'] + metric['fn']))
        metric['tnr'] = prc(metric['tn']/(metric['fp'] + metric['tn']))
        metric['fpr'] = prc(metric['fp']/(metric['fp'] + metric['tn']))
        metric['fnr'] = prc(metric['fn']/(metric['tp'] + metric['fn']))

        metric['acc'] = prc(
            (metric['tp'] + metric['tn'])/
            (metric['tp'] + metric['fp'] + metric['fn'] + metric['tn']))
        metric['precision'] = prc(metric['tp']/(metric['tp'] + metric['fp']))
        metric['recall'] = prc(metric['tp']/(metric['tp'] + metric['fn']))

        fpr_tmp, tpr_tmp, _ = roc_curve(model['y_test'],
                                        model['y_pred_proba'],
                                        pos_label=1)

        metric['tpr_roc'] = np.interp(fpr_roc, fpr_tmp, tpr_tmp)
        metric['fpr_roc'] = fpr_roc
        metric['roc_auc'] = roc_auc_score(model['y_test'],
                                          model['y_pred_proba'],
                                          average='micro')
        yield metric


def visualize_metrics(metrics):
    """ Plot/pring the metrics object
    """

    tpr_mean = np.mean([m['tpr'] for m in metrics])
    tnr_mean = np.mean([m['tnr'] for m in metrics])
    fpr_mean = np.mean([m['fpr'] for m in metrics])
    fnr_mean = np.mean([m['fnr'] for m in metrics])

    acc_mean = np.mean([m['acc'] for m in metrics])
    precision_mean = np.mean([m['precision'] for m in metrics])
    recall_mean = np.mean([m['recall'] for m in metrics])

    roc_auc_mean = np.mean([m['roc_auc'] for m in metrics])

    fpr_roc_mean = np.mean([m['fpr_roc'] for m in metrics], axis=0)
    tpr_roc_mean = np.mean([m['tpr_roc'] for m in metrics], axis=0)

    print()
    print(f'True Positives: {tpr_mean:.1f}%')
    print(f'True Negatives: {tnr_mean:.1f}%')
    print(f'False Positives: {fpr_mean:.1f}%')
    print(f'False Negatives: {fnr_mean:.1f}%')
    print()
    print(f'Accuracy: {acc_mean:.1f}')
    print(f'Precision: {precision_mean:.1f}')
    print(f'Recall: {recall_mean:.1f}')
    print(f'Area under the curve: {roc_auc_mean:.3f}')

    area = auc(fpr_roc_mean, tpr_roc_mean)
    plt.plot(fpr_roc_mean,
             tpr_roc_mean,
             label=f'ROC curve (area = {area:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.fill_between(fpr_roc_mean, tpr_roc_mean, color='silver')

    plt.plot(fpr_mean/(fpr_mean + tnr_mean),
             tpr_mean/(tpr_mean + fnr_mean),
             marker='o',
             color='black')

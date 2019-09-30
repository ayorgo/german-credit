import numpy as np
import pandas as pd
import scipy.stats as st
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                     StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import imblearn.pipeline as imb_pipe


class AggregatedClassifier:
    def __init__(self, estimators, aggregator, **kwargs):
        self.__aggregator = aggregator
        self.kwargs = kwargs
        self.estimators = estimators

    def __predict_single(self, estimator, X):
        return estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.__aggregator([self.__predict_single(estimator, X)
                                  for estimator
                                  in self.estimators],
                                 **self.kwargs)


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


def nested_cv(X, y, est, p_grid, scoring, inner_splits, outer_splits,
              rand_state=None, randcv_budget=20):

    # We fix rand_state here for the nested CV splits to be the same
    # for each classifier so we can meaningfully compare their scores
    if rand_state is None:
        rand_state = np.random.randint(123456789)

    # According to Kohavi, 1995:
    # "... stratification is generally a better scheme, both in terms of bias
    # and variance, when compared to regular cross-validation."
    inner_cv = StratifiedKFold(n_splits=inner_splits,
                               shuffle=True,
                               random_state=rand_state)

    # Inner CV (search for the best set of hyper-parameters)
    clf = RandomizedSearchCV(estimator=est,
                             param_distributions=p_grid,
                             n_iter=randcv_budget,
                             cv=inner_cv,
                             iid=False, # True?
                             scoring=scoring,
                             n_jobs=-1)

    # Outer CV (cross-validate the best classifier)
    if outer_splits:
        outer_cv = StratifiedKFold(n_splits=outer_splits,
                                   shuffle=True,
                                   random_state=rand_state)
        cv_results = cross_validate(estimator=clf,
                                    X=X,
                                    y=y,
                                    cv=outer_cv,
                                    scoring=scoring,
                                    return_estimator=True)
        score = cv_results['test_score'].mean()
        estimator = AggregatedClassifier(cv_results['estimator'], np.mean)

        return score, estimator
    else:
        clf.fit(X, y)

        return clf.best_score_, clf.best_estimator_


def compare_classifiers(X, y, ests, scoring, trials, inner_splits,
                        outer_splits=None, randcv_budget=20):

    # For every model type
    for label, steps, p_grid in ests:
        cv_scores = np.empty(trials, dtype=float)
        cv_estimators = np.empty(trials, dtype=object)

        # Collect results from multiple trials of (nested) CV
        # TODO: RepeatedKFold? RepeatedStratifiedKFold?
        for trial in range(trials):
            est = imb_pipe.Pipeline(steps)
            cv_scores[trial], cv_estimators[trial] = nested_cv(X,
                                                               y,
                                                               est,
                                                               p_grid,
                                                               scoring,
                                                               inner_splits,
                                                               outer_splits,
                                                               randcv_budget)
        yield label, cv_scores, cv_estimators


def visualise_scores(results, y, scoring, balance, inner_splits,
                     outer_splits=None):
    """Plot the scores distribution to assess stability
       of the model selection process
    """

    size = sum(y)*2 if balance else len(y)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    if outer_splits:
        outer_tt_size = size/outer_splits
        outer_tr_size = size - outer_tt_size

        inner_tt_size = outer_tr_size/inner_splits
        inner_tr_size = outer_tr_size - inner_tt_size

        title = (f'CV splits, outer/inner: {outer_splits}/{inner_splits}\n'
                 'Outer training/test set sizes: '
                 f'{outer_tr_size:.0f}/{outer_tt_size:.0f}\n'
                 'Inner training/test set sizes: '
                 f'{inner_tr_size:.0f}/{inner_tt_size:.0f}\n')
    else:
        inner_tt_size = size/inner_splits
        inner_tr_size = size - inner_tt_size

        title = (f'CV splits: {inner_splits}\n'
                 'Training/test set sizes: '
                 f'{inner_tr_size:.0f}/{inner_tt_size:.0f}\n')

    if not balance:
        imbalance = sum(y)/len(y)
        title += f'Imbalance: {imbalance:.0%}/{1 - imbalance:.0%}'


    labels, scores, _ = zip(*results)
    axes[0].boxplot(scores, labels=labels)

    for label, score in zip(labels, scores):
        axes[1].plot(score, label=label)

    plt.legend(loc='upper right')
    plt.ylabel(scoring)
    fig.suptitle(title, x=0, y=0, ha='left', va='top')


def collect_metrics(models, threshold):
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


def aggregate_metrics(metrics):
    """ Aggregate metrics from different models
    """

    metric = dict()
    metric['tpr_mean'] = np.mean([m['tpr'] for m in metrics])
    metric['tnr_mean'] = np.mean([m['tnr'] for m in metrics])
    metric['fpr_mean'] = np.mean([m['fpr'] for m in metrics])
    metric['fnr_mean'] = np.mean([m['fnr'] for m in metrics])

    metric['acc_mean'] = np.mean([m['acc'] for m in metrics])
    metric['precision_mean'] = np.mean([m['precision'] for m in metrics])
    metric['recall_mean'] = np.mean([m['recall'] for m in metrics])

    roc_auc = [m['roc_auc'] for m in metrics]
    metric['roc_auc_mean'] = np.mean(roc_auc)


    metric['fpr_roc_mean'] = np.mean([m['fpr_roc'] for m in metrics], axis=0)
    metric['tpr_roc_mean'] = np.mean([m['tpr_roc'] for m in metrics], axis=0)

    return metric


def visualize_metrics(metric):
    """ Visualize the metric object
    """

    roc_label = f"AUC: {metric['roc_auc_mean']:.3f}"

    print()
    print(f"True Positives: {metric['tpr_mean']:.1f}%")
    print(f"True Negatives: {metric['tnr_mean']:.1f}%")
    print(f"False Positives: {metric['fpr_mean']:.1f}%")
    print(f"False Negatives: {metric['fnr_mean']:.1f}%")
    print()
    print(f"Accuracy: {metric['acc_mean']:.1f}")
    print(f"Precision: {metric['precision_mean']:.1f}")
    print(f"Recall: {metric['recall_mean']:.1f}")
    print(roc_label)

    plt.plot(metric['fpr_roc_mean'], metric['tpr_roc_mean'], label=roc_label)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.fill_between(metric['fpr_roc_mean'],
                     metric['tpr_roc_mean'],
                     color='silver')

    plt.plot(metric['fpr_mean']/(metric['fpr_mean'] + metric['tnr_mean']),
             metric['tpr_mean']/(metric['tpr_mean'] + metric['fnr_mean']),
             marker='o',
             color='black')

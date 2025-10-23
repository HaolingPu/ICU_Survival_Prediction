# EECS 445 - Fall 2024
# Project 1 - project1.py

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.impute import KNNImputer

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)


# Q1a
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: dataframe with columns [Time, Variable, Value]

    Returns:
        a dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'max_HR': 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df.replace(to_replace=-1, value=np.nan, inplace=True)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for var in static_variables:
        value = static.loc[static["Variable"] == var]["Value"].values[0]
        feature_dict[var] = value

    # TODO  3) extract max of time-varying variables into feature dict
    for var in timeseries_variables:
        value = timeseries.loc[timeseries["Variable"] == var]["Value"].max()
        feature_dict[f"max_{var}"] = value
    return feature_dict


# for challenge part
def generate_feature_vector_challenge(df: pd.DataFrame) -> dict[str, float]:
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df.replace(to_replace=-1, value=np.nan, inplace=True)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    
    icu_type = static.loc[static["Variable"] == "ICUType"]["Value"].values[0]

    # One-hot encode ICUType (categories: 1, 2, 3, 4)
    feature_dict['ICUType_2'] = 1 if icu_type == 2 else 0
    feature_dict['ICUType_3'] = 1 if icu_type == 3 else 0
    feature_dict['ICUType_4'] = 1 if icu_type == 4 else 0

    for var in static_variables:
        if var not in ['ICUType']:
            value = static.loc[static["Variable"] == var]["Value"].values[0]
            feature_dict[var] = value

    # TODO  3) extract MEAN of time-varying variables into feature dict
    for var in timeseries_variables:
        value = timeseries.loc[timeseries["Variable"] == var]["Value"].mean()
        feature_dict[f"mean_{var}"] = value
    return feature_dict


# Q1b
def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    # TODO: implement this function according to spec
    n, d = X.shape
    for i in range(d):
        mean_value = np.nanmean(X[:, i])
        nan_rows = np.isnan(X[:,i])
        X[nan_rows, i] = mean_value
    return X

def impute_missing_values_challenge(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    imputer = KNNImputer(n_neighbors=5)
    return imputer.fit_transform(X)


# Q1c
def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (N, d) matrix.
    
    Returns:
        X: (N, d) matrix. Values are normalized per column.
    """
    # TODO: implement this function according to spec
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    # n, d = X.shape
    # for i in range(d):
    #     min_val = np.min(X[:, i])
    #     max_val = np.max(X[:, i])
    #     if min_val != max_val:
    #         X[:, i] = (X[:, i] - min_val) / (max_val - min_val)
    #     else:
    #         X[:, i] = 0.5
    # return X
    # ???
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    X_scaled = scaler.fit_transform(X)  # Scale X to [0, 1]
    return X_scaled

#Q1.d
# report the name, mean value, and interquartile range for each of the 40 features in the training set
# X train in a table
def report_40_feature (X_train, feature_names):
    print("\n ...printing 1d ...\n")
    report = []
    for i, feature_name in enumerate(feature_names):
        feature_data = X_train[:, i]
        mean_value = np.mean(feature_data)
        q1 = np.percentile(feature_data, 25)  # Compute 25th percentile (Q1)
        q3 = np.percentile(feature_data, 75)  # Compute 75th percentile (Q3)
        iqr = q3 - q1  # Compute the interquartile range
        report.append([feature_name, mean_value, iqr])    
        
    df_summary = pd.DataFrame(report, columns=['Feature Name', 'Mean', 'IQR'])
    print(df_summary)  


# def get_classifier(
#     loss: str = "logistic",
#     penalty: str | None = None,
#     C: float = 1.0,
#     class_weight: dict[int, float] | None = None,
#     kernel: str = "rbf",
#     gamma: float = 0.1,
# ) -> KernelRidge | LogisticRegression:
#     """
#     Return a classifier based on the given loss, penalty function
#     and regularization parameter C.

#     Args:
#         loss: Specifies the loss function to use.
#         penalty: The type of penalty for regularization (default: None).
#         C: Regularization strength parameter (default: 1.0).
#         class_weight: Weights associated with classes.
#         kernel : Kernel type to be used in Kernel Ridge Regression. 
#             Default is 'rbf'.
#         gamma (float): Kernel coefficient (default: 0.1).
#     Returns:
#         A classifier based on the specified arguments.
#     """
#     # TODO (optional, but highly recommended): implement function based on docstring

#     if loss == "logistic":
#         return LogisticRegression(
#             penalty=penalty,
#             C=C,
#             class_weight=class_weight,
#             solver="liblinear" if penalty == "l1" else "lbfgs",  # "liblinear" solver for l1, "lbfgs" for l2
#         )
#     elif loss == "squared_error":
#         return KernelRidge(
#             kernel=kernel,
#             alpha=1 / (2 * C),  # Inverse of regularization strength C for KernelRidge (alpha is equivalent to 1/(2C))
#             gamma=gamma,
#         )
#     else:
#         raise ValueError(f"Invalid loss function '{loss}' specified.")


def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, metric: str):
    """
    Helper function to calculate a given performance metric.
    """
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred, zero_division=0.0)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_pred, zero_division=0.0) # such values will be excluded from the average
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_scores)
    elif metric == "average_precision":
        return metrics.average_precision_score(y_true, y_scores)
    elif metric == "sensitivity":  # Sensitivity = recall for the positive class
        return metrics.recall_score(y_true, y_pred, zero_division=0.0)
    elif metric == "specificity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        if (tn + fp) == 0:
            return 0
        return tn / (tn + fp)  # True negative rate
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    metric: str = "accuracy",
    bootstrap: bool=True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X, using 1,000 
    bootstrapped samples of the test set if bootstrap is set to True. Otherwise,
    returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.
    
    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision', 
                'sensitivity', and 'specificity')
    Returns:
        if bootstrap is True: the median performance and the empirical 95% confidence interval in np.float64
        if bootstrap is False: peformance 
    """
    # TODO: Implement this function
    # This is an optional but VERY useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if isinstance(clf_trained, LogisticRegression):
        y_pred = clf_trained.predict(X)
        y_scores = clf_trained.decision_function(X)  # For metrics like AUROC or average_precision
    elif isinstance(clf_trained, KernelRidge):
        y_pred = np.sign(clf_trained.predict(X))  # Ridge returns regression values, so use sign for binary labels
        y_scores = clf_trained.predict(X)
        y_pred[y_pred >= 0] = 1
    else:
        raise ValueError("Unsupported classifier type.")

    # Single sample performance if bootstrapping is not required
    if not bootstrap:
        return calculate_metric(y_true, y_pred, y_scores, metric)
    


    n_samples = len(y_true)
    bootstrap_results = []
    for _ in range(1000):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_X, boot_y_true = X[indices], y_true[indices]
        
        # Predict on the bootstrap sample
        if isinstance(clf_trained, LogisticRegression):
            y_pred_sample = clf_trained.predict(boot_X)
            y_scores_sample = clf_trained.decision_function(boot_X)
        else:  # KernelRidge
            y_scores_sample = clf_trained.predict(boot_X)
            y_pred_sample = np.sign(y_scores_sample)
            y_pred_sample[y_pred_sample >= 0] = 1
        
        # Calculate metric for this bootstrap sample
        bootstrap_results.append(calculate_metric(boot_y_true, y_pred_sample, y_scores_sample, metric))
    
    # Calculate median and 95% confidence intervals
    median_perf = np.median(bootstrap_results)
    lower_ci = np.percentile(bootstrap_results, 2.5)
    upper_ci = np.percentile(bootstrap_results, 97.5)
    
    return median_perf, lower_ci, upper_ci


# Q2.1a
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # TODO: Implement this function

    # NOTE: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # Initialize StratifiedKFold object with k folds
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    
    # To store the performance metrics across all folds
    performance_scores = []

    # Iterate over each fold
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and validation sets
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        
        # Fit the model on the training set
        clf.fit(X_train, y_train)

        # Use the decision_function or predict method depending on the metric
        fold_performance = performance(clf, X_valid, y_valid, metric=metric, bootstrap=False)
        performance_scores.append(fold_performance)

    mean_performance = np.mean(performance_scores)
    min_performance = np.min(performance_scores)
    max_performance = np.max(performance_scores)

    # TODO: Return the average, min,and max performance scores across all fold splits in a size 3 tuple.
    performance_tuple = (mean_performance, min_performance, max_performance)
    return performance_tuple


# Q2.1b
def select_param_logreg(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over
    
    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # TODO: Implement this function
    # NOTE: You should be using your cv_performance function here
    # to evaluate the performance of each logistic regression classifier

    best_C = None
    best_penalty = None
    best_performance = -np.inf  # We want to maximize performance
    
    for C in C_range:
        for penalty in penalties:
            try:
                clf = LogisticRegression(penalty=penalty, C=C, solver='liblinear', 
                                         fit_intercept=False, random_state=seed)
            except ValueError as e:
                print(f"Skipping invalid combination of C={C}, penalty={penalty}: {e}")
                continue

            # Perform cross-validation and get the mean performance
            #??? what is the difference if k is larger?
            mean_performance, _, _ = cv_performance(clf, X, y, metric=metric, k=k)
            
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_penalty = penalty
    return best_C, best_penalty


#Q2.1c
def table_for_2c (X_train, y_train, metric_list):
    print("\n....printing table for 2.1c...\n")
    results = []
    for metric in metric_list:
        print(f"Evaluating for metric: {metric}")
        best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric)

        clf = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear', fit_intercept=False, random_state =seed)
        # Get the mean, min, and max performance from k-fold CV
        mean_perf, min_perf, max_perf = cv_performance(clf, X_train, y_train, metric=metric, k=5)
        
        results.append([metric, best_C, best_penalty, mean_perf, min_perf, max_perf])

    df_results = pd.DataFrame(results, columns=["Performance Measure", "Best C", "Best Penalty", "Mean Performance", "Min Performance", "Max Performance"])
    print(df_results)


#Q2.1d
def table_for_2d(X_train, y_train, X_test, y_test, metric_list):
    print("\n...Now the result of 2d....\n")
    best_C = 1.0
    best_penalty = 'l1'
    final_clf = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear', fit_intercept=False, random_state=seed)
    final_clf.fit(X_train, y_train)
    results = []
    for metric in metric_list:
        median_perf, lower_ci, upper_ci = performance(final_clf, X_test, y_test, metric=metric, bootstrap=True)
        results.append([metric, median_perf, f"({lower_ci:.4f}, {upper_ci:.4f})"])

    df_results = pd.DataFrame(results, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    print(df_results)


# Q2.1e
def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
    
    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        # elements of norm0 should be the number of non-zero coefficients for a given setting of C
        norm0 = []
        for C in C_range:
            # TODO Initialize clf according to C and penalty
            clf = LogisticRegression(C=C, penalty=penalty, solver='liblinear', fit_intercept=False, random_state=seed)
            # TODO Fit classifier with X and y
            clf.fit(X, y)
            # TODO: Extract learned coefficients/weights from clf into w
            # Note: Refer to sklearn.linear_model.LogisticRegression documentation
            # for attribute containing coefficients/weights of the clf object
            w = clf.coef_

            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0

            non_zero_count = np.sum(w != 0)

            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    # NOTE: plot will be saved in the current directory
    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()

#Q2.f
def feature_and_coefficient (X_train, y_train, feature_names):
    print("\n...printing 2f now... \n")
    clf = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', fit_intercept=False, random_state=seed)
    clf.fit(X_train, y_train)
    coefficients = clf.coef_.flatten()
    coeff_with_names = list(zip(feature_names, coefficients))
    sorted_coeff = sorted(coeff_with_names, key=lambda x: x[1])
    most_negative_coeffs = sorted_coeff[:4]
    most_positive_coeffs = sorted_coeff[-4:]
    # Print the results
    print("Most Negative Coefficients:")
    for feature, coeff in most_negative_coeffs:
        print(f"Feature: {feature}, Coefficient: {coeff}")

    print("\nMost Positive Coefficients:")
    for feature, coeff in most_positive_coeffs:
        print(f"Feature: {feature}, Coefficient: {coeff}")

#Q3.1b
def modified_linear_model(X_train, y_train, X_test, y_test, class_weight, metrics_list):
    print("\n...table of 3.1 and 3.2 (b).....\n")
    clf = LogisticRegression(
        penalty='l2',               # L2 regularization
        C=1.0,                      
        class_weight=class_weight, # Set class weights: Wn = 1, Wp = 50
        solver='liblinear',         
        fit_intercept=False,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    results = []
    for metric in metrics_list:
        median, ci_lower, ci_upper = performance(clf, X_test, y_test, metric=metric, bootstrap=True)
        results.append([metric, median, f"({ci_lower:.4f}, {ci_upper:.4f})"])

    df_results = pd.DataFrame(results, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    print(df_results)


#Q3.2
def find_better_weights(X_train, y_train):
    print("\n...finding the weights for 2 classes...\n")
    W_n_range = [1, 2, 3, 4, 5, 7, 10]
    W_p_range = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0]
    best_weights = None
    best_performance = -np.inf

    # Perform grid search for class weights
    for W_n in W_n_range:
        for W_p in W_p_range:
            # Define class weights
            class_weight = {-1: W_n, 1: W_p}
            
            # Logistic regression model with L2 penalty and the specified class weights
            clf = LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight=class_weight,
                solver='liblinear',
                fit_intercept=False,
                random_state=seed
            )
            mean_performance, _, _ = cv_performance(clf, X_train, y_train, metric='auroc', k=5)
            
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_weights = class_weight

    print(f"Best class weights: {best_weights}")
    print(f"Best AUROC (mean across CV folds): {best_performance}")
        

#Q3.3
def ROC_curve (X_train, y_train, X_test, y_test):
    weights_1 = {1: 1, -1: 1}
    weights_2 = {1: 5, -1: 1}

    # Logistic regression with equal class weights
    clf_1 = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight=weights_1,
        solver='liblinear',
        fit_intercept=False,
        random_state=seed
    )

    # Logistic regression with Wp = 5
    clf_2 = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight=weights_2,
        solver='liblinear',
        fit_intercept=False,
        random_state=seed
    )

    # Train both models
    clf_1.fit(X_train, y_train)
    clf_2.fit(X_train, y_train)

    y_scores_equal = clf_1.decision_function(X_test)
    y_scores_weighted = clf_2.decision_function(X_test)

    # Compute ROC curve and ROC area for both models
    fpr_equal, tpr_equal, _ = roc_curve(y_test, y_scores_equal)
    fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_scores_weighted)

    auc_equal = metrics.roc_auc_score(y_test, y_scores_equal)
    auc_weighted = metrics.roc_auc_score(y_test, y_scores_weighted)


    # Plot ROC curves
    plt.figure()

    # Plot ROC curve for Wn=1, Wp=1
    plt.plot(fpr_equal, tpr_equal, color='blue', lw=2, label=f'Wn=1, Wp=1 (AUC = {auc_equal:.2f})')

    # Plot ROC curve for Wn=1, Wp=5
    plt.plot(fpr_weighted, tpr_weighted, color='green', lw=2, label=f'Wn=1, Wp=5 (AUC = {auc_weighted:.2f})')

    # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random guessing')

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Logistic Regression with Different Class Weights')
    plt.legend(loc='lower right')
    plt.savefig("ROC_curve.png", dpi=200)
    plt.close()


#Q4.b
def kernel_performance(X_train, y_train, X_test, y_test, metric_list):
    print("\n...printing 2 tables for 4b...\n")
    Log_clf = LogisticRegression(penalty='l2', C= 1.0,fit_intercept=False, random_state=seed)
    Kernel_clf = KernelRidge(alpha=1/(2*1.0), kernel='linear')
    Log_clf.fit(X_train, y_train)
    Kernel_clf.fit(X_train, y_train)
    print("Performance of L2-regularized Log. Regression: \n")
    Log_result = []
    for metric in metric_list:
        median, ci_lower, ci_upper = performance(Log_clf, X_test, y_test, metric=metric, bootstrap=True)
        Log_result.append([metric, median, f"({ci_lower:.4f}, {ci_upper:.4f})"])
    df_log_results = pd.DataFrame(Log_result, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    print(df_log_results)


    print("Performance of Ridge Regression: \n")
    Ridge_result = []
    for metric in metric_list:
        median, ci_lower, ci_upper = performance(Kernel_clf, X_test, y_test, metric=metric, bootstrap=True)
        Ridge_result.append([metric, median, f"({ci_lower:.4f}, {ci_upper:.4f})"])
    df_ridge_results = pd.DataFrame(Ridge_result, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    print(df_ridge_results)

# Q4c
def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over
    
    Returns:
        The parameter value for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    print(f"RBF Kernel Ridge Regression Model Hyperparameter Selection based on {metric}:")
    # TODO: Implement this function acording to the docstring
    # NOTE: This function should be very similar in structure to your implementation of select_param_logreg()
    best_C = None
    best_gamma = None
    best_performance = -np.inf  # We want to maximize performance
    
    for C in C_range:
        for gamma in gamma_range:
            try:
                clf = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
            except ValueError as e:
                print(f"Skipping invalid combination of C={C}, gamma={gamma}: {e}")
                continue

            # Perform cross-validation and get the mean performance
            mean_performance, _, _ = cv_performance(clf, X, y, metric=metric, k=k)
            
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_gamma = gamma
    return best_C, best_gamma

#Q4.d
def CV_perf_for_4d (X_train, y_train, gamma_range):
    print("\n ...printing 4.d now ....\n")
    for gamma in gamma_range:
        Kernel_clf = KernelRidge(alpha= 1 / (2*1.0), kernel= 'rbf', gamma=gamma)
        mean_performance, min, max = cv_performance(Kernel_clf, X_train, y_train, metric='auroc', k=5)
        print(f"gamma:{gamma}  mean: {mean_performance}  min: {min}  max: {max}")

#Q4.e
def CI_for_4e (X_train, y_train, X_test, y_test, metric_list):
    print("\n ...printing 4.e now ....\n")
    C_range = [0.01, 0.1, 1.0, 10, 100]
    gamma_list = [0.01, 0.1, 1, 10]
    best_C, best_gamma = select_param_RBF(X_train, y_train, metric='auroc',k=5, C_range= C_range, gamma_range=gamma_list)
    print(f"C :{best_C}    gamma: {best_gamma}\n")
    final_clf = KernelRidge(alpha= 1/(2*best_C), kernel='rbf',gamma= best_gamma)
    final_clf.fit(X_train, y_train)
    results = []
    for metric in metric_list:
        median_perf, lower_ci, upper_ci = performance(final_clf, X_test, y_test, metric=metric, bootstrap=True)
        results.append([metric, median_perf, f"({lower_ci:.4f}, {upper_ci:.4f})"])
    df_results = pd.DataFrame(results, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    print(df_results)




#Q5
def challenge_feature_engineer (metric, metrics_list):
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()
    # X_train, X_test, y_train, y_test = train_test_split(X_challenge, y_challenge, test_size=0.20, stratify=y_challenge, random_state=3)
    best_C, best_penalty = select_param_logreg(X_challenge, y_challenge, metric=metric)
    clf_trained = LogisticRegression(C=best_C, penalty=best_penalty, class_weight={-1:1.0, 1:5.0}, solver='liblinear',random_state=seed)  # 'liblinear' works with both 'l1' and 'l2'
    clf_trained.fit(X_challenge, y_challenge)
    y_label = clf_trained.predict(X_heldout).astype(int)
    y_score = clf_trained.decision_function(X_heldout)
    generate_challenge_labels(y_label, y_score, "phling")
    # Compute AUROC with bootstrapping
    result = []
    for item in metrics_list:
        mean_performance, min, max = cv_performance(clf_trained, X_challenge, y_challenge, metric=item, k=5)
        result.append(([item, mean_performance, f"({min:.4f}, {max:.4f})"]))
    # print(f"{metric}: Median={median:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")
    df_results = pd.DataFrame(result, columns=["Performance Measure", "Mean Performance", "min and max"])
    print(df_results)
        # Generate predictions on X_challenge
    y_pred_challenge = clf_trained.predict(X_challenge)
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_challenge, y_pred_challenge)
    # Print the confusion matrix
    print("Confusion Matrix:\n", cm)



def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    #Q1d
    report_40_feature(X_train,feature_names)
    # #Q2c
    # table_for_2c (X_train, y_train, metric_list)
    # #Q2d
    # table_for_2d(X_train, y_train, X_test, y_test, metric_list)
    # #Q2e
    # plot_weight(X_train, y_train, C_range=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], penalties=["l1", "l2"])
    # #Q2f
    # feature_and_coefficient(X_train, y_train, feature_names)
    # #Q3.1 b
    # modified_linear_model(X_train, y_train, X_test, y_test, {-1: 1, 1: 50}, metric_list)
    # #Q3.2 a
    # find_better_weights(X_train, y_train)
    # #Q3.2 b
    # modified_linear_model(X_train, y_train, X_test, y_test, {-1: 1.0, 1: 5.0}, metric_list)
    # #3.3 a
    # ROC_curve(X_train,y_train,X_test, y_test)
    # #Q4.b
    # kernel_performance(X_train, y_train, X_test, y_test, metric_list)
    # #Q4.d
    # CV_perf_for_4d (X_train, y_train, [0.001,0.01,0.1, 1, 10, 100])
    # #Q4.e
    # CI_for_4e (X_train, y_train, X_test, y_test, metric_list)



    # Read challenge data
    # TODO: Question 5: Apply a classifier to 1heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # challenge_feature_engineer ("auroc", metric_list)
    challenge_feature_engineer ("f1_score", metric_list)
    # challenge_feature_engineer ("accuracy", metric_list)




if __name__ == "__main__":
    main()

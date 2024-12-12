import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from xgboost import XGBClassifier
import copy

def get_test_scores(X, y, preprocessor, model, param_grid, n_splits = 5, n_seeds = 5):
    """
    Given unpreprocessed full X and full y, runs an n_seeds number of Stratified K-Fold
    cross-validation hyperparameter tuning efforts, returning the results 
    (best estimators, best estimators' test scores) of the best model from each seed.

    Arguments:
        - X: unpreprocessed design matrix
        - y: target variable
        - preprocessor: a transformer, for data preprocessing in the pipeline
        - model: an estimator, for the model in the pipeline
        - param_grid: a dictionary, as the hyperparameters to grid search over
        - n_splits: number of splits in the Stratified K-Fold
        - n_seeds: number of seeds to run the Stratified K-Fold

    Returns (in order):
        - best_test_scores: list of test scores from the best model from each seed
        - best_estimators: list of best estimators from each seed
        - unpreprocessed_test_sets: list of unpreprocessed test sets from each seed
        - preprocessed_test_sets: list of preprocessed test sets from each seed
        - predicted_labels: list of predicted labels from each seed
        - baseline_scores: list of baseline test scores from each seed
    """
    best_test_scores: list[float] = list()
    best_estimators: list[Pipeline] = list()
    unpreprocessed_test_sets = list()
    preprocessed_test_sets = list()
    predicted_labels = list()
    baseline_scores = list()

    for random_state in range(n_seeds):
        print(f'Processing Seed {random_state + 1} of {n_seeds}...')
        X_other, X_test, y_other, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, stratify = y)
        best_estimator, _, _ = stratified_kfold_cv_pipe(X_other, y_other, preprocessor, model, param_grid, n_splits, random_state)

        best_test_score, X_test_preprocessed, y_pred = test_pipe(X_test, y_test, best_estimator)

        unpreprocessed_test_sets.append((X_test, y_test))
        preprocessed_test_sets.append((X_test_preprocessed, y_test))
        predicted_labels.append(y_pred)

        best_test_scores.append(best_test_score)
        best_estimators.append(best_estimator)

        # Phishing (1) is the minority class. F2 is defined only when predicting minority class (majority forces precision to be undefined).
        f2_baseline = fbeta_score(y_test, [1] * len(y_pred), beta = 2)
        baseline_scores.append(f2_baseline)

    return best_test_scores, best_estimators, unpreprocessed_test_sets, preprocessed_test_sets, predicted_labels, baseline_scores

def stratified_kfold_cv_pipe(X_other, y_other, preprocessor: TransformerMixin, model, param_grid, n_splits: int = 5, random_state: int = 42):
    """
    Given unpreprocessed non-test X and non-test y, runs a Stratified K-Fold 
    Cross-Validation Grid Search on a pipeline with a preprocessor and model.

    Arguments:
        - X: non-test, unpreprocessed design matrix
        - y: non-test, target variable
        - preprocessor: a transformer, for data preprocessing in the pipeline
        - model: an estimator, for the model in the pipeline
        - param_grid: a dictionary, as the hyperparameters to grid search over
        - n_splits: number of splits in the Stratified K-Fold
        - random_state: random state in the Stratified K-Fold

    Returns:
        - (best CV estimator, best CV params, best CV score) from grid search
    """
    pipeline = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)

    # fbeta emphasizing recall, because false negatives are more costly than false positives (i.e. not flagging a phishing URL could be worse than flagging a benign URL and the user having to override it -- the ramifications for undetected phishing are much worse)
    f2_score = make_scorer(fbeta_score, beta = 2)

    # Per documentation: "For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used." 
    # (https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html)
    grid_search = GridSearchCV(pipeline, param_grid, scoring = f2_score, cv = cv, n_jobs = -1, verbose = 4)

    grid_search.fit(X_other, y_other)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def test_pipe(X_test, y_test, pipeline: Pipeline):
    """
    Given unpreprocessed test X and test y, predicts the target variable
    and produces a test score using the best found estimator via a pipeline.

    Scores using f2, to emphasize recall.

    Arguments:
        - X_test: test design matrix
        - y_test: test target variable
        - pipeline: the best pipeline containing the best estimator

    Returns:
        - f2_score: the f2 score of the pipeline on the test data
        - X_test_preprocessed: the preprocessed test design matrix
        - y_pred: the predicted labels from the pipeline
    """
    transformed_data = pipeline.named_steps['preprocessor'].transform(X_test)
    X_test_preprocessed = pd.DataFrame(transformed_data, index = X_test.index, columns = pipeline.named_steps['preprocessor'].get_feature_names_out())

    y_pred = pipeline.predict(X_test)

    f2_score = fbeta_score(y_test, y_pred, beta = 2)

    return f2_score, X_test_preprocessed, y_pred

def get_xgb_classifier_test_scores(X, y, preprocessor: TransformerMixin, param_grid: dict, n_splits: int = 5, n_seeds: int = 5):
    """
    Given unpreprocessed full X and full y, runs an n_seeds number of Stratified K-Fold
    cross-validation hyperparameter tuning efforts, returning the results 
    (best estimators, best estimators' test scores) of the best model from each seed.

    Note: This method should only be used for XGBClassifier models.

    Arguments:
        - X: unpreprocessed design matrix
        - y: target variable
        - preprocessor: a transformer, for data preprocessing in the pipeline
        - param_grid: a dictionary, as the hyperparameters to grid search over
        - n_splits: number of splits in the Stratified K-Fold
        - n_seeds: number of seeds to run the Stratified K-Fold

    Returns (in order):
        - best_test_scores: list of test scores from the best model from each seed
        - best_estimators: list of best estimators from each seed
        - unpreprocessed_test_sets: list of unpreprocessed test sets from each seed
        - preprocessed_test_sets: list of preprocessed test sets from each seed
        - predicted_labels: list of predicted labels from each seed
        - baseline_scores: list of baseline test scores from each seed
    """
    best_test_scores: list[float] = list()
    best_estimators: list[Pipeline] = list()
    unpreprocessed_test_sets = list()
    preprocessed_test_sets = list()
    predicted_labels = list()
    baseline_scores = list()

    for random_state in range(n_seeds):
        print(f'Processing Seed {random_state + 1} of {n_seeds}...')
        X_other, X_test, y_other, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, stratify = y)
        best_estimator, fitted_preprocessor, _, _ = xgb_classifier_cv(X_other, y_other, preprocessor, param_grid, n_splits, random_state)

        # NOTE: XGBClassifier with early stopping requires a validation set. We cannot refit the best_estimator on X_other, because
        # then we would not have a validation set to use for early stopping (X_test must be used to test). Thus, the found best_estimator
        # above must be retained, despite being trained on K-1 folds of X_other.

        best_test_score, X_test_preprocessed, y_pred = xgbc_test_model(X_test, y_test, best_estimator, fitted_preprocessor)

        unpreprocessed_test_sets.append((X_test, y_test))
        preprocessed_test_sets.append((X_test_preprocessed, y_test))
        predicted_labels.append(y_pred)

        best_test_scores.append(best_test_score)
        best_estimators.append(best_estimator)

        # Phishing (1) is the minority class. F2 is defined only when predicting minority class (majority forces precision to be undefined).
        f2_baseline = fbeta_score(y_test, [1] * len(y_pred), beta = 2)
        baseline_scores.append(f2_baseline)

    return best_test_scores, best_estimators, unpreprocessed_test_sets, preprocessed_test_sets, predicted_labels, baseline_scores

def xgb_classifier_cv(X_other, y_other, preprocessor: TransformerMixin, param_grid: dict, n_splits: int = 5, random_state: int = 42):
    """
    Given unpreprocessed non-test X and non-test y, runs a Stratified K-Fold 
    Cross-Validation Grid Search with a preprocessor and XGBClassifier.

    Note: This method should only be used for XGBClassifier models.

    Arguments:
        - X: non-test, unpreprocessed design matrix
        - y: non-test, target variable
        - preprocessor: a transformer, for data preprocessing in the pipeline
        - param_grid: a dictionary, as the hyperparameters to grid search over
        - n_splits: number of splits in the Stratified K-Fold
        - random_state: random state in the Stratified K-Fold

    Returns:
        - (best CV model, fitted preprocessor for best CV model, best CV params, best CV score) from grid search
    """
    best_model = None
    fitted_preprocessor = None
    best_params = None
    best_score = float('-inf')

    for params in ParameterGrid(param_grid):
        # print(f'Processing Parameter Combination {params}...')
        model = XGBClassifier(**params, early_stopping_rounds = 10, n_jobs = -1)

        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)

        cv_scores = list()
        
        for train_idx, val_idx in skf.split(X_other, y_other):
            X_train, y_train = X_other.iloc[train_idx], y_other.iloc[train_idx]
            X_val, y_val = X_other.iloc[val_idx], y_other.iloc[val_idx]

            X_train_preprocessed = pd.DataFrame(preprocessor.fit_transform(X_train), index = X_train.index, columns = preprocessor.get_feature_names_out())
            X_val_preprocessed = pd.DataFrame(preprocessor.transform(X_val), index = X_val.index, columns = preprocessor.get_feature_names_out())

            model.fit(X_train_preprocessed, y_train, eval_set = [(X_val_preprocessed, y_val)], verbose = False)

            y_pred = model.predict(X_val_preprocessed)

            score = fbeta_score(y_val, y_pred, beta = 2)

            cv_scores.append(score)

        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_model = model
            fitted_preprocessor = copy.deepcopy(preprocessor)
            best_params = params
            best_score = mean_score

    return best_model, fitted_preprocessor, best_params, best_score

def xgbc_test_model(X_test, y_test, model, fitted_preprocessor):
    """
    Given unpreprocessed test X and test y, predicts the target variable
    and produces a test score using the best found model.

    Scores using f2, to emphasize recall.

    Note: This method should only be used for XGBClassifier models.

    Arguments:
        - X_test: test design matrix
        - y_test: test target variable
        - model: the best model found using SKF CV
        - fitted_preprocessor: the fitted preprocessor matching the training data for the best model

    Returns:
        - f2_score: the f2 score of the pipeline on the test data
        - X_test_preprocessed: the preprocessed test design matrix
        - y_pred: the predicted labels from the model
    """
    X_test_preprocessed = pd.DataFrame(fitted_preprocessor.transform(X_test), index = X_test.index, columns = fitted_preprocessor.get_feature_names_out())

    y_pred = model.predict(X_test_preprocessed)

    f2_score = fbeta_score(y_test, y_pred, beta = 2)

    return f2_score, X_test_preprocessed, y_pred
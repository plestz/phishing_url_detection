import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_test_scores(X, y, preprocessor, model, param_grid, n_splits = 5, n_seeds = 5):
    """
    Given unpreprocessed full X and full y, runs an n_seeds number of Stratified K-Fold
    cross-validation hyperparameter tuning efforts, returning the results 
    (best estimators, best estimators' test scores)of the best model from each seed.

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
        - best_test_score: the best test score from all seeds
        - best_estimator: the best estimator from all seeds
    """
    best_test_scores: list[float] = list()
    best_estimators: list[Pipeline] = list()

    for random_state in range(n_seeds):
        print(f'Processing Seed {random_state + 1} of {n_seeds}...')
        X_other, X_test, y_other, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
        best_estimator, _, _ = stratified_kfold_cv_pipe(X_other, y_other, preprocessor, model, param_grid, n_splits, random_state)

        best_estimator.fit(X_other, y_other)
        best_test_score = test_pipe(X_test, y_test, best_estimator)

        best_test_scores.append(best_test_score)
        best_estimators.append(best_estimator)

    best_idx = np.argmax(best_test_scores)

    return best_test_scores, best_estimators, best_test_scores[best_idx], best_estimators[best_idx]

def stratified_kfold_cv_pipe(X, y, preprocessor: TransformerMixin, model, param_grid, n_splits: int = 5, random_state: int = 42):
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

    grid_search.fit(X, y)

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
    """
    y_pred = pipeline.predict(X_test)

    f2_score = fbeta_score(y_test, y_pred, beta = 2)

    return f2_score
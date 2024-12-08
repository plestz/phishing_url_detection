from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold

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
    """
    y_pred = pipeline.predict(X_test)

    f2_score = fbeta_score(y_test, y_pred, beta = 2)

    return f2_score
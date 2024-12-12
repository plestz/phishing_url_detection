import pickle

AVAILABLE_MODELS = ['logistic_regression', 'random_forest_classifier', 'support_vector_classifier', 'xgb_classifier']

def load_data_for(model_name: str) -> tuple:
    """
    Given an available model_name, provides all relevant stored data to that
    model produced from this project in models.ipynb. 

    Note that if multiple seeds were used in model training, there will be
    that many seeds-worth of each returned data variable.

    Arguments:
        model_name -- The name of the model to load data for. Must be in AVAILABLE_MODELS.

    Returns:
        model_results -- A dictionary containing all relevant data for the model.
            - Includes best_estimators, best_scores, baseline_scores
              X_test_unpreprocessed, X_test_preprocessed, y_test, predicted_labels
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f'Files for model {model_name} are unavailable to load in this project.')
    
    filepath_best = f'../results/{model_name}_best_estimators.save'
    filepath_data = f'../results/{model_name}_test_data.save'

    with open(filepath_best, 'rb') as f:
        best_estimators, best_scores, baseline_scores = pickle.load(f)
    with open(filepath_data, 'rb') as f:
        unpreprocessed_data, preprocessed_data, predicted_labels = pickle.load(f)

    X_test_unpreprocessed, y_test = list(zip(*unpreprocessed_data))
    X_test_preprocessed, _ = list(zip(*preprocessed_data))

    model_results = {
        'best_estimators': best_estimators,
        'best_scores': best_scores,
        'baseline_scores': baseline_scores,
        'X_test_unpreprocessed': X_test_unpreprocessed,
        'X_test_preprocessed': X_test_preprocessed,
        'y_test': y_test,
        'predicted_labels': predicted_labels
    }

    return model_results
# Phishing URL Detection

This project is a full ML pipeline (from EDA to analysis) revolving around
machine learning models to detect phishing URL scams on the World Wide Web.

## Navigating the Project

### /data

The PhiUSIIL .csv dataset can be found in this folder.

The original dataset can be found [here](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

Note: For ease of use in analysis, all (un)preprocessed datasets were stored aligned
with their corresponding models in /results' `.save` files.

### /src

All code is in this folder. In order of the corresponding step in the pipeline...

- `eda.ipynb` contains all Exploratory Data Analysis, primarily visualizations
of different features and their relationship (and correlation) with the target
label.

- `pipeline_func.py` contains all functions relevant to training each of
the four models: Logistic Regresison, Random Forest Classifier, 
Support Vector Classifier, and XGBoost Classifier. The primary function for
the first three models (not XGBoost) is *get_test_scores*, which performs
n_seeds worth of n_splits StratifiedKFold Cross-Validation operations on
the non-test dataset to find the n_seeds best estimators (and corresponding)
test scores (found by using the CV best estimator on the holdout test set). 
For various analysis purposes, the best estimators' correponding (un)preprocessed
test datasets, true labels, predicted labels, and baseline scores are all included
in the return tuple. The analogous function for XGBoost is *get_xgb_classifier_test_scores*.

- `models.ipynb` contains the training and storage of all of the results of
each of the four models (aforementioned in `pipeline_func.py`). Hyperparameter
combinations, model instances (for non-XGBoost), and preprocessors are all
created here (to be passed into the test score functions). 

- `data_loader.py` contains a function that returns the specified model's
stored `.save` data in a cohesive way, to be used in `analysis.ipynb`.

- `analysis.ipynb` contains a comprehensive analysis of model performance,
in particular for the best model (one of the XGBoost Classifier estimators). 
Elements of analysis include an F2 score comparison (including means and
standard deviations), a baseline F2 score comparison (and related 
visualizations), confusion matrix plots, a ROC Curve (including ROC AUC score),
a Precision-Recall curve, seven global feature importance metrics measures
(Permutation Feature Importance, SHAP Global Summary, and 5 XGBoost importance
metrics), and local feature importance examinations for specific points (using
SHAP local importances). 

### /figures

This folder contains all relevant figures used in the report in /report.

All of these figures were generated in one of the above `.ipynb` files.

### /results

This folder contains `.save` files (originating from `models.ipynb`) containing,
for each model, the corresponding n_seeds best estimators, corresponding 
best test scores, unpreprocessed and preprocessed test sets, true labels,
predicted labels, and baseline scores. 

A simple extraction mechanism for this data can be found in `../src/data_loader.py`.

### /report

This folder contains the official PDF report describing this project. Contained
sections include I. Introduction, II. EDA, III. Methods, IV. Results, V. Outlook,
and VI. References.

## Running Locally

This project was completed using Python 3.12.5.

For the complete list of packages, refer to `package_list.txt`.

For easy installation, utilize `environment.yml`. You can recreate this
environment in Conda using *conda env create -f environment.yml*. 
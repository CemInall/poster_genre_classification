from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from ml_models import PCA_images, data_splitting, show_model_results
# Codes modified from sklearn's classifier website and grid search website


def do_grid_search(model, parameters, X_train, y_train, X_val, y_val):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'weighted_f1': make_scorer(f1_score, average='weighted')
    }
    grid_search = GridSearchCV(
        estimator=model, param_grid=parameters, cv=5, scoring=scoring, refit="accuracy")
    grid_search.fit(X_train, y_train)
    print(model.__class__.__name__)
    show_model_results(grid_search, "validation", X_val, y_val)
    print(grid_search.best_params_)

def random_forest_search(X_train, y_train, X_val, y_val):
    parameters = {
        'n_estimators': [5 ,50, 100],
        'max_depth': [4, 8, 16, 32],
        'criterion': ['entropy']
    }

    model = RandomForestClassifier(random_state=42)
    do_grid_search(model, parameters, X_train, y_train, X_val, y_val)


def knn_grid_search(X_train, y_train, X_val, y_val):
    parameters = {
        'n_neighbors': [1, 3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    model = KNeighborsClassifier()
    do_grid_search(model, parameters, X_train, y_train, X_val, y_val)


def logistic_regression_search(X_train, y_train, X_val, y_val):
    parameters = {
        'solver': ['saga', 'liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10]
    }

    model = LogisticRegression(random_state=42)
    do_grid_search(model, parameters, X_train, y_train, X_val, y_val)


def xgboost_search(X_train, y_train, X_val, y_val):
    parameters = {
        'n_estimators': [5 ,50, 100],
        'max_depth': [3, 5],
    }

    model = XGBClassifier(random_state=42)
    do_grid_search(model, parameters, X_train, y_train, X_val, y_val)


#do_machine_learning(movielabels, images)

def do_all_grid_searches(movielabels, images):
    pca_images = PCA_images(images)
    X, y = pca_images, movielabels['target']
    X_train, y_train, X_test, y_test, X_val, y_val = data_splitting(X, y)
    #random_forest_search(X_train, y_train, X_val, y_val)
    #knn_grid_search(X_train, y_train, X_val, y_val)
    logistic_regression_search(X_train, y_train, X_val, y_val)
    #xgboost_search(X_train, y_train, X_val, y_val)
    


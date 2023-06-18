from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
# Codes modified from sklearn's classifier website and grid search website


def random_Forest_search(X_train, y_train, X_test, y_test):
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, 14],
        'criterion': ['entropy']
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'weighted_f1': make_scorer(f1_score, average='weighted')
    }

    rfc = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(
        estimator=rfc, param_grid=param_grid_rf, cv=5, scoring=scoring, refit="accuracy")
    grid_search_rf.fit(X_train, y_train)
    y_pred = grid_search_rf.predict(X_test)

    acc_scores = [accuracy_score(y_test, y_pred)]
    f1_scores = [f1_score(y_test, y_pred, average='weighted')]
    mean_acc_score = sum(acc_scores) / len(acc_scores)
    mean_f1_score = sum(f1_scores) / len(f1_scores)

    return grid_search_rf.best_params_, acc_scores, f1_scores, mean_acc_score, mean_f1_score


def knn_grid_search(X_train, y_train, X_test, y_test):
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'weighted_f1': make_scorer(f1_score, average='weighted')
    }

    knn_classifier = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(
        knn_classifier, param_grid=param_grid_knn, cv=5, scoring=scoring, refit="accuracy", n_jobs=-1)
    grid_search_knn.fit(X_train, y_train)
    y_pred = grid_search_knn.predict(X_test)

    acc_scores = [accuracy_score(y_test, y_pred)]
    f1_scores = [f1_score(y_test, y_pred, average='weighted')]
    mean_acc_score = sum(acc_scores) / len(acc_scores)
    mean_f1_score = sum(f1_scores) / len(f1_scores)

    return grid_search_knn.best_params_, acc_scores, f1_scores, mean_acc_score, mean_f1_score


def logistic_regression_search(X_train, y_train, X_test, y_test):
    param_grid = {
        'solver': ['saga', 'liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10]
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'weighted_f1': make_scorer(f1_score, average='weighted')
    }

    logistic_reg = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(estimator=logistic_reg, param_grid=param_grid,
                               cv=5, scoring=scoring, refit="accuracy")
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    acc_scores = [accuracy_score(y_test, y_pred)]
    f1_scores = [f1_score(y_test, y_pred, average='weighted')]
    mean_acc_score = sum(acc_scores) / len(acc_scores)
    mean_f1_score = sum(f1_scores) / len(f1_scores)

    return grid_search.best_params_, acc_scores, f1_scores, mean_acc_score, mean_f1_score


def xgboost_search(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'weighted_f1': make_scorer(f1_score, average='weighted')
    }

    xgb_classifier = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid,
                               cv=5, scoring=scoring, refit="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    acc_scores = [accuracy_score(y_test, y_pred)]
    f1_scores = [f1_score(y_test, y_pred, average='weighted')]
    mean_acc_score = sum(acc_scores) / len(acc_scores)
    mean_f1_score = sum(f1_scores) / len(f1_scores)

    return grid_search.best_params_, acc_scores, f1_scores, mean_acc_score, mean_f1_score

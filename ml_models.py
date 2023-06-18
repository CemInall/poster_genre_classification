from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import settings
import os


def PCA_images(preprocessed_images):
    filename = f'{settings.data_folder}/pca_preprocessed_images_{settings.IMG_SIZE[0]}x{settings.IMG_SIZE[1]}.npy'
    if not os.path.isfile(filename):
        preprocessed_images = preprocessed_images.reshape(preprocessed_images.shape[0], -1)
        pca = PCA(n_components=300)
        pca.fit(preprocessed_images)
        print(f'Explained variance ratio: {sum(pca.explained_variance_ratio_)}')
        transformed_data = pca.transform(preprocessed_images)
        np.save(filename, transformed_data)
    return np.load(filename)


def data_splitting(preprocessed_images, y):
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, y_train, X_test, y_test, X_val, y_val


def show_model_results(model, mode, X, y):
    ypred = model.predict(X)
    acc_scores = [accuracy_score(y, ypred)]
    f1_scores = [f1_score(y, ypred, average='weighted')]
    mean_acc_score = sum(acc_scores) / len(acc_scores)
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    cm = confusion_matrix(y, ypred)
    class_names = model.classes_
    # save_confusion_matrix(cm_val, class_names_val, 'Logistic Regression', 'Validation Set', str(settings.IMG_SIZE))
    print(f'{model.__class__.__name__} {mode} accuracy: {mean_acc_score}, f1: {mean_f1_score}')


def do_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    logistic_reg = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
    logistic_reg.fit(X_train, y_train)

    show_model_results(logistic_reg, 'Train', X_train, y_train)
    show_model_results(logistic_reg, 'Test', X_test, y_test)

    return logistic_reg

def do_knn(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    show_model_results(knn, 'Train', X_train, y_train)
    show_model_results(knn, 'Test', X_test, y_test)

    return knn

def do_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.svm import SVC
    svm = SVC(kernel='poly', C=1, random_state=42)
    svm.fit(X_train, y_train)

    show_model_results(svm, 'Train', X_train, y_train)
    show_model_results(svm, 'Test', X_test, y_test)

    return svm

def do_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=42)
    random_forest.fit(X_train, y_train)

    show_model_results(random_forest, 'Train', X_train, y_train)
    show_model_results(random_forest, 'Test', X_test, y_test)

    return random_forest


def do_machine_learning(movielabels, images):
    pca_images = PCA_images(images)

    X, y = pca_images, movielabels['target']
    X_train, y_train, X_test, y_test, X_val, y_val = data_splitting(X, y)

    random_forest_model = do_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    logreg_model = do_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    knn_model = do_knn(X_train, y_train, X_val, y_val, X_test, y_test)
    svm_model = do_svm(X_train, y_train, X_val, y_val, X_test, y_test)

# def do_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
#     from xgboost import XGBClassifier
#     xgb = XGBClassifier()
#     xgb.fit(X_train, y_train)

#     show_model_results(xgb, 'Train', X_train, y_train)
#     show_model_results(xgb, 'Test', X_test, y_test)

#     return xgb

# def do_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
#     from sklearn.ensemble import VotingClassifier
#     from sklearn.svm import SVC
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.ensemble import RandomForestClassifier
#     from xgboost import XGBClassifier
#     svm = SVC(kernel='linear', C=1, random_state=42)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
#     xgb = XGBClassifier()
#     ensemble = VotingClassifier(estimators=[('svm', svm), ('knn', knn), ('random_forest', random_forest), ('xgb', xgb)], voting='hard')
#     ensemble.fit(X_train, y_train)

#     show_model_results(ensemble, 'Train', X_train, y_train)
#     show_model_results(ensemble, 'Test', X_test, y_test)

#     return ensemble
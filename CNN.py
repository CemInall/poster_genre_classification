import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout,  MaxPooling2D, LeakyReLU, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import settings
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def do_deep_learning(movielabels, images):
    y = movielabels[["action", "comedy", "drama", "horror"]].values
    X_train, X_test, y_train, y_test = data_splitting_CNN(images, y)
    model, history = CNN_model(X_train, y_train)
    analysis = Error_Analysis(model, y_test, X_test)
    gr = CNN_graph(history)
    print(analysis)
    print(gr)


def data_splitting_CNN(X, y):
    X = X.reshape(-1, settings.IMG_SIZE[1], settings.IMG_SIZE[0],3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test


precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()


def f1(y_true, y_pred):
    precision_metric.update_state(y_true, y_pred)
    recall_metric.update_state(y_true, y_pred)
    precision = precision_metric.result()
    recall = recall_metric.result()
    f1 = 2 * ((precision * recall) / (precision +
              recall + tf.keras.backend.epsilon()))
    return f1


def CNN_model(X_train, y_train):
    model = keras.Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=(140, 100, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(4, activation='sigmoid'))

    # https://stackoverflow.
    # /questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001  # tf.keras.optimizers.Adam(learning_rate=0.01) tf.keras.optimizers.RMSprop
                                                     ), loss="categorical_crossentropy", metrics=["accuracy", f1])  # categorical_crossentropy
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=64, callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    # history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.3,
    #                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)])

    model.summary()
    return model, history


def evaluate_model(model, X_test, y_test):
    loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    # Convert the probability distribution to binary labels
    y_pred = (y_pred > 0.5).astype('int')
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    print(f'Test F1-score: {f1:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('Confusion matrix:')
    print(cm)  # I am unable to get a plausible confusion matrix

    # reference: https://github.com/bnsreenu/python_for_microscopists/blob/master/142-multi_label_classification.py
# plot the training and validation accuracy and loss at each epoch


def CNN_graph(history: keras.callbacks.History):
    history = history.history
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    f1 = history['f1']
    val_f1 = history['val_f1']
    plt.plot(epochs, f1, 'y', label='Training F1 score')
    plt.plot(epochs, val_f1, 'r', label='Validation F1 score')
    plt.title('Training and validation F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.show()

# Taken from machine learning notebook of the university


def Error_Analysis(model, y_test, X_test):
    y_pred = model.predict(X_test)
    pred = np.round(y_pred)

    index = 0
    misclassifiedIndexes = []
    for label, predict in zip(y_test, pred):
        if not np.array_equal(label, predict):
            misclassifiedIndexes.append(index)
        index += 1

    plt.figure(figsize=(25, 4))
    for plotIndex, badIndex in enumerate(misclassifiedIndexes[6:10]):
        plt.subplot(1, 5, plotIndex + 1)
        plt.imshow(np.squeeze(X_test[badIndex]), cmap=plt.cm.gray)
        plt.title('Predicted: {}, Actual: {}'.format(
            pred[badIndex], y_test[badIndex]), fontsize=5)
    plt.subplots_adjust(wspace=0.1)
    plt.show()

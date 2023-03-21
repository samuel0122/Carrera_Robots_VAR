#!/usr/bin/env python3
"""
Backpropagation algoritm
    @authors: Samuel Oliva Bulpitt, Luis Jesús Marhuenda Tendero
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import random
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical

def convertOneHotEncodedLabelsToDecimal(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)

def convertDecimalLabelsToOneHotEncoded(decimalLabels, nb_classes):
    return to_categorical(decimalLabels, nb_classes)

def get_csv_data(file, nb_classes):
    # Read CSV file
    data = pd.read_csv(file, sep=";")

    # Split data into inputs and outputs
    inputData = data.iloc[:, :-nb_classes].values
    outputs = data.iloc[:, -nb_classes:].values
    
    return inputData, outputs


def create_model(inputs, outputs):
    """
    BUILD KERAS MODEL USING FUNCTIONAL API
    """
    # Inputs layer
    input_layer = layers.Input(inputs)

    # Hidden layers
    x = input_layer
    
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)

    to_output_layer = x

    # Output layer
    output_layer = layers.Dense(outputs, activation="softmax")(to_output_layer)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def undersampleDataset(X, y):
    labelsIsOneHotEncoded = y[0] is list

    # Create an undersampler to balance the classes
    undersampler = RandomUnderSampler(sampling_strategy='not minority')

    if labelsIsOneHotEncoded:
        # Pass to categorical
        nb_classes = len(y[0])
        y = convertOneHotEncodedLabelsToDecimal(y)

    # Resample the dataset
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    if labelsIsOneHotEncoded:
        # Pass to categorical
        y_resampled = convertDecimalLabelsToOneHotEncoded(y_resampled, nb_classes)

    return X_resampled, y_resampled


def oversampleDataset(X, y):
    # Create an oversampler to balance the classes
    oversampler = SMOTE(sampling_strategy='not majority')

    # Resample the dataset
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    return X_resampled, y_resampled


def countNumberOfEachClass(one_hot_labels):

    # Convert the one-hot encoded label matrix to decimal values
    decimal_labels = convertOneHotEncodedLabelsToDecimal(one_hot_labels)

    # Count the number of occurrences of each number in the list
    counts = Counter(decimal_labels)

    # Print the counts
    for number in counts:
        print(f'There a total of {counts[number]} elements of class {number}')

    print()


def appendDirectoriesToFileNames(fileNames, directory):
    return [directory+file for file in fileNames]

def getDataset(file, nb_classes, trainTestSplit, undersample = False):

    print('Loading dataset...')
    X, y = get_csv_data(file=file, nb_classes=nb_classes)

    if undersample:
        X, y = undersampleDataset(X, y)

    numberElements = len(X)
    indexes = list(range(numberElements))
    random.shuffle(indexes)

    numberExampleSplit = int(numberElements*trainTestSplit)
    trainIndexes = indexes[:numberExampleSplit]
    testIndexes  = indexes[numberExampleSplit:]

    x_train, x_test = X[trainIndexes], X[testIndexes]
    y_train, y_test = y[trainIndexes], y[testIndexes]

    return (x_train, y_train), (x_test, y_test)

def train(model, x_train, y_train, batch, epoch, val_split):


    print(f'Training with {x_train.shape} examples and {y_train.shape} labels...')
    
    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model with validation data    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=directorySaved+'best_model.h5', save_best_only=True, monitor='accuracy', mode='max')
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, validation_split=val_split, verbose=0, callbacks=[checkpoint_callback])

    # Evaluate the model on the test data
    
    return model, history

def evaluate(model, x_test, y_test):

    print('Evaluating...')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


def printHistory(history):
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

doTraining = False
directorySaved = '/home/samuel/P1_Carrera_de_robots/'
filesNames5 = ['datos5_1.csv', 'datos5_2.csv', 'datos5_3.csv', 'datos5_4.csv', 'datos5_5.csv', 'datos5_6.csv']
filesNamesE2 = ['datosE2_3.csv', 'datosE2_4.csv', 'datosE2_5.csv', 'datosE2_6.csv']
filesNamesE5 = ['datosE5_3.csv', 'datosE5_4.csv', 'datosE5_5.csv', 'datosE5_6.csv']
filesNames5 = appendDirectoriesToFileNames(fileNames=filesNames5, directory=directorySaved)
filesNamesE2 = appendDirectoriesToFileNames(fileNames=filesNamesE2, directory=directorySaved)
filesNamesE5 = appendDirectoriesToFileNames(fileNames=filesNamesE5, directory=directorySaved)
trainTestSplit = 0.85
nb_classes = 3

random.seed(0)

(x_train, y_train), (x_test, y_test) = ([], []), ([], [])

# Get dataset
for fileName in filesNames5:
    (x_tr, y_tr), (x_tst, y_tst) = getDataset(file=fileName, nb_classes=nb_classes, trainTestSplit=trainTestSplit, undersample=False)
    x_train.extend(x_tr)
    y_train.extend(y_tr)
    x_test.extend(x_tst)
    y_test.extend(y_tst)

x_train, y_train = np.asarray(x_train), np.asarray(y_train)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)

print(f'Count train:')
countNumberOfEachClass(y_train)

print(f'Count test:')
countNumberOfEachClass(y_test)

if doTraining:
    netInputs = len(x_train[0])

    # Generate model
    model = create_model(inputs=netInputs, outputs=nb_classes)

    # Train the model
    model, hist = train(model=model, x_train=x_train, y_train=y_train, batch=16, epoch=100, val_split=0.95)

    # Evaluate the model
    evaluate(x_test=x_test, y_test=y_test, model=model)

    # Print history
    printHistory(hist)

else:
    # Loads model
    model = keras.models.load_model('/home/samuel/P1_Carrera_de_robots/src/all_listeners/models/models1/model_right.h5')

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    print(x_train[0])
    # Predict's 10 cases
    prediction = model.predict(np.asarray([x_train[0]]))

    print(prediction)
    # Prints them
    print(f'Prediction: {convertOneHotEncodedLabelsToDecimal(prediction)}')
    print(f'Actual:     {convertOneHotEncodedLabelsToDecimal(y_train[:10])}')
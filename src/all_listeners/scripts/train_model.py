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

def get_csv_data(file, inputs):
    # Read CSV file
    data = pd.read_csv('/home/samuel/P1_Carrera_de_robots/src/' + file, sep=";")

    # Split data into inputs and outputs
    inputData = data.iloc[:, :inputs].values
    outputs = data.iloc[:, inputs:].values
    
    return inputData, outputs


def create_model(inputs, outputs):
    """
    BUILD KERAS MODEL USING FUNCTIONAL API
    """
    # Inputs layer
    input_layer = layers.Input(inputs)

    # Hidden layers
    dense_layer1 = layers.Dense(256, activation="relu")(input_layer)
    drop_out1 = layers.Dropout(0.3)(dense_layer1)
    dense_layer2 = layers.Dense(128, activation="relu")(drop_out1)
    drop_out2 = layers.Dropout(0.3)(dense_layer2)
    dense_layer3 = layers.Dense(64, activation="relu")(drop_out2)
    
    # Output layer
    output_layer = layers.Dense(outputs, activation="softmax")(dense_layer3)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def undersampleDataset(X, y):
    # Create an undersampler to balance the classes
    undersampler = RandomUnderSampler(sampling_strategy='not minority')

    # Resample the dataset
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    return X_resampled, y_resampled


def oversampleDataset(X, y):
    # Create an oversampler to balance the classes
    oversampler = SMOTE(sampling_strategy='not majority')

    # Resample the dataset
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    return X_resampled, y_resampled


def countNumberOfEachClass(one_hot_labels):

    # Convert the one-hot encoded label matrix to decimal values
    decimal_labels = np.argmax(one_hot_labels, axis=1)

    # Count the number of occurrences of each number in the list
    counts = Counter(decimal_labels)

    # Print the counts
    for number in counts:
        print(f'There a total of {counts[number]} elements of class {number}')


def getDataset(file, inputs, trainTestSplit, undersample = False):

    print('Loading dataset...')
    X, y = get_csv_data(file=file, inputs=inputs)

    if undersample:
        X, y = oversampleDataset(X, y)

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

    print('Training...')
    
    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model with validation data    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/home/samuel/P1_Carrera_de_robots/src/best_model.h5', save_best_only=True, monitor='accuracy', mode='max')
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


fileName = 'datos1.csv'
trainTestSplit = 0.85
netInputs = 5

nb_classes = 3

random.seed(0)

# Get dataset
(x_train, y_train), (x_test, y_test) = getDataset(file=fileName, inputs=netInputs, trainTestSplit=trainTestSplit)

# x_train, y_train = oversampleDataset(x_train, y_train)

print('Count train:')
countNumberOfEachClass(y_train)

print('Count test:')
countNumberOfEachClass(y_test)

# Generate model
model = create_model(inputs=netInputs, outputs=nb_classes)

# Train the model
model, hist = train(model=model, x_train=x_train, y_train=y_train, batch=16, epoch=50, val_split=0.99)

# Evaluate the model
evaluate(x_test=x_test, y_test=y_test, model=model)

# Print history
printHistory(hist)

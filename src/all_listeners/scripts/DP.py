import tensorflow as tf

# If you want to plot SHAP values, turn this True
DISABLE_EAGER_EXECUTION = False

if DISABLE_EAGER_EXECUTION:
    !pip install shap
    import shap
    tf.compat.v1.disable_eager_execution()
    print("Tensorflow eager_execution is dissabled. SHAP Value will be avaliable, but you can't use ImageDataAugmentation (to equilibrate binary dataset or to duplicate dataset size).")
else:
    print("Tensorflow eager_execution is NOT dissabled. SHAP Value won't be avaliable. To dissable it, change DISABLE_EAGER_EXECUTION variable.")

"""
#################################################################################################
#################################################################################################
########################################     IMPORTS     ########################################
#################################################################################################
#################################################################################################
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string
import time
import warnings
from ast import Return
from tokenize import String

import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import EarlyStopping
from matplotlib.offsetbox import AuxTransformBox
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from tensorflow import keras
from traitlets import Int

# warnings.filterwarnings('ignore')
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import wilcoxon
from keras.utils import to_categorical


import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
# import tensorflow.compat.v1.keras.backend as K


"""
################################################################################################
################################################################################################
########################################     MODELS     ########################################
################################################################################################
################################################################################################
"""
# Model 1
def cnn_model1(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(120, (5,5), activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)
    """
    return Sequential([
        layers.Conv2D(6, (5,5), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(16, (5,5), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(120, (5,5), activation='relu', kernel_initializer='he_uniform'),
        layers.Flatten(),

        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dropout(0.5),

        layers.Dense(nb_classes, activation='softmax')
    ])
    """


# Model 2
def cnn_model2(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(120, (5,5), activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(60, activation='relu')(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)
    """
    return Sequential([
        layers.Conv2D(6, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(16, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(120, (5,5), activation='relu'),
        layers.Flatten(),

        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(60, activation='relu'),

        layers.Dense(nb_classes, activation='softmax')
    ])
    """


# Model 3
def cnn_model3(imagesShape, nb_classes):
    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(60, activation='relu')(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)
    """
    return Sequential([
        layers.Conv2D(6, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(16, (5,5), activation='relu'),
        layers.Flatten(),

        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(60, activation='relu'),

        layers.Dense(nb_classes, activation='softmax')
    ])
    """


# Model 4
def cnn_model4(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(60, activation='relu')(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)
    """
    return Sequential([
        layers.Conv2D(6, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(16, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),

        layers.Dense(100, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(60, activation='relu'),

        layers.Dense(nb_classes, activation='softmax')
    ])
    """


# Model 5
def cnn_model5(imagesShape, nb_classes):
    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)
    """ 
    return Sequential([
        layers.Conv2D(6, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(16, (5,5), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(nb_classes, activation='softmax')
    ])
    """


#Model 6
def cnn_model6(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.Conv2D(16, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)


# Model 7
def cnn_model7(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(16, (5,5), activation='relu')(x)
    x = layers.Conv2D(16, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)


    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)


# Model 8
def cnn_model8(imagesShape, nb_classes):

    input_lay = layers.Input(shape=imagesShape)
    x = input_lay

    x = layers.Rescaling(scale=1./255)(x)

    x = layers.Conv2D(6, (5,5), padding='same', activation='relu')(x)
    x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(16, (5,5), padding='same', activation='relu')(x)
    x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Separation in to two paths
    firstDropout = x

    # First path
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(120, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(120, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # End first path with flatten
    x = layers.Flatten()(x)

    # Second path
    residual = layers.Conv2D(120, 1, strides=2, padding="same", activation='relu')(firstDropout)
    residual = layers.Flatten()(residual)

    # firstDropout = layers.Conv2D(64, (7,7), activation='relu')(firstDropout)
    # firstDropout = layers.Conv2D(64, (3,3), activation='relu')(firstDropout)
    # firstDropout = layers.MaxPool2D((4, 4))(firstDropout)

    x = layers.add([x, residual])  # Add back second dropout

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output_lay = layers.Dense(nb_classes, activation='softmax')(x)

    return keras.Model(input_lay, output_lay)


# A random model
def cnn_random_model(imagesShape, nb_classes):
    input_shape = imagesShape
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(nb_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


"""
####################################################################################################################################
####################################################################################################################################
########################################              GENERAL HELPER FUNCTIONS              ########################################
####################################################################################################################################
####################################################################################################################################
"""


def CH_plot_learning_curves(hist):
    """
    Plots the learning curve using the history recieved from training a model.
    """
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Curvas de aprendizaje')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
    plt.show()


def CH_plot_symbols(X,y,n=15):
    """
    Plots some images selected randomly.
    """
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(25, 2))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i],:,:,:])
        plt.gray()
        ax.set_title(f'{y[index[i]]}-{index[i]}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def CH_format_example(image, img_rows=32, img_cols=32):
    """
    Resizes the image to img_rows & img_cols
    """
    # image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    # image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (img_rows, img_cols))
    return image


def CH_load_data(img_rows, img_cols, name="colorectal_histology") -> tuple:
    """
    Loads the colorectal_histology dataset formatted
    """
    train_ds = tfds.load(name, split=tfds.Split.TRAIN, batch_size=-1)
    train_ds['image'] = tf.map_fn(CH_format_example, train_ds['image'], dtype=tf.float32)
    numpy_ds = tfds.as_numpy(train_ds)
    X, y = numpy_ds['image'], numpy_ds['label']

    return np.array(X), np.array(y)


# Loads the dataset with correct label format
def CH_getDataset(nb_classes, img_rows, img_cols):
    """
    Loads the dataset and formats the labels for binary classification if needed.
    """
    X, y = CH_load_data(img_rows, img_cols)
    if nb_classes==2:
        y[y>0] = 1  # Como es clasificación binaria, hago que todo que no sea 0, se ponga a 1
    
    return X, y



"""
############################################################################################################################
############################################################################################################################
########################################            CNN HELPER FUNCTIONS            ########################################
############################################################################################################################
############################################################################################################################
"""

# preprocessing layers applied for data augmentation
CH_data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ]
)

# Duplicates dataset using data augmentation
def CH_duplicateDataset(X_arr: np.ndarray, y_arr: np.ndarray):
    """
    Duplicates the whole dataset applying data augmentation such as rotations and zoom.
    """
    # Por cada imagen, le aplica data augmentation, lo junta todo y lo convierte en np.array
    newImages = np.array( [ CH_data_augmentation(x) for x in X_arr ] )

    return np.concatenate([X_arr, newImages], 0), np.concatenate([y_arr, y_arr], 0)


# Equilibrate minority class by generating more of it (ONLY BINARY CLASSIFICATION)
def CH_equilibrateBinaryClasses(X_arr:np. ndarray, y_arr: np.ndarray):
    """
    Equilibrates the minority class applying data augmentation.
    #### It only works on binary dataset
    """

    # calculate proportion of minority class
    classZeroCount = np.count_nonzero(np.array(y_arr) == 0)
    classOneCount  = np.count_nonzero(np.array(y_arr) == 1)
    
    proportion      = classOneCount//classZeroCount if(classZeroCount < classOneCount) else classZeroCount//classOneCount
    minorityClass   = 0                             if(classZeroCount < classOneCount) else 1

    newImagesQuantity   = classZeroCount * (proportion-1) if(classZeroCount < classOneCount) else classOneCount * (proportion-1)

    # arrays to save new generated images
    # newImages = []
    newImages = np.zeros( (newImagesQuantity, X_arr.shape[1], X_arr.shape[2], X_arr.shape[3] ))
    newLabels = np.zeros(newImagesQuantity)     if minorityClass == 0 else np.ones(newImagesQuantity)

    generatingImg = proportion-1

    i = 0
    for j in range(X_arr.shape[0]):
        # if it's minority class -> augmentate
        if y_arr[j] == minorityClass:
            # generate proportion's number of variations
            for k in range(generatingImg):
                newImages[i+k] = CH_data_augmentation(X_arr[j])
            i += generatingImg

            # newImages[i:i] = [data_augmentation(X[j]) for _ in range(generatingImg)]
            # i+= generatingImg
            # for _ in range(proportion-1):
                # newImages.append(data_augmentation(X[j]).eval(session=sess))

    # Convert list of tensors to np.array
    # newImages = tf.stack(newImages)
    # newImages = newImages.eval(session=tf.compat.v1.Session())

    # concatenate original data with new generated data    
    return np.concatenate([X_arr, newImages], 0), np.concatenate([y_arr, newLabels], 0)


def CH_applyExtraAugmentationF(X_set, imagesShape, augType = 0):
    """
    Applies extra data augmentation such as color change or gray scale.

    Input: augType
    ----------------
    0: Color Jitter
    1: Normalize
    2: Gray Scale
    """

    if augType == 0:
        print('\nColorJitter\n\n')
        
        # Define la transformación de la imagen en rangos aleatorios
        colorJitt = transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.5,1), saturation=(0.8,1.2), hue=(-0.1,0.1))

        # Operador que convierte el array en PILImage y le aplica el data augmentation
        transform = transforms.Compose([transforms.ToPILImage(), colorJitt])

        # Aplico la operacion sobre todo el array
        return np.array( [ np.asarray(transform(x)) for x in X_set.astype('uint8') ] )

    elif augType == 1:
        print('\nNormalize\n\n')

        # Datos para el preprocesamiento
        mean = (np.mean(X_set[:,:,:,0]), np.mean(X_set[:,:,:,1]), np.mean(X_set[:,:,:,2]))
        std  = (np.std(X_set[:,:,:,0]), np.std(X_set[:,:,:,1]), np.std(X_set[:,:,:,2]))

        # Operador que convierte la imagen en tensor y lo normaliza
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Aplico la operacion sobre todo el array
        return np.array( [ np.asarray(transform(x)).transpose(1, 2, 0) for x in X_set ] )

    elif augType == 2:
        print('\nGrayScale\n\n')
                
        # Operador que convierte el array en PILImage y le aplica el data augmentation
        grayScale = transforms.Grayscale()
        transform = transforms.Compose([transforms.ToPILImage(), grayScale])

        # Aplico gray scale sobre todo el dataset
        X_arr = np.array( [ np.asarray(transform(x)) for x in X_set.astype('uint8') ] )

        # Hago reshape para recuperar el 4º canal, shape = (nImg, 32, 32, 1)
        X_arr = X_arr.reshape(X_arr.shape[0], imagesShape[0], imagesShape[1], 1)

        return X_arr


def CH_preprocessDataset4Model(X_train, y_train, y_test, nb_classes, duplicateMulticlassDataset, equilibrateBinaryDataset):
    """
    Applies dataset preprocess for binary and multiclass clasification, such as duplication or equilibrating dataset and formatting the labels.
    """


    # Si estoy clasificando múltiples clases, adapto la etiqueta y
    if nb_classes > 2:
        print('______________________________\nMulticlass clasification\n______________________________\n')
        if duplicateMulticlassDataset:
            print(f'DUPLICATING DATASET {X_train.shape}')
            X_train, y_train = CH_duplicateDataset(X_train, y_train)

        y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)

        # return X_train, to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)

    else:
        print('______________________________\nBinary clasification\n______________________________\n')
        if equilibrateBinaryDataset:
            print(f'EQUILIBRATING DATASET {X_train.shape}')
            X_train, y_train = CH_equilibrateBinaryClasses(X_train, y_train)


    print(f'X_train shape after preprocess {X_train.shape}\n')

    return X_train, y_train, y_test


# Compare 2 arrays to tell the best model
def CH_wilcoxonCompareResults(modelF1, modelF2, list1, list2):
    """
    Applies Wilcoxon test to compare 2 models using their metrics list.

    Inputs
    ---
    ModelF1 & ModelF2:  the models to compare.
    list1 & list2:      the list with the results foreach model.
    """

    msg = None

    print('\n--- Wilcoxon test ---\n')
    # print('If p-value is lower than 0.05, first model is better than second')

    # First vs Second
    wilcox_W, p_value =  wilcoxon(list1, list2, alternative='greater', zero_method='wilcox', correction=False)

    print(f'--- {modelF1.__name__} vs {modelF2.__name__}')
    print(f'Wilcox W: {wilcox_W}, p-value: {p_value:.2f}')
    if p_value <= 0.05:
        msg = f'\n{modelF1.__name__} is better!\n'

    # Second vs First
    wilcox_W, p_value =  wilcoxon(list2, list1, alternative='greater', zero_method='wilcox', correction=False)

    print(f'{modelF2.__name__} vs {modelF1.__name__}')
    print(f'Wilcox W: {wilcox_W}, p-value: {p_value:.2f}')
    if p_value <= 0.05:
        msg = f'\n{modelF2.__name__} is better!\n'

    if msg is not None:
        print(msg)


def CH_plotShapValue(model, X_test, y_test, nb_classes):
    """
    Plots SHAP values for a model. Tensorflow.eager_execution must be dissabled.
    """

    class_names = ['TUMOR', 'HEALTHY'] if nb_classes ==  2 else ['TUMOR','STROMA','COMPLEX','LYMPHO','DEBRIS','MUCOSA','ADIPOSE','EMPTY']
    BACKGROUND_SIZE = 100
    background_images = X_test[:BACKGROUND_SIZE]
    
    test_images = X_test[BACKGROUND_SIZE:]
    test_targets = y_test[BACKGROUND_SIZE:]

    # Inicia el evaluador de SHAP integrando el modelo con SHAP usando el background_images
    expl = shap.DeepExplainer(model, background_images)

    # Predice las imagenes
    pred = model.predict(test_images)

    # Itera sobre las imagenes predichas
    index = np.random.randint(len(test_images), size=5)
    for i in index:
        # Obtiene la imagen reescalado al rango [0...1]
        ti = test_images[[i]]
        # Obtiene el shap_values para la imagen
        sv = expl.shap_values(ti)

        # Prepare plot for the image with the analysis
        shap.image_plot(sv, ti/255., show=False)

        # Get plotting figure
        fig = plt.gcf()
        allaxes = fig.get_axes()

        # Plot the predicted and the actual classes names
        actual_name = class_names[np.argmax(test_targets[i])]
        pred_name   = class_names[np.argmax(pred[i])]
        allaxes[0].set_title('Actual: {}, pred: {}'.format(actual_name, pred_name), fontsize=8)

        # Plot the probability for each class
        prob = pred[i]
        for x in range(1, len(allaxes)-1):
            allaxes[x].set_title('{:.2%}'.format(prob[x-1]), fontsize=14)
        plt.show()




"""
############################################################
############################################################
#################### TRAINING FUNCTIONS ####################
############################################################
############################################################
"""

printModelInfo = '   Training: {} (batch: {}, epoch: {}, split:{})'

def CH_trainModel(modelF, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape, plotHist = False):
    """
    Trains the model with the parameters passed. It returns the model & the history
    """

    print(f'\nTraining with X_train shape: {X_train.shape}')
    print(printModelInfo.format(modelF.__name__, batch, epoch, val_split))
    
    model = modelF(imagesShape, nb_classes)

    # Compilo el modelo
    if nb_classes == 2:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entreno el modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    hist = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, validation_split=val_split, verbose=0, callbacks=[early_stopping]) #, callbacks=[early_stopping])

    """ Entrenamiento con ImageDataGenerator
    # Indico el ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=val_split )

    datagen.fit(X_train)

    # Preparo los subsets
    training_subset = datagen.flow(X_train, y_train, batch_size=batch, subset='training')
    validation_subset = datagen.flow(X_train, y_train, batch_size=batch//4, subset='validation')
    
    model = modelF(imagesShape, nb_classes)

    # Compilo el modelo 
    if nb_classes == 2:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entreno el modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit( training_subset, 
        epochs = epoch,  validation_data = validation_subset, 
        verbose = 0, callbacks=[early_stopping]) 
    """

    if plotHist:
        CH_plot_learning_curves(hist)
    
    return model, hist



# Compares 3 models using AUC and F1 metrics
def CH_compareModelsWilcoxon(modelF1, modelF2, modelF3, X, y, batch, epoch, val_split, nb_classes, imagesShape, duplicateMulticlassDataset, equilibrateBinaryDataset):
    """
    Uses the Wilcoxon test to compare 3 models using CrossValidation.
    """

    print('\n   _________________________________\n__/ Compare Models with Wilcoxon ___\n________________________________/   \n')

    # Inicializo variables de estadística
    AUClist1 = []
    AUClist2 = []
    AUClist3 = []

    F1list1 = []
    F1list2 = []
    F1list3 = []

    # Instancio el stratifield KFold con 10-CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state = np.random.RandomState())

    # Imprimo modelos a comparar
    # print(printModelInfo.format(modelF1.__name__, batch, epoch, val_split))
    # print(printModelInfo.format(modelF2.__name__, batch, epoch, val_split))
    # print(printModelInfo.format(modelF3.__name__, batch, epoch, val_split))
    
    # Inicio el bucle de KFold
    for train_index, test_index in skf.split(X, y):

        # Obtengo los batches de entrenamiento y de test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        

        # Preprocesamiento
        X_train, y_train, y_test = CH_preprocessDataset4Model(X_train, y_train, y_test, nb_classes, duplicateMulticlassDataset, equilibrateBinaryDataset)

        # Genero el modelo
        model1, h = CH_trainModel(modelF1, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape)
        model2, h = CH_trainModel(modelF2, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape)
        model3, h = CH_trainModel(modelF3, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape)

        # Preparo test estadistico
        y_testLabels = None if nb_classes == 2 else y_test.argmax(axis=1)

        # Evalúo el modelo y actualizo los datos estadísticos
        for modeloPred, AUClist, F1list in zip([model1, model2, model3], [AUClist1, AUClist2, AUClist3], [F1list1, F1list2, F1list3]):
            
            y_scores = modeloPred.predict(X_test) # Confidence prediction per class

            if nb_classes > 2:
                # AUC for multiclass classification
                AUClist.append(metrics.roc_auc_score(y_test, y_scores, multi_class='ovr'))
                # F1
                y_pred = y_scores.argmax(axis=1)
                F1list.append(f1_score(y_testLabels, y_pred, average="macro"))

            else:
                #AUC for binary classification
                AUClist.append(metrics.roc_auc_score(y_test, np.round(y_scores[:,1],2)))


    # Imprimo los resultados
    print(f'AUC list para modelo {modelF1.__name__}:\t{np.round(AUClist1, 4)}')
    print(f'AUC list para modelo {modelF2.__name__}:\t{np.round(AUClist2, 4)}')
    print(f'AUC list para modelo {modelF3.__name__}:\t{np.round(AUClist3, 4)}')

    if nb_classes > 2:
        print(f'F1 list para modelo {modelF1.__name__}:\t{np.round(F1list1, 4)}')
        print(f'F1 list para modelo {modelF2.__name__}:\t{np.round(F1list2, 4)}')
        print(f'F1 list para modelo {modelF3.__name__}:\t{np.round(F1list3, 4)}')

    # Comparo los resultados 1 vs 2
    print('\nAUC\n\n')
    CH_wilcoxonCompareResults(modelF1, modelF2, AUClist1, AUClist2)

    # Comparo los resultados 3 vs 2
    CH_wilcoxonCompareResults(modelF3, modelF2, AUClist3, AUClist2)

    # Comparo los resultados 3 vs 1
    CH_wilcoxonCompareResults(modelF3, modelF1, AUClist3, AUClist1)


    if nb_classes > 2:
        # Comparo los resultados 1 vs 2
        print('\nF1\n\n')
        CH_wilcoxonCompareResults(modelF1, modelF2, F1list1, F1list2)

        # Comparo los resultados 3 vs 2
        CH_wilcoxonCompareResults(modelF3, modelF2, F1list3, F1list2)

        # Comparo los resultados 3 vs 1
        CH_wilcoxonCompareResults(modelF3, modelF1, F1list3, F1list1)


# Does a comparison between AUC and F1 metrics
def CH_viewMetrics(modelF, X, y, batch, epoch, val_split, nb_classes, imagesShape, duplicateMulticlassDataset, equilibrateBinaryDataset, plotSHAP, plotHist = False,):
    """
    Plots F1 report, AUC score and the SHAP value for a model.
    """

    print('\n   ____________________\n__/ Compare Metrics ___\n___________________/   \n')

    # Separa un conjunto de validacion y otro de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)

    # Preprocesa
    X_train, y_train, y_test = CH_preprocessDataset4Model(X_train, y_train, y_test, nb_classes, duplicateMulticlassDataset, equilibrateBinaryDataset)

   
    # print(printModelInfo.format(modelF.__name__, batch, epoch, val_split))

    # Entreno el modelo
    model, h = CH_trainModel(modelF, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape, plotHist)

    # Evalúo el modelo
    loss, acc = model.evaluate(X_test, y_test, batch_size=batch//4)

    # Examino el modelo con SHAP
    if plotSHAP:
        CH_plotShapValue(model, X_test, y_test, nb_classes)

    # Preparación para los datos estadísticos
    y_scores    =  model.predict(X_test)                                    # Confidence prediction per class
    y_pred      =  y_scores.argmax(axis=1)    # Select classes with most confidence prediction

    y_testLabels = y_test if nb_classes ==  2 else y_test.argmax(axis=1)      # Get class's label
    target_names = ['TUMOR', 'HEALTHY'] if nb_classes ==  2 else ['TUMOR','STROMA','COMPLEX','LYMPHO','DEBRIS','MUCOSA','ADIPOSE','EMPTY']

    ############################
    # Print datos estadísticos #
    ############################
    # Print acc y loss
    print(f'Acurracy: {acc} - Loss: {loss}')

    # Prints F1 report
    print(metrics.classification_report(y_testLabels, y_pred, target_names=target_names))
    
    """
    # Print F1 stadistics. Macro: all classes are are equals / Micro: mayority class has higher weight
    f1_metric = f1_score(y_testLabels, y_pred, average=None)
    print(f'F1 array: \t {f1_metric}')
    print(f'F1 macro average: \t {f1_score(y_testLabels, y_pred, average="macro")}')
    print(f'F1 micro average: \t {f1_score(y_testLabels, y_pred, average="micro")}')
    """
    
    # Obtengo métrica AUC. OVR -> one class vs the rest (All classes that are not, are negative classes) -> multiclass / OVO -> one vs one class
    AUC = metrics.roc_auc_score(y_test, np.round(y_scores[:,1],2))  if nb_classes == 2 else  metrics.roc_auc_score(y_test, y_scores, multi_class='ovr')

    if nb_classes > 2:
        for i in range(len(target_names)):
            individual_auc = metrics.roc_auc_score(np.round(y_test[:,i],2), np.round(y_scores[:,i],2), multi_class='ovr')
            print(f'{target_names[i]} class AUC: \t{individual_auc}')

    print(f'AUC score: {AUC}') 


# Executes Cross Validation on a single model
def CH_crossValidation(modelF, X, y, batch, epoch, val_split, nb_classes, imagesShape, duplicateMulticlassDataset, equilibrateBinaryDataset):
    """
    Applies CrossValidation to a model, applying data augmentation if needed. Plots the Acurracy, Loss and AUC metrics.
    """

    print('\n   _____________________\n__/ Cross Validation  __\n_____________________/  \n')

    # Inicializo variables de estadística
    msgToReturn = ''
    lossList = []
    accList = []
    AUClist = []

    # Instancio el stratifield KFold con 10-CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state = np.random.RandomState())

    # Imprimo modelo a entrenar
    # print(printModelInfo.format(modelF.__name__, batch, epoch, val_split))
    msgToReturn += printModelInfo.format(modelF.__name__, batch, epoch, val_split) + '\n'

    start_time = time.time()
    
    # Recorro el bucle de KFold
    for train_index, test_index in skf.split(X, y):

        # Obtengo los sets de entrenamiento y de test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        
        # Aplico preprocesamiento
        X_train, y_train, y_test = CH_preprocessDataset4Model(X_train, y_train, y_test, nb_classes, duplicateMulticlassDataset, equilibrateBinaryDataset)    

        # print(f'Training with {X_train.shape[0]} images')
        # print(printModelInfo.format(modelF.__name__, batch, epoch, val_split))

        # Genero el modelo
        model, h = CH_trainModel(modelF, X_train, y_train, batch, epoch, val_split, nb_classes, imagesShape)
        

        # Evalúo el modelo y actualizo los datos estadísticos
        loss, acc = model.evaluate(X_test, y_test, batch_size=batch//4)

        lossList.append(loss)
        accList.append(acc)

        y_scores = model.predict(X_test) # Confidence prediction per class

        AUC = metrics.roc_auc_score(y_test, np.round(y_scores[:,1],2))  if nb_classes == 2 else  metrics.roc_auc_score(y_test, y_scores, multi_class='ovo')


        AUClist.append(AUC) # Append AUC value
        print(f'AUC: {AUC}')

    print('###############################')
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'Promedio Loss: {np.mean(lossList):.04f}')
    print(f'Loss list:     {np.round(lossList, 4)}')
    print(f'Promedio Acc:  {np.mean(accList):.04f}')
    print(f'Acurracy list: {np.round(accList, 4)}')
    print(f'Promedio AUC:  {np.mean(AUClist):.03f}')
    print(f'AUC list:      {np.round(AUClist, 4)}')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\n')

    msgToReturn += f'AUC list:      {np.round(AUClist, 4)}\n'
    msgToReturn += f'Promedio AUC:  {np.mean(AUClist):.03f}\n\n'

    return msgToReturn


def executeHistologyColorectal(nb_classes, equilibrateBinaryDataset, duplicateMulticlassDataset, plotSHAP, applyExtraAugmentation, extraAugmentationType):
    
    print('\n\n¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?\nEXECUTING HISTOLOGY COLORECTAL\n¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?¿?\n')

    if DISABLE_EAGER_EXECUTION:
        equilibrateBinaryDataset, duplicateMulticlassDataset = False, False   # ONLY FALSE
    else:
        plotSHAP = False                    # ONLY FALSE

    # Define la cantidad de filas y columnas por imagen (ancho y largo)
    img_cols, img_rows = 32, 32

    imagesShape = (img_cols,img_rows, 1) if extraAugmentationType == 2 else (img_cols,img_rows, 3)
    
    #########################################################################
    #########################################################################
    #########################################################################
    batch_size = 32
    epochs = 50
    validation_split = 0.15
    #########################################################################
    #########################################################################
    #########################################################################

    X, y = CH_getDataset(nb_classes, img_rows, img_cols)

    if applyExtraAugmentation:
        print('EXTRA AUGMENTATION')
        X = CH_applyExtraAugmentationF(X, imagesShape, extraAugmentationType)
    
    CH_viewMetrics(cnn_model5, X, y, batch=batch_size, epoch=epochs, val_split=validation_split, 
                nb_classes=nb_classes, imagesShape=imagesShape,
                duplicateMulticlassDataset=duplicateMulticlassDataset, equilibrateBinaryDataset=equilibrateBinaryDataset, 
                plotSHAP=plotSHAP, plotHist=True)

    # CH_compareModelsWilcoxon(  cnn_model5, cnn_model6, cnn_model7, X, y, batch=batch_size, epoch=epochs, val_split=validation_split, 
    #                         nb_classes=nb_classes, imagesShape=imagesShape, 
    #                         duplicateMulticlassDataset=duplicateMulticlassDataset, equilibrateBinaryDataset=equilibrateBinaryDataset)

    # CH_crossValidation(cnn_model7, X, y, batch=batch_size, epoch=epochs, val_split=validation_split, 
    #                 nb_classes=nb_classes, imagesShape=imagesShape, 
    #                 duplicateMulticlassDataset=duplicateMulticlassDataset, equilibrateBinaryDataset=equilibrateBinaryDataset)


##################################################################################
# Main program
if __name__ == "__main__":
    executeHistologyColorectal(nb_classes=2, equilibrateBinaryDataset=False, duplicateMulticlassDataset=True, plotSHAP=True, applyExtraAugmentation=False, extraAugmentationType=0)
    executeHistologyColorectal(nb_classes=2, equilibrateBinaryDataset=True, duplicateMulticlassDataset=True, plotSHAP=True, applyExtraAugmentation=True, extraAugmentationType=1)
    executeHistologyColorectal(nb_classes=2, equilibrateBinaryDataset=True, duplicateMulticlassDataset=True, plotSHAP=True, applyExtraAugmentation=True, extraAugmentationType=2)



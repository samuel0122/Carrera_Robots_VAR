import tensorflow as tf
import numpy as np

def create_model(inputs, outputs):
    """
    BUILD KERAS MODEL USING FUNCTIONAL API
    """
    # Inputs layer
    input_layer = tf.keras.layers.Input(inputs)

    # Hidden layers
    to_output_layer = tf.keras.layers.Dense(5, activation="relu")(input_layer)

    # Output layer
    output_layer = tf.keras.layers.Dense(outputs, activation="softmax")(to_output_layer)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def mutateWeights(model_weights, mutation_rate, mutation_range):
    """
        ### Generate a new child by mutating the model's parameters
    """
    mutation = lambda weight: weight + np.random.uniform(-mutation_range, mutation_range, size=weight.shape)

    return [mutation(weight=weight) if np.random.uniform() < mutation_rate else weight for weight in model_weights]

def crossover_models_weight(model1_weights, model2_weights, crossover_rate):
    """
        ### Cross-Over boths weights.

        crossover_rate indicates how much probability there is to keep the first model's weight
    """
    # Cross-over both weights
    crossover_weight = lambda weight1, weight2: weight1 if np.random.uniform() < crossover_rate else weight2

    return [crossover_weight(weight1=weight1, weight2=weight2) for weight1, weight2 in zip(model1_weights, model2_weights)]

# Define a simple TensorFlow model with two dense layers
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model1 = create_model(5, 3)
model2 = create_model(5, 3)
model3 = create_model(5, 3)

# Compile the model with a categorical cross-entropy loss and an Adam optimizer
model1.compile(loss='categorical_crossentropy', optimizer='adam')
model2.compile(loss='categorical_crossentropy', optimizer='adam')
model3.compile(loss='categorical_crossentropy', optimizer='adam')

# Retrieve the current values of the model's trainable parameters
weights = model1.get_weights()

print('Original')
for i in model1.get_weights():
    print(i)

print('Copy')
for i in copyModel.get_weights():
    print(i)

# Define the mutation rate and range
mutation_rate = 0.1
mutation_range = 0.01

print('Mutating...')

# Generate a new child by mutating the model's parameters
for i in range(len(weights)):
    if np.random.uniform() < mutation_rate:
        weights[i] += np.random.uniform(-mutation_range, mutation_range, size=weights[i].shape)

# Set the model's parameters to the new values
model1.set_weights(weights)

print('Original')
for i in model1.get_weights():
    print(i)

print('Copy')
for i in copyModel.get_weights():
    print(i)

# Evaluate the fitness of the new child on a validation dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_test = x_test.reshape(10000, 784) / 255.0
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
# loss = model.evaluate(x_test, np.asarray(y_test))
# print('Validation loss:', loss)
# print('Validation accuracy:', accuracy)
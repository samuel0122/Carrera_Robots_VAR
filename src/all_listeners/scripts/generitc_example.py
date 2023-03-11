import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Define the problem
def fitness_function(individual):
    # Define the hyperparameters to be optimized
    learning_rate = individual[0]
    batch_size = individual[1]
    num_epochs = individual[2]
    num_hidden_layers = individual[3]
    num_neurons_per_layer = individual[4]

    # Define the neural network architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(x_test, y_test),
                        verbose=0)

    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1],  # return the accuracy as a fitness value

# Create the toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Define the hyperparameter search space
learning_rate_range = [1e-6, 1e-3]
batch_size_range = [32, 128]
num_epochs_range = [5, 20]
num_hidden_layers_range = [1, 5]
num_neurons_per_layer_range = [16, 128]
hyperparameters = [learning_rate_range, batch_size_range, num_epochs_range, num_hidden_layers_range, num_neurons_per_layer_range]

# Define the genetic operators
toolbox.register("attr_float", random.uniform)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float,)*len(hyperparameters), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the algorithm
toolbox.register("evaluate", fitness_function)
toolbox.register("algorithm", algorithms.eaSimple, toolbox=toolbox, cxpb=0.5, mutpb=0.2, ngen=10)

# Load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Run the genetic algorithm
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
pop, logbook = toolbox.algorithm(pop, halloffame=hof)

# Print the best individual
best_individual = hof[0]
print("Best individual:", best_individual)

# Plot the evolution of the fitness values
fitness_values = [logbook[i]['max'] for i in range(len(logbook))]

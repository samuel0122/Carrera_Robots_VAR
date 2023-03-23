#!/usr/bin/env python3
"""
Genetic algoritm test XOR
    @authors: Samuel Oliva Bulpitt, Luis Jesús Marhuenda Tendero
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image

from gazebo_msgs.msg import ModelState

import math
from nav_msgs.msg import Odometry
from tensorflow import keras
import numpy as np
import time
import datetime

import threading
from functools import cmp_to_key

import cv2
from cv_bridge import CvBridge

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# CONSTANTS
PRUEBAS = np.array([[0,0], [0,1], [1, 0], [1,1]])
RESULTADOS = np.array([0, 1, 1, 0])

def create_model(inputs, outputs) :
    """
    BUILD KERAS MODEL USING FUNCTIONAL API
    """
    # Inputs layer
    input_layer = keras.layers.Input(inputs)

    # Hidden layers
    x = input_layer
    x = keras.layers.Dense(10, activation="relu")(x)
    x = keras.layers.Dense(5, activation="relu")(x)

    to_output_layer = x
    # Output layer
    output_layer = keras.layers.Dense(outputs, activation="softmax")(to_output_layer)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def mutateWeights(model_weights, mutation_rate, mutation_range):
    """
        ### Generate a new child by mutating the model's parameters
    """
    mutation = lambda weight: np.asarray([w + np.random.uniform(-mutation_range, mutation_range, size=w.shape) 
                                          if np.random.uniform() < mutation_rate 
                                          else w 
                                          for w in weight]
                                          )

    return [mutation(weight=weight) for weight in model_weights]


def crossover_models_weight(model1_weights, model2_weights, crossover_rate):
    """
        ### Cross-Over boths weights.

        crossover_rate indicates how much probability there is to keep the first model's weight
    """
    # Cross-over both weights
    crossover_weight = lambda weight1, weight2: weight1 if np.random.uniform() < crossover_rate else weight2

    return [crossover_weight(weight1=weight1, weight2=weight2) for weight1, weight2 in zip(model1_weights, model2_weights)]


class XOR:
    resultados = []
    distanciaEsperados = 0
    def __init__(self, loadModel = None):
        
        self.modelAI = create_model(inputs=2, outputs=2)

    def setWeights(self, weights):
        self.modelAI.set_weights(weights)

    def getWeights(self):
        return self.modelAI.get_weights()

    def executeXOR(self):
        # self.resultados = np.argmax(self.modelAI.predict(PRUEBAS), axis=1)
        self.resultados = np.asarray(self.modelAI.predict(PRUEBAS, verbose = 0) [:, 1])
        self.distanciaEsperados = sum(abs(self.resultados - RESULTADOS))


def compare_childs(XOR1: XOR, XOR2: XOR):

    return -1 if XOR1.distanciaEsperados < XOR2.distanciaEsperados else 1


class Population:
    MUTATION_RANGE = 0.1
    MUTATION_RATE  = 0.3
    CROSSOVER_RATE = 0.5
    GenVersion = 0
    bestDistance = 4
    
    def __init__(self, sizePopulation = 9):
        self.generation = [XOR() for _ in range(sizePopulation)]
        self.sizePopulation = sizePopulation
        self.GenVersion = 0

    def simulateGeneration(self):
        print(f'{bcolors.OKGREEN}Executin {self.sizePopulation} XOR\'s...{bcolors.ENDC}')
        for i in range(self.sizePopulation):
            self.generation[i].executeXOR()
    
    def nextGen(self):
        self.GenVersion += 1

        print(f'{bcolors.OKCYAN}Creating Gen{self.GenVersion} of XOR\'s!{bcolors.ENDC}')
        
        # Sort the robots
        self.generation.sort(key=cmp_to_key(compare_childs))
        
        print([x.distanciaEsperados for x in self.generation])
        self.bestDistance = self.generation[0].distanciaEsperados

        # Keep top 3
        xor1W = self.generation[0].getWeights()
        xor2W = self.generation[1].getWeights()
        xor3W = self.generation[2].getWeights()

        # Mutate top 3
        xor4W = mutateWeights(model_weights=xor1W, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)
        xor5W = mutateWeights(model_weights=xor2W, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)
        xor6W = mutateWeights(model_weights=xor3W, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)

        # Crossover top 3 in to 3 new ones
        xor7W = crossover_models_weight(model1_weights=xor1W, model2_weights=xor2W, crossover_rate=self.CROSSOVER_RATE)
        xor8W = crossover_models_weight(model1_weights=xor1W, model2_weights=xor3W, crossover_rate=self.CROSSOVER_RATE)
        xor9W = crossover_models_weight(model1_weights=xor2W, model2_weights=xor3W, crossover_rate=self.CROSSOVER_RATE)
        
        # Update weights
        self.generation[3].setWeights(xor4W)
        self.generation[4].setWeights(xor5W)
        self.generation[5].setWeights(xor6W)
        self.generation[6].setWeights(xor7W)
        self.generation[7].setWeights(xor8W)
        self.generation[8].setWeights(xor9W)

    def printBest(self):
        self.generation.sort(key=cmp_to_key(compare_childs))
        print(self.generation[0].resultados)

if __name__ == '__main__':

    pop = Population()
    while pop.bestDistance > 0.2:
        pop.simulateGeneration()
        pop.nextGen()
    

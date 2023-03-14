#!/usr/bin/env python3
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
CHECK_EVERY_SECONDS = 20
MINIMUM_DISTANCE_DIFFERENCE = 1.0
LATER_WALL_CRASH_PATIENTE = 15
LASER_MAX_DISTANCE = 5.0
LASER_MIN_DISTANCE = 0.2
INITIAL_ROBOT_X, INITIAL_ROBOT_Y, INITIAL_ROBOT_Z = -6.5, 8.5, 0.2
INITIAL_ROBOT_X, INITIAL_ROBOT_Y, INITIAL_ROBOT_Z = 0, -9, 0.2
STATES_SAVE_DIRECTORY = '/home/samuel/Carrera_Robots_VAR/src/all_listeners/states/'
STATESLIST_SAVE_DIRECTORY = '/home/samuel/Carrera_Robots_VAR/src/all_listeners/statesLists/'

INITIAL_ROBOT_X = 2

SAVE_STATE_EVERY_GENERATIONS = 15

def create_model(inputs, outputs):
    """
    BUILD KERAS MODEL USING FUNCTIONAL API
    """
    # Inputs layer
    input_layer = keras.layers.Input(inputs)

    # Hidden layers
    dense1 = keras.layers.Dense(10, activation="relu")(input_layer)
    to_output_layer = keras.layers.Dense(5, activation="relu")(dense1)

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


class Wander:
    def __init__(self, loadModel = None):
        
        # Variables to know if the robot is alive
        self.robotCrashedEvent = None
        self.timeStarted = None
        self.timeCrashed = None
        
        self.checkPoint = 0

        # Checks of loops
        self.lastTimeChecked = None
        self.last_x_checked, self.last_y_checked = None, None

        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = None
        self.laser_sub = None
        
        # Odometry: this is prepared for the fitness
        self.odom_sub = None
        self.xMax, self.yMax = INITIAL_ROBOT_X, INITIAL_ROBOT_Y
        self.xMin, self.yMin = INITIAL_ROBOT_X, INITIAL_ROBOT_Y

        # Movement variables
        self.forward_vel, self.rotate_vel = 0, 0
        
        self.previousScannnerData = None
        self.scannerData = None

        # Controlled by AI
        if loadModel is not None:
            print(f'Loaded model {loadModel}')
        self.modelAI = keras.models.load_model(loadModel) if loadModel is not None else create_model(inputs=9, outputs=3)

    """
        Robot data management
    """
    def resetData(self):
        # Variables to know if the robot is alive
        self.robotCrashedEvent = threading.Event()
        
        self.checkPoint = 5
        
        # Checks of loops
        self.last_x_checked, self.last_y_checked = INITIAL_ROBOT_X, INITIAL_ROBOT_Y

        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = None
        self.laser_sub = None
        
        # Odometry: this is prepared for the fitness
        self.odom_sub = None
        self.xMax, self.yMax = INITIAL_ROBOT_X, INITIAL_ROBOT_Y
        self.xMin, self.yMin = INITIAL_ROBOT_X, INITIAL_ROBOT_Y

        # Movement variables
        self.forward_vel, self.rotate_vel = 0, 0
        
        self.previousScannnerData = None
        self.scannerData = None
    
    def restartTimers(self):
        self.timeStarted = time.time()
        self.lastTimeChecked = self.timeStarted + 5

    def spawnToInitialPosition(self):
        """
            Sends a message to spawn the object
        """
        print('Spawning TurtleBot...')

        # Create publisher and message to move the model
        spawnPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        spawnMsg = ModelState()

        # Robot model name
        spawnMsg.model_name = 'turtlebot3'

        # Spawn coordinates
        spawnMsg.pose.position.x, spawnMsg.pose.position.y, spawnMsg.pose.position.z = INITIAL_ROBOT_X, INITIAL_ROBOT_Y, INITIAL_ROBOT_Z

        # Spawn orientation
        spawnMsg.pose.orientation.x, spawnMsg.pose.orientation.y, spawnMsg.pose.orientation.z = 0, 0, 180

        # Send the message and close the publisher
        time.sleep(0.5)
        spawnPub.publish(spawnMsg)
        time.sleep(0.5)
        spawnPub.unregister()

        # print('TurtleBot spawned!')
    
    def getAreaRun(self):
        xMoved = abs(self.xMax - self.xMin)
        yMoved = abs(self.yMax - self.yMin)
        return xMoved * yMoved

    def getMovedDistances(self):
        return abs(self.xMax - self.xMin), abs(self.yMax - self.yMin)

    def getTimeAlive(self):
        return self.timeCrashed - self.timeStarted

    def getRobotWeights(self):
        return self.modelAI.get_weights()
    
    def setRobotWeights(self, weights):
        self.modelAI.set_weights(weights)

    """
        Robot movement simuation
    """
    def getAIKey(self):
        """
            Asks the models to choose a key
        """

        if self.modelAI is None:
            rospy.logerr('No model loaded!')
            return
        
        # TODO: define sensor to pick
        predict = self.modelAI.predict([self.scannerData], verbose=0)
        move = np.argmax(predict, axis=1)[0]
        if   move == 0:   # Left
            self.forward_vel, self.rotate_vel = 0.3, 0.5
        elif move == 2:   # Right
            self.forward_vel, self.rotate_vel = 0.3, -0.5
        elif move == 1:   # Forward
            self.forward_vel, self.rotate_vel = 0.5, 0

    def isColliding(self):
        if self.scannerData is None or self.previousScannnerData is None:
            return False 
        return (    # Checks if any of the 3 frontal scanners are lower than threshold and that the data isn't disturbed by noise
                   self.scannerData[0] < LASER_MIN_DISTANCE and abs(self.scannerData[0] - self.previousScannnerData[0]) < 0.05 
                or self.scannerData[1] < LASER_MIN_DISTANCE and abs(self.scannerData[1] - self.previousScannnerData[1]) < 0.05
                or self.scannerData[2] < LASER_MIN_DISTANCE and abs(self.scannerData[2] - self.previousScannnerData[2]) < 0.05
                #or self.scannerData[3] < LASER_MIN_DISTANCE and abs(self.scannerData[3] - self.previousScannnerData[3]) < 0.05
                #or self.scannerData[4] < LASER_MIN_DISTANCE and abs(self.scannerData[4] - self.previousScannnerData[4]) < 0.05
                )

    def getScanValues(self, msg: LaserScan):
        """
            Get proximity sensor's lvels
        """

        self.previousScannnerData = self.scannerData

        # Get 9 points (forward, forward-left, forward-right, backward-left, backward-right)
        self.scannerData = [msg.ranges[0], msg.ranges[30], msg.ranges[-30], msg.ranges[60], msg.ranges[-60], msg.ranges[90], msg.ranges[-90], msg.ranges[120], msg.ranges[-120]]
        self.scannerData = [min(num, LASER_MAX_DISTANCE) for num in self.scannerData]

    def command_callback(self, msg: LaserScan):
        """
            Reads the scanner and sends the robot's speed
        """

        if self.isColliding():
            rospy.logerr('The robot has collided!')
            self.robotCrashedEvent.set()
            return

        # Get the scan values
        self.getScanValues(msg)

        # Get the key pressed or ask the AI to choose a key
        self.getAIKey()

        # Writte the message
        msg = Twist()
        msg.linear.x = self.forward_vel
        msg.angular.z = self.rotate_vel

        # Publish the message
        if not self.robotCrashedEvent.is_set():
            self.command_pub.publish(msg)


    """
        Robot odometry callback
    """
    def getCheckPoints(self):
        return self.checkPoint
    
    def checkPoints(self, newX, newY):
        if self.checkPoint < 1:
            if newY < 7 and newX < 9 and newX > 10:
                self.checkPoint = -1
            else:
                self.checkPoint = 0
            
        if self.checkPoint < 1:   # Checkpoint 1
            if newX > 5 and newX < 10 and newY < 5.95:
                self.checkPoint = 1
        elif self.checkPoint < 2: # Checkpoint 2
            if newX > 5 and newX < 10 and newY < 3.95:
                self.checkPoint = 2
        elif self.checkPoint < 3: # Checkpoint 3
            if newX > 5 and newX < 10 and newY < 1.95:
                self.checkPoint = 3
        elif self.checkPoint < 4: # Checkpoint 4
            if newX > 8 and newX < 10 and newY < -3:
                self.checkPoint = 4
        elif self.checkPoint < 5: # Checkpoint 5
            if newX < 3 and newY > -10 and newY < -8:
                self.checkPoint = 5
        elif self.checkPoint < 6: # Checkpoint 6
            if newX < -7 and newY > -10 and newY < -8:
                self.checkPoint = 6
        elif self.checkPoint < 7: # Checkpoint 7
            if newY > -7 and newX > -10 and newX < -8:
                self.checkPoint = 7
        elif self.checkPoint < 8: # Checkpoint 8
            if newX > -6 and newY > -3 and newY < -1:
                self.checkPoint = 8
        elif self.checkPoint < 9: # Checkpoint 9
            if newX > -2 and newY > -5 and newY < -3:
                self.checkPoint = 9
        elif self.checkPoint < 10: # Checkpoint 10
            if newY > 0 and newX > 1 and newX < 3:
                self.checkPoint = 10
        elif self.checkPoint < 11: # Checkpoint 11
            if newX < 0.5 and newY > 1 and newY < 3:
                self.checkPoint = 11
        elif self.checkPoint < 12:  # Checkpoint 12
            if newX < -7 and newY > 1 and newY < 3:
                self.checkPoint = 12
        elif self.checkPoint < 13: # Checkpoint 13
            if newY > 4 and newX > -10 and newX < -8:
                self.checkPoint = 13
        elif self.checkPoint < 14: # Checkpoint 14
            if newX > -8 and newY > 7 and newY < 10:
                self.checkPoint = 14
        
    def odom_callback(self, msg: Odometry):
        
        # Keep odometry data     
        newX = msg.pose.pose.position.x
        newY = msg.pose.pose.position.y
        
        self.checkPoints(newX=newX, newY=newY)
        
        if time.time() > self.lastTimeChecked + CHECK_EVERY_SECONDS:
            #Its time to check again
            if abs(self.last_x_checked - newX) < MINIMUM_DISTANCE_DIFFERENCE and abs(self.last_y_checked - newY) < MINIMUM_DISTANCE_DIFFERENCE:
                rospy.logerr('The robot has staid in a loop')
                self.robotCrashedEvent.set()
                return
            elif abs(INITIAL_ROBOT_X - newX) < MINIMUM_DISTANCE_DIFFERENCE and abs(INITIAL_ROBOT_Y - newY) < MINIMUM_DISTANCE_DIFFERENCE:
                rospy.logerr('The robot has returned to it\'s initial position')
                self.robotCrashedEvent.set()
                return
            
            self.last_x_checked, self.last_y_checked = newX, newY
            self.lastTimeChecked = time.time()

        # Update the maximum coordinates achieved
        if newX < self.xMin:
            self.xMin = newX
        if newX > self.xMax:
            self.xMax = newX

        if newY < self.yMin:
            self.yMin = newY
        if newY > self.yMax:
            self.yMax = newY

    """
        Robot simulate call
    """
    def simulateRobot(self):
        """
            Make the robot run in the circuit until it crashes
        """
        # Restart the data
        self.resetData()

        # Do preps for starting the robot
        self.spawnToInitialPosition()

        # Restart time started
        self.restartTimers()
        
        # Create the publisher and subscribers
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # The robot is controlled by AI, wait until the crashed event is actived
        self.robotCrashedEvent.wait()

        # Set the time crashed
        self.timeCrashed = time.time()

        # Do preps when crashed
        self.command_pub.unregister()
        self.laser_sub.unregister()
        self.odom_sub.unregister()

    def saveModel(self, model_name = None):
        if model_name is None:
            model_name = f'modelAI_{int(self.getTimeAlive())}_{int(self.getAreaRun())}.h5'
        
        fileSave = STATES_SAVE_DIRECTORY + model_name
        self.modelAI.save(fileSave)

        return fileSave


def compare_robots(robot1: Wander, robot2: Wander):

    checkPoint1 = robot1.getCheckPoints()
    checkPoint2 = robot2.getCheckPoints()
    
    if checkPoint1 != checkPoint2:
        return -1 if checkPoint1 > checkPoint2 else 1
    
    DISTANCE_DIFFERENCE = 1

    robot1X, robot1Y = robot1.getMovedDistances()
    robot2X, robot2Y = robot2.getMovedDistances()

    if abs(robot1X - robot2X) < DISTANCE_DIFFERENCE:
        # Same X distance
        if abs(robot1Y - robot2Y) < DISTANCE_DIFFERENCE:
            # If they both runned the same distance
            robot1Time = robot1.getTimeAlive()
            robot2Time = robot2.getTimeAlive()

            # if robot1Time < 10 and robot2Time < 10:
            #     # If both times are low (at beggining) put first the lowest
            #     return -1 if robot1Time < robot2Time else 1
            # else:
            # If time are long, put first the one who survived more time
            return -1 if robot1Time > robot2Time else 1
        else:
            # Same X distance, different Y distance
            # Put first the robot who runned more Y distance
            return -1 if robot1Y > robot2Y else 1
    else:
        # Different X distances
        if abs(robot1Y - robot2Y) > DISTANCE_DIFFERENCE:
            # If different X and different Y, look who got further X-wise and Y-wise
            return -1 if (robot1X - robot2X) + (robot1Y - robot2Y) > 0 else 1
        
        # If same y, put first the robot who runned more X distance
        return -1 if robot1X > robot2X else 1


class Population:
    MUTATION_RANGE = 0.02
    MUTATION_RATE  = 0.3
    CROSSOVER_RATE = 0.5
    GenVersion = 0
    
    def __init__(self, sizePopulation = 9):
        self.generation = [Wander() for _ in range(sizePopulation)]
        self.sizePopulation = sizePopulation
        self.GenVersion = 0
    
    def saveBestRobots(self, top_robots = 3):

        # Check if the requested top if higher than avaliable
        top_robots = top_robots if top_robots <= len(self.generation) else len(self.generation)

        print(f'{bcolors.OKBLUE}Saving top {top_robots} robots of Gen{self.GenVersion}!{bcolors.ENDC}')

        # Sort the robots
        self.generation.sort(key=cmp_to_key(compare_robots))

        # Get date
        currentTime = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        # Create folder to save if it doesn't exist
        folder = f'tops{currentTime}'
        if not os.path.exists(STATES_SAVE_DIRECTORY + folder):
            os.mkdir(STATES_SAVE_DIRECTORY + folder)

        # Save models
        for i in range(top_robots):
            self.generation[i].saveModel(f'{folder}/model_top{i}_{currentTime}')

    def saveState(self, fileList):

        print(f'{bcolors.OKBLUE}Saving state of Gen{self.GenVersion}!{bcolors.ENDC}')

        # Get date
        currentTime = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        # Create folder to save if it doesn't exist
        folder = f'state{currentTime}'
        if not os.path.exists(STATES_SAVE_DIRECTORY + folder):
            os.mkdir(STATES_SAVE_DIRECTORY + folder)
            
        # Create folder to save state list
        if not os.path.exists(STATESLIST_SAVE_DIRECTORY):
            os.mkdir(STATESLIST_SAVE_DIRECTORY)
            
        # Save models
        filesSaved = [str(self.GenVersion)]   # First number Gen
        for i in range(self.sizePopulation):   # Then append all models files
            filesSaved.append(self.generation[i].saveModel(f'{folder}/model{i}_state{currentTime}'))
        
        # Write all files
        with open(STATESLIST_SAVE_DIRECTORY + fileList, 'w') as f:
            f.write('\n'.join(filesSaved))
    
    def loadState(self, listFile):
        modelsFiles = []
        print(f'{bcolors.OKBLUE}Loading state {listFile}{bcolors.ENDC}')

        with open(STATESLIST_SAVE_DIRECTORY + listFile, 'r') as f:
            modelsFiles = f.read().split('\n')
            self.GenVersion = int(modelsFiles[0])    # Get gen number from begining
            del modelsFiles[0]  # Remove gen number from list
        
        self.generation = [Wander(loadModel=modelFile) for modelFile in modelsFiles]

    def simulateGeneration(self):
        for i in range(self.sizePopulation):
            print(f'{bcolors.OKGREEN}Simulating robot {i+1} of {self.sizePopulation}...{bcolors.ENDC}')
            self.generation[i].simulateRobot()
    
    def nextGen(self):
        self.GenVersion += 1

        print(f'{bcolors.OKCYAN}Creating Gen{self.GenVersion} of robots!{bcolors.ENDC}')
        
        # Sort the robots
        self.generation.sort(key=cmp_to_key(compare_robots))
        
        print([r.getTimeAlive() for  r in self.generation])

        # Keep top 3
        robot1Weights = self.generation[0].getRobotWeights()
        robot2Weights = self.generation[1].getRobotWeights()
        robot3Weights = self.generation[2].getRobotWeights()

        # Mutate top 3
        robot4Weights = mutateWeights(model_weights=robot1Weights, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)
        robot5Weights = mutateWeights(model_weights=robot2Weights, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)
        robot6Weights = mutateWeights(model_weights=robot3Weights, mutation_range=self.MUTATION_RANGE, mutation_rate=self.MUTATION_RATE)

        # Crossover top 3 in to 3 new ones
        robot7Weights = crossover_models_weight(model1_weights=robot1Weights, model2_weights=robot2Weights, crossover_rate=self.CROSSOVER_RATE)
        robot8Weights = crossover_models_weight(model1_weights=robot1Weights, model2_weights=robot3Weights, crossover_rate=self.CROSSOVER_RATE)
        robot9Weights = crossover_models_weight(model1_weights=robot2Weights, model2_weights=robot3Weights, crossover_rate=self.CROSSOVER_RATE)
        
        # Update weights
        self.generation[3].setRobotWeights(robot4Weights)
        self.generation[4].setRobotWeights(robot5Weights)
        self.generation[5].setRobotWeights(robot6Weights)
        self.generation[6].setRobotWeights(robot7Weights)
        self.generation[7].setRobotWeights(robot8Weights)
        self.generation[8].setRobotWeights(robot9Weights)



if __name__ == '__main__':
    rospy.init_node('AI_robot_controlled')

    pop = Population()
    pop.loadState('trainingDefault.txt')
    
    for _ in range(5):
        
        for _ in range(SAVE_STATE_EVERY_GENERATIONS):
            pop.simulateGeneration()
            pop.nextGen()
            
        pop.saveState(f'Gen{pop.GenVersion}.txt')

    pop.saveState(f'trainingDefault.txt')
        
        
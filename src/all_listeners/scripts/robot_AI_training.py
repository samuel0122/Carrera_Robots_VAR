#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image

from gazebo_msgs.msg import ModelState

import math
from nav_msgs.msg import Odometry
from tensorflow import keras
import numpy as np
import time
import threading

import cv2
from cv_bridge import CvBridge


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


class Wander:
    def __init__(self):
        
        # Variables to know if the robot is alive
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2
        self.isAlive = True
        self.robotCrashedEvent = threading.Event()

        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = None#rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = None#rospy.Subscriber('/scan', LaserScan, self.command_callback)
        
        # Odometry: this is prepared for the fitness
        self.odom_sub = None#rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.initialX, self.initialY, self.initialZ = -6.5, 8.5, 0.2
        self.xMax, self.yMax = self.initialX, self.initialY
        self.xMin, self.yMin = self.initialX, self.initialY

        # Movement variables
        self.forward_vel, self.rotate_vel = 0, 0
        self.categoricalOutput = None
        
        self.previousScannnerData = None
        self.scannerData = None

        # Controlled by AI
        self.modelAI = keras.models.load_model('./best_model.h5')


    def spawnToInitialPosition(self):
        """
            Sends a message to spawn the object
        """
        # Create publisher and message to move the model
        spawnPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        spawnMsg = ModelState()

        # Robot model name
        spawnMsg.model_name = 'turtlebot3'

        # Spawn coordinates
        spawnMsg.pose.position.x, spawnMsg.pose.position.y, spawnMsg.pose.position.x = self.initialX, self.initialY, self.initialZ

        # Spawn orientation
        spawnMsg.pose.orientation.x, spawnMsg.pose.orientation.y, spawnMsg.pose.orientation.z = 0, 0, 0

        # Send the message and close the publisher
        spawnPub.publish(spawnMsg)
        spawnPub.publish(spawnMsg)
        spawnPub.publish(spawnMsg)
        spawnPub.publish(spawnMsg)
        spawnPub.publish(spawnMsg)
        spawnPub.publish(spawnMsg)
        # spawnPub.unregister()

        print('SPAWNED')


    def getAIKey(self):
        """
            Asks the models to choose a key
        """

        if self.modelAI is None:
            return
        
        # TODO: define sensor to pick
        predict = self.modelAI.predict([self.scannerData])
        move = np.argmax(predict, axis=1)[0]
        if move == 0:
            # Left
            self.forward_vel, self.rotate_vel = 0.2, 0.5
            self.categoricalOutput = [1.0, 0.0, 0.0]
        if move == 2:
            # Right
            self.forward_vel, self.rotate_vel = 0.2, -0.5
            self.categoricalOutput = [0.0, 0.0, 1.0]
        if move == 1:
            # Forward
            self.forward_vel, self.rotate_vel = 0.5, 0
            self.categoricalOutput = [0.0, 1.0, 0.0]


    def getAreaRun(self):
        xMoved = self.xMax - self.xMax
        yMoved = self.yMax - self.yMax
        return xMoved * yMoved


    def isColliding(self):
        if self.scannerData is None or self.previousScannnerData is None:
            return False 
        return (    # Checks if any of the 3 frontal scanners are lower than threshold and that the data isn't disturbed by noise
                   self.scannerData[0] < self.laserMinDistance and abs(self.scannerData[0] - self.previousScannnerData[0]) < 0.05 
                or self.scannerData[1] < self.laserMinDistance and abs(self.scannerData[1] - self.previousScannnerData[1]) < 0.05
                or self.scannerData[2] < self.laserMinDistance and abs(self.scannerData[2] - self.previousScannnerData[2]) < 0.05
                )


    def getScanValues(self, msg: LaserScan):
        """
            Get proximity sensor's lvels
        """

        self.previousScannnerData = self.scannerData

        # Get 5 points (forward, forward-left, forward-right, backward-left, backward-right)
        self.scannerData = [msg.ranges[0], msg.ranges[60], msg.ranges[-60], msg.ranges[120], msg.ranges[-120]]
        self.scannerData = [num if num < 5 else 5.0 for num in self.scannerData]


    def command_callback(self, msg: LaserScan):
        """
            Reads the scanner and sends the robot's speed
        """

        if self.isColliding():
            rospy.logerr("The robot has collided!")
            self.robotCrashedEvent.set()
            self.isAlive = False
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
        self.command_pub.publish(msg)


    def odom_callback(self, msg: Odometry):

        # Keep odometry data     
        newX = msg.pose.pose.position.x
        newY = msg.pose.pose.position.y
        
        if newX < self.xMin:
            self.xMin = newX
        if newX > self.xMax:
            self.xMax = newX

        if newY < self.yMin:
            self.yMin = newY
        if newY > self.yMax:
            self.yMax = newY


    def simulateRobot(self):
        """
            Make the robot run in the circuit until it crashes
        """
        # Do preps for starting the robot
        self.spawnToInitialPosition()

        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # The robot is controlled by AI, wait until the crashed event is actived
        self.robotCrashedEvent.wait()

        # Do preps when crashed
        self.command_pub.unregister()
        self.laser_sub.unregister()
        self.odom_sub.unregister()

if __name__ == '__main__':
    rospy.init_node('AI_robot_controlled')

    # TODO: insert the AI
    wand = Wander()
    wand.simulateRobot()
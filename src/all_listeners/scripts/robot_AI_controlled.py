#!/usr/bin/env python3
"""
Robot controlled by AI
    @authors: Samuel Oliva Bulpitt, Luis Jesús Marhuenda Tendero
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import math
from nav_msgs.msg import Odometry
from tensorflow import keras
import numpy as np
import time
from gazebo_msgs.msg import ModelState
import threading

import cv2
from cv_bridge import CvBridge

# CONSTANTS
LASER_MAX_DISTANCE = 5.0
LASER_MIN_DISTANCE = 0.2
INITIAL_ROBOT_X, INITIAL_ROBOT_Y, INITIAL_ROBOT_Z = -6.5, 8.5, 0.2
INITIAL_ROTATION_X, INITIAL_ROTATION_Y, INITIAL_ROTATION_Z = 0, 0, 0

class Wander:
    def __init__(self, loadModel):
        
        # Variables to know if the robot is alive
        self.robotCrashedEvent = threading.Event()
        self.timeStarted = None
        self.timeCrashed = None
        
        self.checkPoint = 0
        self.lapsCompleted = 0

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
        print(f'Loaded model {loadModel}')
        self.modelAI = keras.models.load_model(loadModel)

    """
        Robot data management
    """
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
        spawnMsg.pose.orientation.x, spawnMsg.pose.orientation.y, spawnMsg.pose.orientation.z = INITIAL_ROTATION_X, INITIAL_ROTATION_Y, INITIAL_ROTATION_Z

        # Send the message and close the publisher
        time.sleep(0.5)
        spawnPub.publish(spawnMsg)
        time.sleep(0.5)
        spawnPub.unregister()

    
    def getAreaRun(self):
        xMoved = abs(self.xMax - self.xMin)
        yMoved = abs(self.yMax - self.yMin)
        return xMoved * yMoved

    def getMovedDistances(self):
        return abs(self.xMax - self.xMin), abs(self.yMax - self.yMin)

    def getTimeAlive(self):
        if self.timeCrashed is None or self.timeStarted is None:
            return 0
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
        
        # Feedforward the model to get the key to press
        predict = self.modelAI.predict([self.scannerData], verbose=0)
        move = np.argmax(predict, axis=1)[0]
        if   move == 0:   # Left
            self.forward_vel, self.rotate_vel = 0.5, 0.75
        elif move == 2:   # Right
            self.forward_vel, self.rotate_vel = 0.5, -0.75
        elif move == 1:   # Forward
            self.forward_vel, self.rotate_vel = 0.6, 0

    def isColliding(self):
        if self.scannerData is None or self.previousScannnerData is None:
            return False 
        return (    # Checks if any of the 3 frontal scanners are lower than threshold and that the data isn't disturbed by noise
                   self.scannerData[0] < LASER_MIN_DISTANCE and abs(self.scannerData[0] - self.previousScannnerData[0]) < 0.05 
                or self.scannerData[1] < LASER_MIN_DISTANCE and abs(self.scannerData[1] - self.previousScannnerData[1]) < 0.05
                or self.scannerData[2] < LASER_MIN_DISTANCE and abs(self.scannerData[2] - self.previousScannnerData[2]) < 0.05
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
            # If robot crashed, set the crashed event to stop
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
    

    def odom_callback(self, msg: Odometry):
        
        # Keep odometry data     
        newX = msg.pose.pose.position.x
        newY = msg.pose.pose.position.y
        

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

        # Do preps for starting the robot
        self.spawnToInitialPosition()
        
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



if __name__ == '__main__':
    rospy.init_node('AI_robot_controlled')

    wand = Wander('/home/samuel/Carrera_Robots_VAR/src/all_listeners/3LapsModels/modelAI_17-03-2023_21:12:41.h5')
    wand.simulateRobot()
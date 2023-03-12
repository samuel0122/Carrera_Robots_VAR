#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import math
from nav_msgs.msg import Odometry
from tensorflow import keras
import numpy as np
import time

import cv2
from cv_bridge import CvBridge


class Wander:
    def __init__(self, controlledByOneAI = False, pathSavedModelLeft = None, pathSavedModelRight = None, pathSavedModelForward = None):
        
        # Variables to know if the robot is alive
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2
        self.isAlive = True

        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        
        # Odometry: this is prepared for the fitness
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.xMax, self.yMax = -math.inf, -math.inf
        self.xMin, self.yMin =  math.inf,  math.inf
        self.initialX, self.initialY, self.initialZ = -6.5, 8.5, 0.2

        # Movement variables
        self.forward_vel, self.rotate_vel = 0, 0
        self.categoricalOutput = None
        
        self.scannnerData5Points = None
        self.scannnerDataEvery5 = None
        self.scannnerDataEvery2 = None

        # Controlled by AI
        self.controlledByOneAI = controlledByOneAI
        self.modelAI = None
        self.modelLeft = None
        self.modelRight = None
        self.modelForward = None

        # Get one AI
        if self.controlledByOneAI:
            self.modelAI = keras.models.load_model('/home/samuel/P1_Carrera_de_robots/best_model.h5')

        # Get the paths of the saved models (if any passed)
        else:
            self.modelLeft = keras.models.load_model(pathSavedModelLeft)
            self.modelRight = keras.models.load_model(pathSavedModelRight)
            self.modelForward = keras.models.load_model(pathSavedModelForward)


    def getAIKey(self):
        """
            Asks the models to choose a key
        """

        if self.controlledByOneAI:
            if self.modelAI is None:
                return
            
            # TODO: define sensor to pick
            predict = self.modelAI.predict([self.scannnerData5Points])
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
        else:
            if self.modelLeft is None or self.modelRight is None or self.modelForward is None:
                return
            
            # TODO: define sensor to pick
            predictLeft = self.modelLeft.predict([self.scannnerData5Points])
            predictRight = self.modelRight.predict([self.scannnerData5Points])
            predictForward = self.modelForward.predict([self.scannnerData5Points])

            if np.argmax(predictLeft)[0] == 1:
                # Left
                self.forward_vel, self.rotate_vel = 0.2, 0.5
                self.categoricalOutput = [1.0, 0.0, 0.0]

            elif np.argmax(predictRight)[0] == 1:
                # Right
                self.forward_vel, self.rotate_vel = 0.2, -0.5
                self.categoricalOutput = [0.0, 0.0, 1.0]

            elif np.argmax(predictForward)[0] == 1:
                # Forward
                self.forward_vel, self.rotate_vel = 0.5, 0
                self.categoricalOutput = [0.0, 1.0, 0.0]


    def checkNumberOfScansColliding(self, msg: LaserScan):
        return sum(self.laserMinDistance > range for range in msg.range[:30]) + sum(self.laserMinDistance > range for range in msg.range[-30:])


    def getScanValues(self, msg: LaserScan):
        """
            Get proximity sensor's lvels
        """

        # Get every 5º
        self.scannnerDataEvery5 = list(msg.ranges)
        del self.scannnerDataEvery5[150:210]
        self.scannnerDataEvery5 = [num if num < 5 else 5.0 for num in self.scannnerDataEvery5[0::5]]

        # Get every 2º
        self.scannnerDataEvery2 = list(msg.ranges)
        del self.scannnerDataEvery2[150:210]
        self.scannnerDataEvery2 = [num if num < 5 else 5.0 for num in self.scannnerDataEvery2[0::2]]

        # Get 5 points (forward, forward-left, forward-right, backward-left, backward-right)
        self.scannnerData5Points = [msg.ranges[0], msg.ranges[60], msg.ranges[-60], msg.ranges[120], msg.ranges[-120]]
        self.scannnerData5Points = [num if num < 5 else 5.0 for num in self.scannnerData5Points]


    def command_callback(self, msg: LaserScan):
        """
            Reads the scanner and sends the robot's speed
        """

        if self.checkNumberOfScansColliding(msg) > self.laserWallCrashPatiente:
            rospy.logerr("The robot has collided!")
            self.isAlive = False

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


    def loop(self):
        """
            Loop where the player
        """

        # The robot is controlled by AI, stop this Thread's execution and just execute the callbacks
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('AI_robot_controlled')

    # TODO: insert the AI
    wand = Wander(controlledByOneAI = True)
    wand.loop()
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
    def __init__(self, writeSensorInformation = True, overridePreviousWriteFile = True, controlledByAI = False, pathSavedModelLeft = None, pathSavedModelRight = None, pathSavedModelForward = None):
        
        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2
        
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        displayVideo = False

        if displayVideo:
            self.initializeRGB()
            self.video_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageRGB_callback)

        

        # Default forward
        self.forward_vel = None
        self.rotate_vel = None
        self.categoricalOutput = None
        
        self.odometryX = None
        self.odometryY = None
        self.scannnerData = None
        self.scannnerDataEvery5 = None
        self.scannnerDataEvery2 = None

        self.keyInserted = 'w'

        # Get the paths of the saved models (if any passed)
        self.controlledByAI = controlledByAI
        if self.controlledByAI:
            self.modelLeft = keras.models.load_model(pathSavedModelLeft)
            self.modelRight = keras.models.load_model(pathSavedModelRight)
            self.modelForward = keras.models.load_model(pathSavedModelForward)

        # If set to save information
        self.saveInformation = writeSensorInformation
        if self.saveInformation:
            self.fileName = "/home/samuel/P1_Carrera_de_robots/datos5_.csv"
            self.fileNameEvery2 = "/home/samuel/P1_Carrera_de_robots/datosE2.csv"
            self.fileNameEvery5 = "/home/samuel/P1_Carrera_de_robots/datosE5.csv"

            if overridePreviousWriteFile:
                with open(self.fileName, 'w') as f:
                    f.write('sensor0;sensor1;sensor2;sensor3;sensor4;left;up;right\n')
        
        # print()
        # print('Waiting 5 seconds...')
        # print()
        # time.sleep(5)
        print('GO!')
        # self.initializeRGB()

    def __del__(self):
        if self.image_sub is not None:
            cv2.destroyAllWindows()
    
    def getKeyPressed(self):
        """
            Looks at the key that the played had pressed
        """

        if self.keyInserted == 'a':
            self.forward_vel, self.rotate_vel = 0.2, 0.5
            self.categoricalOutput = [1.0, 0.0, 0.0]

        elif self.keyInserted == 'd':
            self.forward_vel, self.rotate_vel = 0.2, -0.5
            self.categoricalOutput = [0.0, 0.0, 1.0]

        elif self.keyInserted == 'w':
            self.forward_vel, self.rotate_vel = 0.5, 0
            self.categoricalOutput = [0.0, 1.0, 0.0]
    
    def getAIKey(self):
        """
            Asks the models to choose a key
        """
        predictLeft = self.modelLeft.predict(self.scannnerData)
        predictRight = self.modelRight.predict(self.scannnerData)
        predictForward = self.modelForward.predict(self.scannnerData)

        if np.argmax(predictLeft) == 1:
            self.forward_vel, self.rotate_vel = 0.2, 0.5
            self.categoricalOutput = [1.0, 0.0, 0.0]

        elif np.argmax(predictRight) == 1:
            self.forward_vel, self.rotate_vel = 0.2, -0.5
            self.categoricalOutput = [0.0, 0.0, 1.0]

        elif np.argmax(predictForward) == 1:
            self.forward_vel, self.rotate_vel = 0.5, 0
            self.categoricalOutput = [0.0, 1.0, 0.0]

    def checkNumberOfScansColliding(self, msg: LaserScan):
        return sum(self.laserMinDistance > range for range in msg.range[:30]) + sum(self.laserMinDistance > range for range in msg.range[-30:])
        # return len([True for range in msg.ranges[ :30] if range < self.laserMinDistance]) + len([True for range in msg.ranges[-30:] if range < self.laserMinDistance])

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
        self.scannnerData = [msg.ranges[0], msg.ranges[60], msg.ranges[-60], msg.ranges[120], msg.ranges[-120]]
        self.scannnerData = [num if num < 5 else 5.0 for num in self.scannnerData]


    def writeCurrentState(self):
        """
            Writes the sensor's levels and the player's key pressed in a file
        """

        # Write 5 points scanner
        outputs = ';'.join([str(data) for data in self.scannnerData])
        inputs =  ';'.join([str(data) for data in self.categoricalOutput])

        with open(self.fileName, 'a') as f:
            f.write(outputs + ';' + inputs + '\n')

        # Write every 2º points scanner
        outputs = ';'.join([str(data) for data in self.scannnerDataEvery2])
        inputs =  ';'.join([str(data) for data in self.categoricalOutput])

        with open(self.fileNameEvery2, 'a') as f:
            f.write(outputs + ';' + inputs + '\n')
        
        # Write every 5º points scanner
        outputs = ';'.join([str(data) for data in self.scannnerDataEvery5])
        inputs =  ';'.join([str(data) for data in self.categoricalOutput])

        with open(self.fileNameEvery5, 'a') as f:
            f.write(outputs + ';' + inputs + '\n')


    def command_callback(self, msg: LaserScan):
        """
            Reads the scanner and sends the robot's speed
        """

        # Get the scan values
        self.getScanValues(msg)

        # Get the key pressed or ask the AI to choose a key
        if self.controlledByAI:
            self.getAIKey()
        else:
            self.getKeyPressed()

        # Writte the message
        msg = Twist()
        msg.linear.x = self.forward_vel
        msg.angular.z = self.rotate_vel

        # Write the state in file
        if self.saveInformation:
            self.writeCurrentState()

        # Publish the message
        self.command_pub.publish(msg)


    def odom_callback(self, msg: Odometry):

        # Keep odometry data     
        self.odometryX = msg.pose.pose.position.x
        self.odometryY = msg.pose.pose.position.y



    def initializeRGB(self):
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageRGB_callback)
        cv2.namedWindow('viewRGB')
        cv2.startWindowThread()
        
    def imageRGB_callback(self, msg: Image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('viewRGB', cv_image)
            cv2.waitKey(30)
        except ...:
            rospy.logerr('Could not convert from \'{msg.encoding}\' to \'bgr8\'.')


    def loop(self):
        """
            Loop where the player
        """

        if self.controlledByAI:
            # If the robot is controlled by AI, stop this Thread's execution and just execute the callbacks
            rospy.spin()
        else:
            # Otherwise, if the robot is controlled by the player, recieve keyboard's inputs
            while not rospy.is_shutdown():
                value = input()
                if len(value) > 0:
                    self.keyInserted = value[0]

if __name__ == '__main__':
    rospy.init_node('all_listeners')

    # TODO: insert the AI
    wand = Wander(writeSensorInformation = True, controlledByAI = False, pathSavedModelLeft = None, pathSavedModelRight = None, pathSavedModelForward = None)
    wand.loop()
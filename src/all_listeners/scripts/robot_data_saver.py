#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import math
from nav_msgs.msg import Odometry
import numpy as np
import time
import datetime

import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage

# If save proximity sensor
saveSensorData = False

# If save camera images
saveCamaraData = True
everyXSeconds = 1
folderToSaveImages = '/home/samuel/P1_Carrera_de_robots/Imagenes/'

# If display camera view
displayCameraView = False


"""
    File to control the robot and save it's data in different csv files
"""
class Wander:
    def __init__(self, overridePreviousWriteFile = True):

        self.lastCameraSave = time.time()
        
        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2


        # Video sub
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageRGB_callback)
        
        if displayCameraView:
            self.initializeDisplayRGB()

        # Movement variables
        self.forward_vel = 0
        self.rotate_vel = 0
        self.categoricalOutput = None
        
        # Default key
        self.keyInserted = 'w'
        
        # Scanner data
        self.scannerData = None
        self.scannnerData5Points = None
        self.scannnerDataEvery5 = None
        self.scannnerDataEvery2 = None

        # If set to save information
        if saveSensorData:
            self.fileNameAllPoints  = "/home/samuel/P1_Carrera_de_robots/datosAll.csv"
            self.fileName5Points = "/home/samuel/P1_Carrera_de_robots/datos5.csv"
            self.fileNameEvery2 = "/home/samuel/P1_Carrera_de_robots/datosE2.csv"
            self.fileNameEvery5 = "/home/samuel/P1_Carrera_de_robots/datosE5.csv"

            self.allFiles = [self.fileNameAllPoints, self.fileName5Points, self.fileNameEvery2, self.fileNameEvery5]

            if overridePreviousWriteFile:
                for fileName in self.allFiles:
                    with open(fileName, 'w') as f:
                        f.write('') # Create and clear the file

                with open(self.fileName5Points, 'w') as f:
                    f.write('sensor0;sensor1;sensor2;sensor3;sensor4;left;up;right\n')  # Write header for 5 points


    def __del__(self):
        if self.image_sub is not None:
            cv2.destroyAllWindows()


    def getKeyPressed(self):
        """
            Looks at the key that the played had pressed
        """

        if self.keyInserted == 'a':  # Left
            self.forward_vel, self.rotate_vel = 0, 0.5
            self.categoricalOutput = [1.0, 0.0, 0.0]

        elif self.keyInserted == 'd':# Right
            self.forward_vel, self.rotate_vel = 0, -0.5
            self.categoricalOutput = [0.0, 0.0, 1.0]

        elif self.keyInserted == 'w':# Forward
            self.forward_vel, self.rotate_vel = 0.3, 0
            self.categoricalOutput = [0.0, 1.0, 0.0]

        elif self.keyInserted == 's':# Forward
            self.forward_vel, self.rotate_vel = -0.3, 0
            self.categoricalOutput = [0.0, 1.0, 0.0]

    def checkNumberOfScansColliding(self, msg: LaserScan):
        # Only get the front 60º
        return sum(self.laserMinDistance > range for range in msg.range[:30]) + sum(self.laserMinDistance > range for range in msg.range[-30:])


    def getScanValues(self, msg: LaserScan):
        """
            Get proximity sensor's lvels
        """

        # Get all points
        self.scannerData = list(msg.ranges)
        self.scannerData = [num if num < 5 else 5.0 for num in self.scannerData]

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


    def writeCurrentState(self):
        """
            Writes the sensor's levels and the player's key pressed in a file
        """

        # Write all poitns scanner
        outputs = ';'.join([str(data) for data in self.scannerData])
        inputs =  ';'.join([str(data) for data in self.categoricalOutput])

        with open(self.fileNameAllPoints, 'a') as f:
            f.write(outputs + ';' + inputs + '\n')

        # Write 5 points scanner
        outputs = ';'.join([str(data) for data in self.scannnerData5Points])
        inputs =  ';'.join([str(data) for data in self.categoricalOutput])

        with open(self.fileName5Points, 'a') as f:
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

        # if self.checkNumberOfScansColliding(msg) > self.laserWallCrashPatiente:
        #     rospy.logerr("COLLIDING")

        # Get the scan values
        self.getScanValues(msg)

        # Get the player's key input
        self.getKeyPressed()

        # Writte the message
        msg = Twist()
        msg.linear.x = self.forward_vel
        msg.angular.z = self.rotate_vel

        # Write the state in file
        if saveSensorData:
            self.writeCurrentState()

        # Publish the message
        self.command_pub.publish(msg)


    def initializeDisplayRGB(self):
        cv2.namedWindow('viewRGB')
        cv2.startWindowThread()
        
    def imageRGB_callback(self, msg: Image):
        if displayCameraView:
            try:
                cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
                cv2.imshow('viewRGB', cv_image)
                cv2.waitKey(30)
            except ...:
                rospy.logerr('Could not convert from \'{msg.encoding}\' to \'bgr8\'.')

        if saveCamaraData:
            if time.time() > self.lastCameraSave + everyXSeconds:
                fileName = folderToSaveImages + 'image_' + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.png'
                
                # Convert the ROS image message to a PIL Image object
                cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
                
                # Save the image
                PILImage.fromarray(cv_image).save(fileName, format='PNG')
                
                self.lastCameraSave = time.time()

    def loop(self):
        """
            Loop where the player
        """
        print('Robot created. Insert a move key and press enter to send it to the robot.')
        # Otherwise, if the robot is controlled by the player, recieve keyboard's inputs
        while not rospy.is_shutdown():
            value = input()
            if len(value) > 0:
                self.keyInserted = value[0]

if __name__ == '__main__':
    rospy.init_node('player_controlled_saver_robot')

    wand = Wander()
    wand.loop()
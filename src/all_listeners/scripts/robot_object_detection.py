#!/usr/bin/env python3
"""
Robot controlled by keyboard saving data
    @authors: Samuel Oliva Bulpitt, Luis Jesús Marhuenda Tendero
"""

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

import torch

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

# If save proximity sensor
saveSensorData = False

# If save camera images
saveCamaraData = True
everyXSeconds = 1

# If display camera view
displayCameraView = True

COLORS = ['red', 'green', 'blue', 'yellow', 'purple']
FONT_SIZE = 20
RECTANGLE_WIDHT = 3

"""
    File to control the robot and save it's data in different csv files
"""
class Wander:
    def __init__(self, detectObjectModelPath):
        
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=detectObjectModelPath)
        
        # Subscribers to get the data and publisher to indicate the robot it's speeds
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2

        # Video sub
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageRGB_callback)
        
        self.initializeDisplayRGB()

        # Movement variables
        self.forward_vel = 0
        self.rotate_vel = 0
        
        # Default key
        self.keyInserted = 'w'
        

    def __del__(self):
        cv2.destroyAllWindows()

    def getKeyPressed(self):
        """
            Looks at the key that the played had pressed
        """

        if self.keyInserted == 'a':  # Left
            self.forward_vel, self.rotate_vel = 0, 0.5

        elif self.keyInserted == 'd':# Right
            self.forward_vel, self.rotate_vel = 0, -0.5

        elif self.keyInserted == 'w':# Forward
            self.forward_vel, self.rotate_vel = 0.3, 0

        elif self.keyInserted == 's':# Backward
            self.forward_vel, self.rotate_vel = -0.3, 0


    def command_callback(self, msg: LaserScan):
        """
            Reads the scanner and sends the robot's speed
        """

        # Get the player's key input
        self.getKeyPressed()

        # Writte the message
        msg = Twist()
        msg.linear.x = self.forward_vel
        msg.angular.z = self.rotate_vel

        # Publish the message
        self.command_pub.publish(msg)


    def initializeDisplayRGB(self):
        cv2.namedWindow('viewRGB')
        cv2.startWindowThread()
    
    def drawImageWithBoxes(self, resultYolo, image):
        # Font for text
        font = ImageFont.truetype('FreeSerif.ttf', size=FONT_SIZE)
        
        # Open PIL Image and start drawing
        draw = ImageDraw.Draw(image)

        # Get bounding boxes and labels from results
        boxes = resultYolo.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = resultYolo.pandas().xyxy[0]['name'].values
        classes = resultYolo.pandas().xyxy[0]['class'].values
        
        # Draw bounding boxes and labels on image
        for box, label, classNum in zip(boxes, labels, classes):
            x0, y0, x1, y1 = box.astype(int)
            draw.rectangle([(x0, y0), (x1, y1)], outline=COLORS[classNum], width=RECTANGLE_WIDHT)
            draw.text((x0+RECTANGLE_WIDHT+1, y0-FONT_SIZE-1), label, fill=COLORS[classNum], font=font)
        
        return image
  
    def imageRGB_callback(self, msg: Image):
        if displayCameraView:
            try:
                print('Image callback')
                
                cv_image = CvBridge().imgmsg_to_cv2(msg, "rgb8")
                results = self.model(cv_image)
                PIL_image = self.drawImageWithBoxes(resultYolo=results, image=PILImage.fromarray(cv_image))
                cv2.imshow('viewRGB', cv2.UMat( np.array(PIL_image)))
                # cv2.imshow('viewRGB', CvBridge().imgmsg_to_cv2(msg, "rgb8"))
                cv2.waitKey(30)
            except ...:
                rospy.logerr('Could not convert from \'{msg.encoding}\' to \'bgr8\'.')

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

    wand = Wander(detectObjectModelPath='/home/samuel/Carrera_Robots_VAR/src/all_listeners/models/yolo-weights.pt')
    wand.loop()
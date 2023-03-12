#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import math
from nav_msgs.msg import Odometry

import cv2
from cv_bridge import CvBridge


class Wander:
    def __init__(self, displayRGBImage = False, displayDepthImage = False):
        self.command_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.command_callback)
        self.laserWallCrashPatiente = 15
        self.laserMinDistance = 0.2
        
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.xMax = -math.inf
        self.yMax = -math.inf
        self.xMin = math.inf
        self.yMin = math.inf
        self.initialX = None
        self.initialY = None
        self.needsInitialization = True

        self.displayRGBImage = displayRGBImage
        self.displayDepthImage = displayDepthImage

        if displayRGBImage:
            self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageRGB_callback)
        elif displayDepthImage:
            self.image_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.imageDepth_callback)

    def imageRGB_callback(self, msg: Image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('viewRGB', cv_image)
            cv2.waitKey(30)
        except ...:
            rospy.logerr('Could not convert from \'{msg.encoding}\' to \'bgr8\'.')

    def imageDepth_callback(self, msg: Image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg)
            
            showLog = False
            if showLog:
                # Get image dimension once
                static_run_once = True
                if static_run_once:
                    print(f'Image dimension (Row,Col): {cv_image.shape[0]} x {cv_image.shape[1]}')
                    static_run_once = False

                # Get global max depth value
                max_val = cv2.minMaxLoc(cv_image)[1]
                print("Max value: {}".format(max_val))

                # Get global min depth value
                min_val = cv2.minMaxLoc(cv_image)[0]
                print("Min value: {}".format(min_val))

                # Get depth value at a point
                distance_val = cv_image[100, 100]
                print(f'Distance value: {distance_val}m')

            cv2.imshow('viewDepth', cv_image)
            cv2.waitKey(30)
        except ...:
            rospy.logerr('Could not convert from \'{msg.encoding}\' to \'bgr8\'.')

    def command_callback(self, msg: LaserScan):

        rospy.loginfo(f'Scan message recieved with {len(msg.ranges)} ranges')

        scannnerData = [msg.ranges[0], msg.ranges[60], msg.ranges[-60], msg.ranges[120], msg.ranges[-120]]
        scannnerData = [num if num < 5 else 5.0 for num in scannnerData]

        print()
        print(scannnerData)
        print()

        # Process laser 
        minDistances  = len([True for range in msg.ranges[ :30] if range < self.laserMinDistance])
        minDistances += len([True for range in msg.ranges[-30:] if range < self.laserMinDistance])
        
        if(minDistances > self.laserWallCrashPatiente):
            # Case robot has crashed
            rospy.logerr(f'The robot is touching a wall at {minDistances} points!')
        else:
            # Case robot hasn't crashed
            
            # Create message
            cmd = Twist() 

            # Edit message
            # cmd.linear.x = 0.2  
            # cmd.angular.z = 0.3

            # Publish message
            self.command_pub.publish(cmd)
    
    def odom_callback(self, msg: Odometry):
        
        newX = msg.pose.pose.position.x
        newY = msg.pose.pose.position.y
        
        if self.needsInitialization:
            self.initialX = newX
            self.initialY = newY
            self.needsInitialization = False
        
        if newX < self.xMin:
            self.xMin = newX
        elif newX > self.xMax:
            self.xMax = newX

        if newY < self.yMin:
            self.yMin = newY
        elif newY > self.yMax:
            self.yMax = newY

        # rospy.loginfo(f'xMax, yMax: ({self.xMax}, {self.yMax}), xMin, yMin: ({self.xMin}, {self.yMin})')

    def loop(self):

        rospy.loginfo("All Listeners has been started")

        if self.displayRGBImage:
            cv2.namedWindow('viewRGB')
            cv2.startWindowThread()
        elif self.displayDepthImage:
            cv2.namedWindow('viewDepth')
            cv2.startWindowThread()
                
        rospy.spin()    # Se queda detenido recibiendo y procesando los mensajes

        if self.displayRGBImage or self.displayDepthImage:
            cv2.destroyAllWindows()

        # rate = rospy.Rate(1)  # Specify the loop rate in Hertz. Here we're using 1 Hz, but we'll typically use a value of 10 (i.e., a loop every 100ms).
        
        # while not rospy.is_shutdown():
        #     msg = Twist()  # This message has two components: linear and angular. It allows us to specify these velocities.
        #                    # Each component has three possible values: x, y, z, for each component of the velocity. In the case of
        #                    # robots that receive linear and angular velocity, we need to specify linear x and angular z.
        #     msg.linear.x = self.forward_vel
        #     msg.angular.z = self.rotate_vel
        #     self.command_pub.publish(msg)
        #     rate.sleep()

if __name__ == '__main__':
    rospy.init_node('all_listeners')  # Initialize a new node called 'wander'.
    wand = Wander()  # Create an object of this class and associate it with the ROS system.
    wand.loop()  # Run the main loop.
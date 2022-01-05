#!/usr/bin/env python
""" Simple occupancy-grid-based mapping without localization.

Subscribed topics:
/scan

Published topics:
/map
/map_metadata

"""
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import numpy as np
from helper_functions import HelperFunctions
from map import Map
import xlsxwriter
 
class Mapper(object):
    """
    The Mapper class creates a map from laser scan data.
    """

    def __init__(self):
        """ Start the mapper. """

        rospy.init_node('mapper')
        self._map = Map()
        self._helper = HelperFunctions()    

        # Setting the queue_size to 1 will prevent the subscriber from a
        # buffering scan messages.  This is important because the
        # callback is likely to be too slow to keep up with the scan
        # messages. If we buffer those messages we will fall behind
        # and end up processing really old scans.  Better to just drop
        # old scans and always work with the most recent available.
        rospy.Subscriber('scan',
                         LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber('odom',
                         Odometry, self.odom_callback, queue_size=1)

        # Latched publishers are used for slow changing topics like
        # maps.  Data will sit on the topic until someone reads it.
        self._map_pub = rospy.Publisher('map', OccupancyGrid, latch=True)
        self._map_data_pub = rospy.Publisher('map_metadata',
                                             MapMetaData, latch=True)

        rospy.spin()

    def odom_callback(self, msg):

        global roll, pitch, yaw, Pose, x_r_point, y_r_point, x_r_cell, y_r_cell
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x,
                            orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # print("Global Position of the ROBOT" )
        x_r_point = msg.pose.pose.position.x
        y_r_point = msg.pose.pose.position.y
        theta_r = yaw
        x_r_cell, y_r_cell = self._helper.cell_from_point(x_r_point, y_r_point)
        # print("x_r_point: "+ str(x_r_point))
        # print("y_r_point: "+ str(y_r_point))
        # print("theta_r: "+ str(theta_r))

    def scan_callback(self, scan):
        """ Update the map on every scan callback. """

        # get the object distance from the laser scanner in all of its angles in the range
        r = scan.ranges

        # for every angle in its range
        for i in range(len(r)):

            # if an obstacle has been detected by the laser sensor for thhe angle 'i'
            # i.e. the distance, r[i] is in the range
            if r[i] != np.inf:

                # Convert the angle of the laser from 
                theta_s = math.radians(i)

                # get the co-ordinates from r[i] and theta.
                # (x_c,y_c) are the cell co-ordinates 
                # and the (x_g,y_g) are the actual co-ordinates
                x_c, y_c, x_g, y_g = self._helper.get_grid_cords_from_r_and_theta(
                    r[i], theta_s, x_r_point, y_r_point, yaw)
                 
                # make the occupied cells 'black', by making the probability of the corresponding cell, 1.0
                self._map.grid[x_c, y_c] = 1.0
 
                # write the coordinates of the occupied points to the excel file
                self._helper.write_to_excel(x_g,y_g)

        # rospy.loginfo("Scan is processed, publishing updated map.")
        self.publish_map()

    def publish_map(self):
        """ Publish the map. """
        grid_msg = self._map.to_message()
        self._map_data_pub.publish(grid_msg.info)
        self._map_pub.publish(grid_msg)


if __name__ == '__main__':
    try:
        m = Mapper()
    except rospy.ROSInterruptException:
        pass

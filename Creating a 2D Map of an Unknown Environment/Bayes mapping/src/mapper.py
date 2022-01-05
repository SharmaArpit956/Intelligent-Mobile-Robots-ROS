#!/usr/bin/env python
""" Simple occupancy-grid-based mapping without localization. 

Subscribed topics:
/scan

Published topics:
/map 
/map_metadata

"""
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan

import numpy as np
import math
from helper_functions import HelperFunctions

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from bresenham import bresenham
from map import Map


class Mapper(object):
    """ 
    The Mapper class creates a map from laser scan data.
    """

    def __init__(self):
        """ Start the mapper. """

        rospy.init_node('mapper')
        self._map = Map()
        self._helper = HelperFunctions()

        # Setting the queue_size to 1 will prevent the subscriber from
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
        # maps. Data will sit on the topic until someone reads it.
        self._map_pub = rospy.Publisher(
            'map', OccupancyGrid, latch=True, queue_size=1)
        self._map_data_pub = rospy.Publisher(
            'map_metadata', MapMetaData, latch=True, queue_size=1)

        rospy.spin()

    def odom_callback(self, msg):
        # Initialise the odometry variables
        global roll, pitch, yaw, pos,  x_r_point, y_r_point, x_r_cell, y_r_cell

        orientation_q = msg.pose.pose.orientation
        # print(msg.pose.pose.position) # x, y, z position
        orientation_list = [orientation_q.x,
                            orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        # print("Yaw (heading): ", yaw)
        pos = msg.pose.pose.position

        x_r_point = pos.x
        y_r_point = pos.y

        x_r_cell, y_r_cell = self._helper.cell_from_point(x_r_point, y_r_point)
        # print("Position: ", pos)

    # theta should be between 0 and 360 degrees. Therefore,
    # -theta --> theta        eg. -40  --> 40
    # -theta - 360 --> theta  eg. -400 = -40 - 360 --> 40
    # theta + 360 --> theta   eg.  400 = 40 + 360 --> 40
    def direction_correction(self, theta):
        while theta > 360 or theta < 0:
            if theta < 0:
                if abs(theta) < 360:
                    theta = abs(theta)
                else:
                    theta = abs(theta)-360
            elif theta > 360:
                theta = theta-360
        return math.radians(theta)

    def update_cells(self, x_g, y_g, x_c, y_c, sensor_offset):

        # Store all of the grid cells intersecting the robot and obstacles position
        line_cells = list(
            bresenham(x_r_cell, y_r_cell, x_c, y_c))

        # For all the intersecting cells between the robots position and the obstacles position, do the following
        for i in range(len(line_cells)):
            # ith line cell
            line_cell = [line_cells[i]
                         [0], line_cells[i][1]]

            # If the ith line cell is not the obsacle's cell or the robot's cell
            if line_cell != [x_c, y_c] or line_cell != [x_r_cell, y_r_cell]:

                # The slope(direction) of the line joining robot to line_cell
                cell_robot_direction = math.degrees(
                    math.atan2(line_cells[i][1] - y_r_cell, line_cells[i][0] - x_r_cell))

                # radians to degrees
                sensor_angle = math.degrees(sensor_offset)

                # if the ith line_cell is in the line of sight of the robot
                if cell_robot_direction == sensor_angle:
                    # Draw the intersection point to the map that is the occupancy grid
                    self.bayes_theorem(
                        line_cells[i][0], line_cells[i][1], 0.0)

                # Else if there is some direction error
                else:
                    # as resolution is the width and height of the cell, resolution/2 is the half width and height
                    half_cell_length = self._map.resolution / 2

                    # get the direction of all the four lines joining the robot and the four vertices of the cell
                    upper_left_vertex_to_robot_direction = abs(math.degrees(math.atan2(
                        (line_cells[i][1] + half_cell_length) - y_r_cell, (line_cells[i][0] - half_cell_length) - x_r_cell)))
                    bottom_left_vertex_to_robot_direction = abs(math.degrees(math.atan2(
                        (line_cells[i][1] - half_cell_length) - y_r_cell, (line_cells[i][0] - half_cell_length) - x_r_cell)))
                    upper_right_vertex_to_robot_direction = abs(math.degrees(math.atan2(
                        (line_cells[i][1] + half_cell_length) - y_r_cell, (line_cells[i][0] + half_cell_length) - x_r_cell)))
                    bottom_right_vertex_to_robot_direction = abs(math.degrees(math.atan2(
                        (line_cells[i][1] - half_cell_length) - y_r_cell, (line_cells[i][0] + half_cell_length) - x_r_cell)))

                    direction_error = sensor_angle - cell_robot_direction
                    # Brings the angle between 0 and 360 degrees
                    direction_error = self.direction_correction(
                        direction_error)

                    # get the maximum direction error among the four vertices
                    maximum_direction_error = max(
                        upper_left_vertex_to_robot_direction, bottom_left_vertex_to_robot_direction, upper_right_vertex_to_robot_direction, bottom_right_vertex_to_robot_direction)

                    if direction_error <= maximum_direction_error:
                        # Mark the line_cell with the appropriate probability on the map according to the Bayes theorem
                        self.bayes_theorem(line_cells[i][0], line_cells[i][1], (
                            direction_error / maximum_direction_error) * 0.5)

    def bayes_theorem(self, x_c, y_c, new_probability):

        # prior or posterior probability
        prior_probability = self._map.grid[x_c, y_c]

        # The numerator and denominator for calculating probability
        numerator = new_probability * prior_probability
        denominator = new_probability * prior_probability + \
            (1 - new_probability) * (1 - prior_probability)

        probability_of_interest = numerator / denominator

        # mark the cell (x_c,y_c) with the probability_of_interest on the map
        self._map.grid[x_c, y_c] = probability_of_interest

    def scan_callback(self, scan):

        # get the object distance from the laser scanner in all of its angles in the range
        r = scan.ranges

        # For all of the robots sensors, do the following
        for i in range(len(r)):

            # If the currently iterated sensor is offset by '30' degrees from either side of the robots heading (60 degree cone), do the following
            if i < 30 or i > 330:
                # If the currently iterated sensor has detected an object, do the following
                if r[i] != np.Inf:

                    # Calculate the the angular offset of the sensor
                    theta_s = math.radians(i)

                    # get the co-ordinates from r[i] and theta.
                    # (x_c,y_c) are the cell co-ordinates
                    # and the (x_g,y_g) are the actual co-ordinates
                    x_c, y_c, x_g, y_g = self._helper.get_grid_cords_from_r_and_theta(
                        r[i], theta_s, pos.x, pos.y, yaw)

                    # Update cells between the robots position and the obstacle
                    self.update_cells(
                        x_g, y_g, x_c, y_c,  yaw + theta_s)

                    if self._map.grid[x_c, y_c] > 0.5:
                        # write the coordinates of the occupied points to the excel file
                        self._helper.write_to_excel(x_g,y_g)

                    # update the probaility of the robot's cell with a value little more than 0.0
                    # as there is no chance for improvement at absolute 0.0
                    self.bayes_theorem(x_r_cell, y_r_cell, 0.001) 

                    # update the probaility of the obstacle's cell with a value little less than 1.0
                    # as there is no chance for improvement at absolute 1.0
                    self.bayes_theorem(x_c, y_c, 0.999)

        # Now that the map was updated, so publish it!
        rospy.loginfo("Scan is processed, publishing updated map.")
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

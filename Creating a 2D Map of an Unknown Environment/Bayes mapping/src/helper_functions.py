#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import numpy as np
from map import Map
import xlsxwriter

class HelperFunctions(object):
    
    # init method or constructor   
    def __init__(self):  

        # variables for writing the coordinates of the occupied cells to the excel file
        self.workbook = xlsxwriter.Workbook('coordinates.xlsx')
        self.worksheet = self.workbook.add_worksheet()
        self.worksheet.write("A1", "X co-ordinate")
        self.worksheet.write("B1", "Y co-ordinate")
        self.row = 1
        self.col = 0
        self.max_rows=1000
        self._map = Map()
         
    def cell_from_point(self, x_g, y_g):
        

        # co-ordinates of the origin of the map (-3.5,-3.5)
        x_o = self._map.origin_x
        y_o = self._map.origin_y
 
        if x_g >= x_o and x_g <= -x_o and y_g >= y_o and y_g <= -y_o:
            # positioning the origin in the map to the center of the grid
            x_c = int((x_g - self._map.origin_x) /
                        self._map.resolution)
            y_c = self._map.height - \
                int((y_g - self._map.origin_y) / self._map.resolution)

        return x_c, y_c

    def get_grid_cords_from_r_and_theta(self, r, theta_s, x_r_point, y_r_point, yaw):
        robot_radius = 0.00

        # print("According to Robot's Co-ordinate System")
        x_s = math.cos(theta_s) * r
        y_s = math.sin(theta_s) * r
        # print("r[{0}]= {1}".format(i, object_distance[i]))
        # print("theta_r: " + str(yaw))
        # print("x_s[{0}] = {1}".format(i, x_s))
        # print("y_s[{0}] = {1}".format(i, y_s))

        # print("According to Global Co-ordinate System")
        x_g = x_s * math.cos(yaw) - y_s * math.sin(yaw) + x_r_point
        y_g = x_s * math.sin(yaw) + y_s * math.cos(yaw) + y_r_point
        # x_g = round(x_g, 1)
        # y_g = round(y_g, 1)
        # print("x_g[{0}] = {1}".format(i, x_g))
        # print("y_g[{0}] = {1}".format(i, y_g))

        # print("Co-ordinates of the Occupancy Grid are : ")
        # x_g = round(x_g, 1)

        x_c, y_c = self.cell_from_point(x_g, y_g)

        # print("x co-ordinate  = {0}".format(x_c))
        # print("y co-ordinate  = {0}".format(y_c))
 
        return x_c, y_c,x_g,y_g

    # To write the coordinates of the occupied cells to the excel file
    def write_to_excel(self, x_g, y_g):

        # write the co-ordinate to the next row in the excel file
        self.worksheet.write(self.row, self.col, x_g)
        self.worksheet.write(self.row, self.col + 1, y_g)
        self.row += 1

        # just print a total of max_rows=1000 rows in the excel file named coordinates.xlsx, stored in the same folder 
        if self.row ==self.max_rows:
            self.workbook.close()
 
         
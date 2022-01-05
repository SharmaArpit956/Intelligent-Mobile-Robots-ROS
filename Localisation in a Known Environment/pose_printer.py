#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import xlsxwriter
import re
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

# variables for writing the Estimated Robot position and heading to the excel file
workbook = xlsxwriter.Workbook('estimated_robot_pose.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write("A1", "Time (in seconds)")
worksheet.write("B1", "X co-ordinate")
worksheet.write("C1", "Y co-ordinate")
worksheet.write("D1", "Theta")
row = 1
col = 0
max_rows=500
seconds=1

def callback(data):
      
    global seconds
    estpose= data.pose
 
    estposition = estpose.position
    estorientation = estpose.orientation

    quaternion_pose = [estorientation.x, estorientation.y,
                        estorientation.z, estorientation.w]
    _, _, yaw = euler_from_quaternion(quaternion_pose)

    x=estposition.x
    y=estposition.y
    theta=math.degrees(yaw)

    to_be_printed = '  After {seconds} seconds, the estimated Robot position and heading using HAC algorithm was : ( x = {x: .1f}, y = {y: .1f}, theta = {theta: .2f} )'.format(
       seconds=seconds, x=x, y=y, theta=theta)

    print
    print(to_be_printed)
 
    # write the the Estimated Robot position and heading to the excel file
    global workbook, worksheet, row, col, max_rows
    worksheet.write(row, col, seconds)
    worksheet.write(row, col+1,  x)
    worksheet.write(row, col + 2,  y)
    worksheet.write(row, col + 3, theta)
    row += 1
    seconds += 1

    # just print a total of max_rows rows in the excel file named estimated_robot_pose.xlsx, stored in the same folder 
    if row ==max_rows:
        workbook.close()
 
    rospy.sleep(1)
       

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('/estimatedpose', PoseStamped, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

# # To write the coordinates of the occupied cells to the excel file
# def write_to_excel( x_g, y_g,theta):

    
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from pf_base import PFLocaliserBase
import math
import rospy
import numpy as np
import scipy as sp
from util import rotateQuaternion, getHeading
import random
from numpy import hstack, array, vstack
from scipy.cluster.vq import kmeans, kmeans2, vq
from scipy.cluster.hierarchy import linkage, fcluster
import copy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from time import time
# import statistics


class PFLocaliser(PFLocaliserBase):
    # Initializing Parameters
    def __init__(self):
        # Superclass Constructor
        super(PFLocaliser, self).__init__()

        # SENSOR MODEL PARAMETERS

        # Total number of readings of the laser sensor for weight calculations
        self.NUMBER_PREDICTED_READINGS = 60
        # Total number of particles in the Particle Filter
        self.NUMBER_OF_PARTICLES = 1000

        # self.CLUSTRING_TECHNIQUE= "HAC"    # "global mean", "best particle"

        ######################################################################
        ### FOR INITIALIZING PARTICLE CLOUD ###
        ######################################################################
        # Noise scaling factors and sigma values for INITIALIZING Particle Cloud
        # Initial Odometry model x axis (forward) noise scaling factor
        self.INITIAL_ODOM_TRANSLATION_NOISE_SCALING_FACTOR = 0.1
        # Initial Odometry model y axis (side-to-side) noise scaling factor
        self.INITIAL_ODOM_DRIFT_NOISE_SCALING_FACTOR = 0.1
        # Initial Odometry model z axis (rotation) noise scaling factor
        self.INITIAL_ODOM_ROTATION_NOISE_SCALING_FACTOR = 0.07
        # Initial Gaussian Standard Deviation - Distance(spread) between particles positions
        self.INITIAL_GAUSS_SIGMA = 150
        # Initial VonMises Standard Deviation - Difference(spread) between particles rotations
        self.INITIAL_VONMISES_SIGMA = 100

        ######################################################################
        ### FOR UPDATING PARTICLE CLOUD ###
        ######################################################################
        # Noise scaling factors and sigma values for UPDATING Particle Cloud
        # Odometry model x axis (forward) noise scaling factor for updating Particle Cloud
        self.UPDATE_ODOM_TRANSLATION_NOISE_SCALING_FACTOR = 1  # 0.05
        #   Odometry model y axis (side-to-side) noise scaling factor for updating Particle Cloud
        self.UPDATE_ODOM_DRIFT_NOISE_SCALING_FACTOR = 1  # 0.05
        #   Odometry model z axis (rotation) noise scaling factor for updating Particle Cloud
        self.UPDATE_ODOM_ROTATION_NOISE_SCALING_FACTOR = 0.15
        #   Gaussian Standard Deviation - Distance(spread) between particles positions for updating Particle Cloud
        self.UPDATE_GAUSS_SIGMA = 1
        #   VonMises Standard Deviation - Difference(spread) between particles rotations for updating Particle Cloud
        self.UPDATE_VONMISES_SIGMA = 75
        #   Noise reduction factor
        self.NOISE_REDUCTION_FACTOR = 1.0001

        # Pose Estimation Constant - Cut off distance (difference between a new cluster and actual particle)
        # of the hierarchical clustering tree for HAC algorithm
        self.CUTOFF_DISTANCE = 0.35

    # Roulette Wheel Selection for Resampling
    # The function is likely to return an index of high weight
    def roulette_wheel_selection(self, weightArray, totalWeight):
        some_percent_total_weight = random.random() * totalWeight
        for j in range(len(weightArray)):
            some_percent_total_weight -= weightArray[j]
            index = j
            if some_percent_total_weight <= 0:
                break
        return index

    # The function adds Gaussian and von Mises Noise to the particles
    def add_random_noise(self, pose):
        # Adds noise to the forward/backward direction
        pose.position.x += random.gauss(0,
                                        self.UPDATE_GAUSS_SIGMA) * self.UPDATE_ODOM_TRANSLATION_NOISE_SCALING_FACTOR
        # Adds noise to the side-to-side direction
        pose.position.y += random.gauss(0,
                                        self.UPDATE_GAUSS_SIGMA) * self.UPDATE_ODOM_DRIFT_NOISE_SCALING_FACTOR
        # Adds noise to the rotation angle
        rotationAngle = ((random.vonmisesvariate(
            0, self.UPDATE_VONMISES_SIGMA) - math.pi) * self.UPDATE_ODOM_ROTATION_NOISE_SCALING_FACTOR)
        pose.orientation = rotateQuaternion(pose.orientation, rotationAngle)

        # ADDITIONAL WORK - Reduce the noise till a stage as the algorithm progresses
        if self.UPDATE_ODOM_TRANSLATION_NOISE_SCALING_FACTOR > 0.05:
            self.UPDATE_ODOM_TRANSLATION_NOISE_SCALING_FACTOR = self.UPDATE_ODOM_TRANSLATION_NOISE_SCALING_FACTOR / \
                self.NOISE_REDUCTION_FACTOR
        if self.UPDATE_ODOM_DRIFT_NOISE_SCALING_FACTOR > 0.05:
            self.UPDATE_ODOM_DRIFT_NOISE_SCALING_FACTOR = self.UPDATE_ODOM_DRIFT_NOISE_SCALING_FACTOR / \
                self.NOISE_REDUCTION_FACTOR
        if self.UPDATE_ODOM_ROTATION_NOISE_SCALING_FACTOR > 0.07:
            self.UPDATE_ODOM_ROTATION_NOISE_SCALING_FACTOR = self.UPDATE_ODOM_ROTATION_NOISE_SCALING_FACTOR / \
                self.NOISE_REDUCTION_FACTOR

        # return the position after adding noise
        return pose

    # Set Particle Cloud to Initial Pose Plus Noise
    def initialise_particle_cloud(self, initialpose):

        # Create an instance of the poseArray
        particle_array = PoseArray()
        # For every particle, add noise to all its component directions: x,y,theta
        for i in range(self.NUMBER_OF_PARTICLES):
            # pose of the current particle
            current_pose = Pose()

            # Calculate Gaussian Noise for the X-Coordinate
            x_noise = random.gauss(0, self.INITIAL_GAUSS_SIGMA) * \
                self.INITIAL_ODOM_DRIFT_NOISE_SCALING_FACTOR
            # Calculate Gaussian Noise for the Y-Coordinate
            y_noise = random.gauss(0, self.INITIAL_GAUSS_SIGMA) * \
                self.INITIAL_ODOM_TRANSLATION_NOISE_SCALING_FACTOR

            # Add Gaussian Noise to the initial position of the particle for the X-Coordinate
            current_pose.position.x = initialpose.pose.pose.position.x + x_noise
            # Add Gaussian Noise to the initial position of the particle for the Y-Coordinate
            current_pose.position.y = initialpose.pose.pose.position.y + y_noise
            # Add von Mises Noise to the initial angle of the particle
            rotation_angle = ((random.vonmisesvariate(
                0, self.INITIAL_VONMISES_SIGMA) - math.pi) * self.INITIAL_ODOM_ROTATION_NOISE_SCALING_FACTOR)
            current_pose.orientation = rotateQuaternion(
                initialpose.pose.pose.orientation, rotation_angle)
            # Add the pose of the particle to the particle_array
            particle_array.poses.append(current_pose)
        return particle_array

    # Update Particlecloud, Given Map and Laser Scan
    # Function resamples based on the weights acquired by
    def update_particle_cloud(self, scan):
        # particles which is obtained from the sensor model
        self.latest_scan = scan
        # Create an instance of the poseArray
        particle_array = PoseArray()

        weights = []
        totalWeight = 0
        for pose in self.particlecloud.poses:   # Finding total weight
            thisWeight = self.sensor_model.get_weight(scan, pose)
            weights.append(thisWeight)
            totalWeight += thisWeight

        # Roulette Wheel Algorithm picks the highly weighted
        for _ in range(len(self.particlecloud.poses)):
            # particles with high probability and adds noise
            index = self.roulette_wheel_selection(weights, totalWeight)
            # to generate updated resampled particles.
            particle_array.poses.append(copy.deepcopy(
                self.particlecloud.poses[index]))
        for thisPose in particle_array.poses:
            thisPose = self.add_random_noise(thisPose)

        self.particlecloud = particle_array

    def estimate_pose(self):

        ###################################           HAC ALGORITHM            ##########################################################
        # Hierarchical Agglomerative Clustering (HAC) for selecting the most dense and promising cluster and get rid of outliers

        position_x = []
        position_y = []
        orientation_z = []
        orientation_w = []

        # Forming a matrix of with the features of the samples
        for particle in self.particlecloud.poses:
            # as columns of a matrix. positions x and y and orienation
            position_x.append(particle.position.x)
            # w and z are the only features that concern us.
            position_y.append(particle.position.y)
            orientation_z.append(particle.orientation.z)
            orientation_w.append(particle.orientation.w)

        position_x = np.array(position_x)
        position_y = np.array(position_y)
        orientation_z = np.array(orientation_z)
        orientation_w = np.array(orientation_w)

        # Stack arrays in sequence vertically (row wise).
        observation_vectors = np.column_stack(
            (position_x, position_y, orientation_z, orientation_w))

        # other tried linkages were: single, complete, average links and centroid
        encoded_hierarchical_clustering = linkage(observation_vectors, 'ward')

        # As the criteria used by HAC is distance
        # Function for clustering that outputs an array of cluster IDs for each particle.
        particle_clusterIDs = fcluster(
            encoded_hierarchical_clustering, self.CUTOFF_DISTANCE, criterion='distance')

        num_clusters = max(particle_clusterIDs)  # total number of clusters
        # Array holding number of particles in each array
        num_cluster_particles = [0] * num_clusters
        # Array holding the total weight of each cluster
        cluster_weight_sums = [0] * num_clusters

        # For particles in every cluster
        for i, particle_clusterID in enumerate(particle_clusterIDs):
            particle = self.particlecloud.poses[i]
            # weight of the crrent particle
            weight = self.sensor_model.get_weight(self.latest_scan, particle)

            # count the number of particles in current cluster
            num_cluster_particles[particle_clusterID-1] += 1

            # the total weight of current cluster
            cluster_weight_sums[particle_clusterID-1] += weight

        # Finding cluster with maximum weight
        # '+1' as IDs start from 1 unlike arrays
        best_cluster_ID = cluster_weight_sums.index(
            max(cluster_weight_sums)) + 1
        num_best_cluster_particles = num_cluster_particles[best_cluster_ID-1]

        # variables for sum of componenets of the positions for each cluster
        x_sum = 0
        y_sum = 0
        z_sum = 0
        w_sum = 0
        # Individual sums of all componenets of the positions for each cluster
        for i, particle_clusterID in enumerate(particle_clusterIDs):
            particle = self.particlecloud.poses[i]
            if (particle_clusterID == best_cluster_ID):
                x_sum += particle.position.x
                y_sum += particle.position.y
                z_sum += particle.orientation.z
                w_sum += particle.orientation.w

        # variable for robots estimatedpose
        estpose = Pose()
        # Find average position x and position y , orientation w, orientation z for robot's pose estimation
        estpose.position.x = x_sum/num_best_cluster_particles
        estpose.position.y = y_sum/num_best_cluster_particles
        estpose.orientation.z = z_sum/num_best_cluster_particles
        estpose.orientation.w = w_sum/num_best_cluster_particles
        #
        #
        #
        #
        #
        #
        #
        #
        # ###################################           GLOBAL  MEAN TECHNIQUE            ##########################################################
        # # Global mean technique  which straight away uses just the average of ALL the particles

        # estpose = Pose()
        # x_sum = 0
        # y_sum = 0
        # qz_sum = 0
        # qw_sum = 0
        # for num, i in enumerate(self.particlecloud.poses):
        #     x_sum += i.position.x
        #     y_sum += i.position.y
        #     qz_sum += i.orientation.z
        #     qw_sum += i.orientation.w

        # estpose.position.x = x_sum / self.NUMBER_OF_PARTICLES
        # estpose.position.y = y_sum / self.NUMBER_OF_PARTICLES
        # estpose.orientation.z = qz_sum / self.NUMBER_OF_PARTICLES
        # estpose.orientation.w = qw_sum / self.NUMBER_OF_PARTICLES
        #
        #
        #
        #
        #
        #
        #
        #
        # ###################################           GLOBAL  MEDIAN TECHNIQUE            ##########################################################
        # # Global median technique  which straight away uses just the median of ALL the particles (NOT EFFECTED BY OUTLIERS)

        # estpose = Pose()
        # x_array = [0]*self.NUMBER_OF_PARTICLES
        # y_array = [0]*self.NUMBER_OF_PARTICLES
        # qz_array = [0]*self.NUMBER_OF_PARTICLES
        # qw_array = [0]*self.NUMBER_OF_PARTICLES

        # for num, i in enumerate(self.particlecloud.poses):
        #     x_array[num] = i.position.x
        #     y_array[num] = i.position.y
        #     qz_array[num] =  i.orientation.z
        #     qw_array[num] =  i.orientation.w

        # estpose.position.x = np.median(x_array)
        # estpose.position.y = np.median(y_array)
        # estpose.orientation.z = np.median(qz_array)
        # estpose.orientation.w = np.median(qw_array)
        #
        #
        #
        #
        #
        #
        #
        # ###################################           BEST PARTICLE  TECHNIQUE          ##########################################################
        # # Best Particle  which straight away uses the best particle (one with the highest weight)

        # weights = []  # array for storing all weights
        # for pose in self.particlecloud.poses:  # find weights for all the particles
        #     thisWeight = self.sensor_model.get_weight(self.latest_scan, pose)
        #     weights.append(thisWeight)

        # # find the particle with maximum weight
        # best_particle = self.particlecloud.poses.index(max(weights))

        # # make estimated pose to be returned as that of the best particle
        # estpose.position.x = best_particle.position.x
        # estpose.position.y = best_particle.position.y
        # estpose.orientation.z = best_particle.orientation.z
        # estpose.orientation.w = best_particle.orienation.w

        # This estimated pose automatically gets printed to CONSOLE and an EXCEL file because of a dedicated listener implemented
        return estpose

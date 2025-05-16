import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

class RobotariumVisualizer:
    def __init__(self, number_of_robots, initial_poses):
        """
        Implementation copied from original visualization logic in robotarium_python_simulator/rps/robotarium_abc.py
        """

        self.number_of_robots = number_of_robots

        # geometric info
        self.boundaries = [-1.6, -1, 3.2, 2]
        self.robot_length = 0.095
        self.robot_width = 0.09

        # storing visualization elements
        self.figure = []
        self.axes = []
        self.left_led_patches = []
        self.right_led_patches = []
        self.chassis_patches = []
        self.right_wheel_patches = []
        self.left_wheel_patches = []
        self.base_patches = []
        self.figure, self.axes = plt.subplots()

        self.poses = initial_poses
        self.axes.set_axis_off()

        # draw robots
        for i in range(number_of_robots):
            p = patches.Rectangle((self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                            0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))), self.robot_length, self.robot_width, angle=(self.poses[2, i] + math.pi/4) * 180/math.pi, facecolor='#FFD700', edgecolor='k')

            rled = patches.Circle(self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),
                                    self.robot_length/2/5, fill=False)
            lled = patches.Circle(self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                    0.015*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                    self.robot_length/2/5, fill=False)
            rw = patches.Circle(self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                            0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                                            0.02, facecolor='k')
            lw = patches.Circle(self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                            0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))),\
                                            0.02, facecolor='k')
            self.chassis_patches.append(p)
            self.left_led_patches.append(lled)
            self.right_led_patches.append(rled)
            self.right_wheel_patches.append(rw)
            self.left_wheel_patches.append(lw)

            self.axes.add_patch(rw)
            self.axes.add_patch(lw)
            self.axes.add_patch(p)
            self.axes.add_patch(lled)
            self.axes.add_patch(rled)

        # draw arena
        self.boundary_patch = self.axes.add_patch(patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

        self.axes.set_xlim(self.boundaries[0]-0.1, self.boundaries[0]+self.boundaries[2]+0.1)
        self.axes.set_ylim(self.boundaries[1]-0.1, self.boundaries[1]+self.boundaries[3]+0.1)
        plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)
    
    def update(self, poses):
        """
        Implementation copied from original visualization logic in robotarium_python_simulator/rps/robotarium.py
        """

        self.poses = poses
        for i in range(self.number_of_robots):
            self.chassis_patches[i].xy = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                    0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
            
            self.chassis_patches[i].angle = (self.poses[2, i] - math.pi/2) * 180/math.pi

            self.chassis_patches[i].zorder = 2

            self.right_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                    0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
            self.right_wheel_patches[i].orientation = self.poses[2, i] + math.pi/4

            self.right_wheel_patches[i].zorder = 2

            self.left_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                    0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
            self.left_wheel_patches[i].orientation = self.poses[2,i] + math.pi/4

            self.left_wheel_patches[i].zorder = 2
            
            self.right_led_patches[i].center = self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                            0.04*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
            self.left_led_patches[i].center = self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                            0.015*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
            self.left_led_patches[i].zorder = 2
            self.right_led_patches[i].zorder = 2 
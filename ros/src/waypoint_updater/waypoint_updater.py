#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
OFFSET = 5
DECEL_RATE = 0.3
STOP_COUNTER_THRESHOLD = OFFSET + LOOKAHEAD_WPS

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.light_idx = None
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.light_wp = None
        self.current_vel = None
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.loop()
        rospy.spin()
    
    def loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                current_idx = self.set_closest_waypoint_idx()
                self.publish_waypoints(current_idx, self.light_idx)
            rate.sleep()

    def red_light_ahead(self, current_idx, light_idx):
        if not light_idx:
            return False
        elif light_idx >= len(self.base_waypoints.waypoints): 
            return True
        elif light_idx == -1:
            return False
        else:
            if light_idx > current_idx:
                return True
            else:
                return False

    def set_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y

        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        
        if self.ahead_of(closest_coord, [x, y]):
            return closest_idx
        else:
            return (closest_idx + 1) % len(self.waypoints_2d)
    
    def ahead_of(self, wp1, wp2):
        x = self.pose.position.x
        y = self.pose.position.y
        
        cl_vect = np.array(wp1)
        prev_vect = np.array(wp2)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            return True
        else:
            return False

    def generate_temp_waypoints(self):
        lane = Lane()
        if self.closest_waypoint_idx == None:
            return lane

        last_index = max(len(self.base_waypoints.waypoints), self.closest_waypoint_idx + LOOKAHEAD_WPS + OFFSET)
        lane.waypoints = self.base_waypoints.waypoints[self.closest_waypoint_idx + OFFSET: last_index]
        return lane

    def publish_waypoints(self, current_idx, light_idx):
        final_lane = self.generate_lane(current_idx, light_idx)
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self, current_idx, light_idx):
        lane = Lane()
        farthest_idx = min(len(self.base_waypoints.waypoints), current_idx + LOOKAHEAD_WPS + OFFSET)

        current_waypoints = self.base_waypoints.waypoints[current_idx + OFFSET:farthest_idx]
        light_ahead = self.red_light_ahead(current_idx, light_idx)
        if light_ahead:
            lane.waypoints = self.decelerate_waypoints(current_waypoints, current_idx, light_idx)            
        else:
            lane.waypoints = current_waypoints
        return lane

    def decelerate_waypoints(self, waypoints, current_idx, light_idx):
        temp = []
       
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            stop_idx = max(light_idx - current_idx - 2, 0)
            dist = self.distance(i, stop_idx)
            vel = self.current_vel
            if i >= stop_idx:
                vel = 0
            elif dist < 25:
                vel = DECEL_RATE * dist
            
            if vel < 1:
                vel = 0
            p.twist.twist.linear.x = vel
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        self.pose = msg.pose
        pass

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d == None:
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints.waypoints]
                self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.light_idx = msg.data
       
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    
    def velocity_cb(self, velocity):
        self.current_vel = velocity.twist.linear.x

    def distance(self, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(self.base_waypoints.waypoints[wp1].pose.pose.position, self.base_waypoints.waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
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
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.light_idx = None
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.light_wp = None
        self.closest_waypoint_idx = None
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.loop()
        rospy.spin()
    
    def loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.set_closest_waypoint_idx()
                self.publish_waypoints()
            rate.sleep()

    def red_light_ahead(self):
        if self.light_idx == None or self.base_waypoints == None:
            return False
        elif self.light_idx >= len(self.base_waypoints.waypoints): 
            return True
        else:
            self.light_wp = self.waypoints_2d[self.light_idx]
            distance = self.distance(self.closest_waypoint_idx, self.light_idx)

            if self.ahead_of(self.light_wp, self.closest_waypoint_idx) and distance <= 33: 
                return True
            else:
                return False

    def set_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        if self.waypoint_tree == None:
            return None

        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]
        if self.ahead_of(closest_coord, prev_coord):
            self.closest_waypoint_idx = closest_idx
        else:
            self.closest_waypoint_idx = (closest_idx + 1) % len(self.waypoints_2d)
    
    def ahead_of(self, wp1, wp2):
        x = self.pose.position.x
        y = self.pose.position.y
        
        cl_vect = np.array(wp1)
        prev_vect = np.array(wp2)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            return False
        else:
            return True

    def generate_temp_waypoints(self):
        lane = Lane()
        if self.closest_waypoint_idx == None:
            return lane

        last_index = max(len(self.base_waypoints.waypoints), self.closest_waypoint_idx + LOOKAHEAD_WPS + OFFSET)
        lane.waypoints = self.base_waypoints.waypoints[self.closest_waypoint_idx + OFFSET: last_index]
        return lane

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        if final_lane != None:
            self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        if self.closest_waypoint_idx == None:
            return None
            
        farthest_idx = min(len(self.base_waypoints.waypoints), self.closest_waypoint_idx + LOOKAHEAD_WPS + OFFSET)

        current_waypoints = self.base_waypoints.waypoints[self.closest_waypoint_idx + OFFSET:farthest_idx]
        if self.red_light_ahead():
            lane.waypoints = self.decelerate_waypoints(current_waypoints)            
        else:
            lane.waypoints = current_waypoints
        return lane

    def decelerate_waypoints(self, waypoints):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.light_idx - self.closest_waypoint_idx - 2, 0)
            dist = self.distance(i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            
            p.twist.twist.linear.x = min(0, wp.twist.twist.linear.x)
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
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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

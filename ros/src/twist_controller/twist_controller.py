import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, dbw_enabled):
        self.min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, self.min_speed, max_lat_accel, max_steer_angle)

        self.speed_controller = PID(0.01, 0.01, 0.01, 0.0, 1.0)

        self.steering_controller = PID(0.2, 0.004, 0.2, -max_steer_angle, max_steer_angle)

        tau = 0.2
        ts = 0.1
        self.lpf = LowPassFilter(tau, ts)

        self.dbw_enabled = dbw_enabled
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()
        self.steering = None


    def control(self, *args, **kwargs):
        target = args[0]
        current = args[1]
        dt = args[2]
        if self.dbw_enabled is False:
            self.speed_controller.reset()
            self.steering_controller.reset()

        error = target.linear.x - current.linear.x
        velocity = self.speed_controller.step(error, dt)

        if velocity > 0:
            throttle = max(0.0, min(1.0,velocity))
            brake = 0.0
        else:
            throttle = 0.0
            brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius * abs(velocity)

        steering = self.yaw_controller.get_steering(target.linear.x, target.angular.z, current.linear.x)
        steering = self.lpf.filt(steering)


        return throttle, brake, steering
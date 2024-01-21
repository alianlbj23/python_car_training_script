from pydantic import BaseModel

from entity.ROS2Point import ROS2Point 
from entity.WheelOrientation import WheelOrientation
from entity.WheelAngularVel import WheelAngularVel
import json
import math

class StateType(BaseModel):
    final_target_pos: ROS2Point
    car_pos: ROS2Point
    car_vel: ROS2Point  # in ROS2 coordinate system
    car_orientation: float = 0  # radians, around ROS2 z axis, counter-clockwise: 0 - 359
    wheel_orientation: WheelOrientation  # around car z axis, counter-clockwise: +, clockwise: -, r/s
    car_angular_vel: float  # r/s, in ROS2 around car z axis, yaw++: -, yaw--: +, counter-clockwise: +, clockwise: -, in Unity:  counter-clockwise: -, clockwise: +
    wheel_angular_vel: WheelAngularVel  # around car wheel axis, front: +, back: -
    min_lidar: list = [] # meter
    min_lidar_direction: list = []
    isFirst: bool = True

    # because orientation is transformed back to Unity coordinate system, here lidar direction alse needs to be transformed back from ROS2 to Unity
    # min_lidar_relative_angle: float # radian, base on car, right(x): 0, front(y): 90,  upper: 180 --->x 0, down: -180 --->x 0

    action_wheel_angular_vel: WheelAngularVel
    action_wheel_orientation: WheelOrientation

class State:
    def __init__(self) -> None:
        self.prev_car_state_training = StateType(final_target_pos=ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_pos=ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_vel=ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_orientation=0.0,
                         wheel_orientation=WheelOrientation(left_front=0.0, right_front=0.0),
                         car_angular_vel=0.0,
                         wheel_angular_vel=WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0,
                                                                  right_front=0.0),
                         min_lidar=[],
                         max_lidar=0.0,
                         min_lidar_direciton=[0.0],
                         isFirst= True,
                         action_wheel_angular_vel=WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0,
                                                                         right_front=0.0),
                         action_wheel_orientation=WheelOrientation(left_front=0.0, right_front=0.0))

        self.current_car_state_training = self.prev_car_state_training
    
    def __euler_from_quaternion(self, orientation):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = orientation[0]
        y = orientation[1]
        z = orientation[2]
        w = orientation[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def __radToPositiveRad(self, rad):
        # left +, right -, up 0, down 180 => clockwise: 0 - 359
        if rad < 0:
            rad = -rad
        elif rad > 0:
            rad = math.pi * 2 - rad

        return rad
    
    def __discretize_wheel_steering_angle(self, action_wheel_steering_angle):  # 20~-20
        if action_wheel_steering_angle < 0:
            return -1
        elif action_wheel_steering_angle > 0:
            return 1
        else:
            return 0

    def __discretize_wheel_angular_vel(self, action_wheel_angular_vel):  # -1200~1200
        if action_wheel_angular_vel < 0:
            return -1
        elif action_wheel_angular_vel > 0:
            return 1
        else:
            return 0
    
    def __parse_json(self, data):
        obs = json.loads(data)

        for key, value in obs.items():
            if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                coordinate_str = value.strip('()')
                coordinates = list(map(float, coordinate_str.split(',')))
                obs[key] = coordinates
        return obs
    
    def __decode(self, obs, ai_action):
        data = self.__parse_json(obs);

        car_quaternion = [data['ROS2CarQuaternion'][0], data['ROS2CarQuaternion'][1],
                          data['ROS2CarQuaternion'][2], data['ROS2CarQuaternion'][3]]
        
        car_roll_x, car_pitch_y, car_yaw_z = self.__euler_from_quaternion(car_quaternion)
        car_orientation = self.__radToPositiveRad(car_yaw_z)

        wheel_quaternion_left_front = [data['ROS2WheelQuaternionLeftFront'][0],
                                       data['ROS2WheelQuaternionLeftFront'][1],
                                       data['ROS2WheelQuaternionLeftFront'][2],
                                       data['ROS2WheelQuaternionLeftFront'][3]]  # 48 49 50 51ROS2WheelQuaternionRightBack
        wheel_left_front_roll_x, wheel_left_front_pitch_y, wheel_left_front_yaw_z = self.__euler_from_quaternion(
            wheel_quaternion_left_front)

        wheel_quaternion_right_front = [data['ROS2WheelQuaternionRightFront'][0],
                                        data['ROS2WheelQuaternionRightFront'][1],
                                        data['ROS2WheelQuaternionRightFront'][2],
                                        data['ROS2WheelQuaternionRightFront'][3]]
        wheel_right_front_roll_x, wheel_right_front_pitch_y, wheel_right_front_yaw_z = self.__euler_from_quaternion(
            wheel_quaternion_right_front)
        
        return car_orientation, wheel_left_front_yaw_z, wheel_right_front_yaw_z, data
        
        
    def update(self, obs, ai_action):
        car_orientation, wheel_left_front_yaw_z, wheel_right_front_yaw_z, data = self.__decode(obs, ai_action)
        
        self.prev_car_state_training = self.current_car_state_training

        self.current_car_state_training = StateType(
            final_target_pos=ROS2Point(x=data['ROS2TargetPosition'][0],
                                       y=data['ROS2TargetPosition'][1],
                                       z=0.0),  # data[5]
            car_pos=ROS2Point(x=data['ROS2CarPosition'][0],
                              y=data['ROS2CarPosition'][1],
                              z=data['ROS2CarPosition'][1]),  # data[2]
                              
            car_vel=ROS2Point(x=data['ROS2CarVelocity'][0],
                              y=data['ROS2CarVelocity'][1],
                              z=0.0),  # data[20]
            car_orientation=car_orientation,
            wheel_orientation=WheelOrientation(left_front=self.__radToPositiveRad(wheel_left_front_yaw_z), \
                                               right_front=self.__radToPositiveRad(wheel_right_front_yaw_z)),

            car_angular_vel=data['ROS2CarAugularVelocity'][2],  # data[21 22]
            # data[28]
            wheel_angular_vel=WheelAngularVel(left_back=data['ROS2WheelAngularVelocityLeftBack'][1],  # data[30]
                                              left_front=data['ROS2WheelAngularVelocityLeftFront'][1],  # data[31][33]
                                              right_back=data['ROS2WheelAngularVelocityRightBack'][1],  # data[34 36]
                                              right_front=data['ROS2WheelAngularVelocityRightFront'][1]
                                              # data[37 39] data[40 41 42 43] ROS2WheelQuaternionLeftBack
                                              ),
            min_lidar=data['ROS2Range'],  # 57 58 59
            min_lidar_direction=data["ROS2RangePosition"],

            action_wheel_angular_vel=WheelAngularVel(left_back=self.__discretize_wheel_angular_vel(ai_action[1]), \
                                                     left_front=self.__discretize_wheel_angular_vel(ai_action[1]), \
                                                     right_back=self.__discretize_wheel_angular_vel(ai_action[1]), \
                                                     right_front=self.__discretize_wheel_angular_vel(ai_action[1])
                                                     ),
            action_wheel_orientation=WheelOrientation(left_front=self.__discretize_wheel_steering_angle(ai_action[0]), \
                                                      right_front=self.__discretize_wheel_steering_angle(ai_action[0])),
            isFirst = data["isFirst"] 
        )
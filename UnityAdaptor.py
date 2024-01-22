import math
import numpy as np
from RL.Utility import clamp
import json
from RL.config import UNITY

DEG2RAD = 0.01745329251


class UnityAdaptor():
    def __init__(self):
        self.action_range = UNITY["action_range"]
        self.steering_angle_range = UNITY["steering_angle_range"]
        self.prev_car_yaw = 0

    def transfer_action(self, ai_action):
        unity_action = [None, None]
        unity_action[0] = ai_action[0] * self.action_range
        unity_action[0] = float(clamp(unity_action[0], -self.action_range, self.action_range))

        unity_action[1] = ai_action[1] * self.action_range
        unity_action[1] = float(clamp(unity_action[1], -self.action_range, self.action_range))

        action_sent_to_unity = [0.0, unity_action[0], unity_action[1]]

        return action_sent_to_unity, unity_action

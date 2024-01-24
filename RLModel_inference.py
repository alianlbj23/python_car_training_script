import torch
import numpy as np
import random
# import time
# import RL.Utility as Utility
from RL.Environment import Environment
from RL.AgentDDPG import Agent
from RL.UnityAdaptor import UnityAdaptor
from RL.entity.State import State
from RL.config import PARAMETER, AGENT

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
# import sys

class RLModel_inference_Node(Node):
    def __init__(self):
        super().__init__("RLModel_inference_Node")
        self.get_logger().info("RL inference start")
        self.subscriber_send_action = self.create_subscription(String, "/Unity2Trainer", self.send_action, 0)
        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, '/Trainer2Unity', 0)
        self.unityState = ""
        self.epoch = 1
        
        self.state = State()
        self.environment_training = Environment()
        self.agent_training = Agent()
        self.unity_adaptor_training = UnityAdaptor()
        self.initialize_seed()
        self.action_Unity_Unity_adaptor = [0, 0]
        self.prev_action_AI_agent = [0,0]
        self.agent_training.load_models(AGENT["load_step"])
    def publish2Ros(self, data):
        data2Ros = Float32MultiArray()
        data2Ros.data = data
        self.publisher_Ai2ros.publish(data2Ros)
        
    def initialize_seed(self):
        seed = 123
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def send_Action_to_Unity(self, prev_pos):
        action_AI_agent = self.agent_training.choose_actions(self.state.current_car_state_training, prev_pos, inference=True)
        action_sent_to_unity, action_Unity_Unity_adaptor = self.unity_adaptor_training.transfer_action(action_AI_agent)
        self.action_Unity_Unity_adaptor = action_Unity_Unity_adaptor
        print(f"speed: [left, right] {action_Unity_Unity_adaptor} ")
        self.publish2Ros(action_sent_to_unity)
        self.prev_action_AI_agent = action_AI_agent

    def send_action(self, msg):
        # time.sleep(0.5)
        self.unityState = msg.data
        self.state.update(self.unityState, self.action_Unity_Unity_adaptor)
        self.send_Action_to_Unity(self.environment_training.prev_pos)

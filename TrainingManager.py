import torch
import numpy as np
import random
import time
import Utility
from Environment import Environment
from AgentDDPG import Agent
from UnityAdaptor import UnityAdaptor
from AINode import AINode
from ROS2NodeManager import ROS2NodeManager
from entity.State import State
from config import PARAMETER

from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

class TrainingManager(Node):
    def __init__(self, mode):
        self.mode = mode
        self.state = State()
        self.environment_training = Environment()
        self.agent_training = Agent()
        self.unity_adaptor_training = UnityAdaptor()
        self.nodeManager = ROS2NodeManager()
        self.ROSnode_transfer_data = AINode()
        self.initialize_ros_node()
        self.initialize_seed()
        self.action_Unity_Unity_adaptor = [0, 0]
        self.sum_of_reward_in_one_episode = 0
        self.done = False

        self.loss_training_Critic = []
        self.loss_training_Actor = [] 
        
        self.learning_rate_critic_history = []
        self.learning_rate_Actor_history = []
       
        self.critic_loss_history = []
        self.actor_loss_history = []
       
        self.critic_loss_history_ = []
        self.actor_loss_history_ = [] 
        
        self.reward_history = []
        self.reward_history_ = []
        
    def initialize_seed(self):
        seed = 123
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def initialize_ros_node(self):
        self.nodeManager.add_node(self.ROSnode_transfer_data)
        self.nodeManager.start()

    def send_Action_to_Unity(self, prev_pos, trail_original_pos):
        action_AI_agent = self.agent_training.choose_actions(self.state.current_car_state_training, prev_pos, trail_original_pos, inference=False)
        # print(action_AI_agent)
        action_sent_to_unity, action_Unity_Unity_adaptor = self.unity_adaptor_training.transfer_action(action_AI_agent)
        self.action_Unity_Unity_adaptor = action_Unity_Unity_adaptor
        self.ROSnode_transfer_data.publish2Ros(action_sent_to_unity)

        return action_AI_agent

    def receive_State_from_Unity(self):
        unity_new_obs = self.ROSnode_transfer_data.return_unity_state()
        self.state.update(unity_new_obs, self.action_Unity_Unity_adaptor)
        reward, self.done = self.environment_training.step(self.state.prev_car_state_training, self.state.current_car_state_training)
        return reward
    
    def train_one_episode(self):
        while (not self.done):
            action_AI_agent = self.send_Action_to_Unity(self.environment_training.prev_pos, 
                                                        self.environment_training.trail_original_pos)

            time.sleep(0.5)
            reward = self.receive_State_from_Unity()

            self.sum_of_reward_in_one_episode += reward
            self.agent_training.store_transition(self.state, 
                                                 action_AI_agent, 
                                                 reward, 
                                                 int(self.done), 
                                                 self.environment_training.prev_pos, 
                                                 self.environment_training.trail_original_pos)
    
    def setup_new_game(self):
        new_target = [1.0]
        self.ROSnode_transfer_data.publish2Ros(new_target)
        # print(new_target)
        self.state.update(self.ROSnode_transfer_data.unityState, self.action_Unity_Unity_adaptor)
        self.environment_training.restart_game(self.state.current_car_state_training)

    def start_next_episode(self):
        self.done = False
        is_restart_game = self.environment_training.restart_episode()
        if is_restart_game:
            self.setup_new_game()
    
    def reset_learning_data(self):
        self.sum_of_reward_in_one_episode = 0
        self.loss_training_Critic = []
        self.loss_training_Actor = []
    
    def update_learning_data(self):
        loss_Critic_sum = 0
        loss_Actor_sum = 0
        learning_rate_Critic_DDPG = 0
        learning_rate_Actor_DDPG = 0
        for j in range(self.environment_training.episode_ctr):
            loss_critic_per_round, loss_actor_per_round, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG = self.agent_training.learn()
            self.loss_training_Critic.append(loss_critic_per_round)
            loss_Critic_sum += loss_critic_per_round
            self.loss_training_Actor.append(loss_actor_per_round)
            loss_Actor_sum += loss_actor_per_round
        self.reward_history.append(self.sum_of_reward_in_one_episode)
        self.learning_rate_critic_history.append(learning_rate_Critic_DDPG)
        self.learning_rate_Actor_history.append(learning_rate_Actor_DDPG)
        self.critic_loss_history.append(Utility.mean(self.loss_training_Critic))
        self.actor_loss_history.append(Utility.mean(self.loss_training_Actor))
        self.critic_loss_history_.append(loss_Critic_sum)
        self.actor_loss_history_.append(loss_Actor_sum)
        self.reward_history_.append(np.mean(self.reward_history[-50:]))
    
    def plot_data(self, i):
        Utility.plot(self.reward_history_, 
                     self.learning_rate_critic_history, 
                     self.learning_rate_Actor_history, 
                     self.critic_loss_history, 
                     self.actor_loss_history, 
                     i, 
                     path=self.agent_training.path_save_result_plot)

    def train(self):
        load_step = 0
        for i in range(load_step + 1, load_step + PARAMETER["epoch"] + 1):
            if self.ROSnode_transfer_data.not_connect_to_Unity():
                self.state.current_car_state_training = self.ROSnode_transfer_data.connect_Unity(self.state, self.action_Unity_Unity_adaptor)
                self.environment_training.restart_game(self.state.current_car_state_training)
            else:
                # print(self.state.current_car_state_training)
                self.start_next_episode()

            self.reset_learning_data()
            self.train_one_episode()
            self.update_learning_data()

            if (i) % 200 == 0:
                self.agent_training.save_models(i)
                self.plot(i)

if __name__ == '__main__':
    mode = 'train'
    training_manager = TrainingManager(mode)
    training_manager.train()
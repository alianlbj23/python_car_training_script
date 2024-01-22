import torch
import numpy as np
import random
import time
import Utility
from Environment import Environment
from AgentDDPG import Agent
from UnityAdaptor import UnityAdaptor
from entity.State import State
from config import PARAMETER

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import sys

class RLModel_Node(Node):
    def __init__(self, mode):
        super().__init__("RL_Node")
        self.get_logger().info("Reinforced learning start")
        self.subscriber_send_action = self.create_subscription(String, "/Unity2Trainer", self.send_action, 10)
        self.subscriber = self.create_subscription(String, "/Unity2Trainer", self.train, 10)
        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, '/Trainer2Unity', 10)
        self.unityState = ""
        self.epoch = 1
        
        self.mode = mode
        self.state = State()
        self.environment_training = Environment()
        self.agent_training = Agent()
        self.unity_adaptor_training = UnityAdaptor()
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

        self.prev_action_AI_agent = [0,0]
        
    
    def publish2Ros(self, data):
        data2Ros = Float32MultiArray()
        data2Ros.data = data
        self.publisher_Ai2ros.publish(data2Ros)
    
    def connect_Unity(self, state, action_Unity_Unity_adaptor):
        self.unityState = self.return_unity_state()
        state.update(self.unityState, action_Unity_Unity_adaptor)
        return state.current_car_state_training
        
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
        action_AI_agent = self.agent_training.choose_actions(self.state.current_car_state_training, prev_pos, inference=False)
        action_sent_to_unity, action_Unity_Unity_adaptor = self.unity_adaptor_training.transfer_action(action_AI_agent)
        self.action_Unity_Unity_adaptor = action_Unity_Unity_adaptor
        print(f"speed: [left, right] {action_Unity_Unity_adaptor} ")
        self.publish2Ros(action_sent_to_unity)
        self.prev_action_AI_agent = action_AI_agent

    # def receive_State_from_Unity(self):
    #     unity_new_obs = self.ROSnode_transfer_data.return_unity_state()
    #     self.state.update(unity_new_obs, self.action_Unity_Unity_adaptor)
    #     reward, self.done = self.environment_training.step(self.state.prev_car_state_training, self.state.current_car_state_training)
    #     return reward
    
    def train_one_episode(self):
        
        if (not self.done):
            reward, self.done = self.environment_training.step(self.state.prev_car_state_training, self.state.current_car_state_training)
            
            self.sum_of_reward_in_one_episode += reward
            self.agent_training.store_transition(self.state, 
                                                    self.prev_action_AI_agent, 
                                                    reward, 
                                                    int(self.done), 
                                                    self.environment_training.prev_pos)
    
    def setup_new_game(self):
        new_target = [1.0]
        self.publish2Ros(new_target)
        # self.state.update(self.unityState, self.action_Unity_Unity_adaptor)
        self.environment_training.restart_game(self.state.current_car_state_training)

    def start_next_episode(self):
        self.done = False
        is_restart_game = self.environment_training.restart_episode()
        if is_restart_game:
            self.setup_new_game()
        else:
            self.send_Action_to_Unity(self.environment_training.prev_pos)

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

    def send_action(self, msg):
        time.sleep(0.5)
        self.unityState = msg.data
        self.state.update(self.unityState, self.action_Unity_Unity_adaptor)
        if(self.state.current_car_state_training.isFirst == True):
            self.setup_new_game()
        elif self.done:
            self.start_next_episode()
        else:
            self.send_Action_to_Unity(self.environment_training.prev_pos)

    def train(self, msg):
        self.unityState = msg.data
        self.state.update(self.unityState, self.action_Unity_Unity_adaptor)
        self.train_one_episode()
        if self.done == True:
            self.update_learning_data()
            self.reset_learning_data()
            self.epoch += 1

        if (self.epoch) % 200 == 0:
            self.agent_training.save_models(self.epoch)
            self.plot_data(self.epoch)
        
        if(self.epoch == PARAMETER["epoch"]):
            print("Finish training!")
            exit()

if __name__ == '__main__':
    rclpy.init()
    
    mode = 'train'
    training_manager = RLModel_Node(mode)
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(training_manager)
    exe.spin()

    rclpy.shutdown()
    sys.exit(0)
import json
import torch
import numpy as np
import math
from datetime import datetime
import os

import Utility
from TCPServer import Server

from Environment import Environment
# from CustomThread import CustomThread
from AgentDDPG import Agent
from UnityAdaptor import UnityAdaptor
# from Entity import State
import Entity
import random
from config import *
import time

import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import base64
from typing import Tuple


class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")  # ros2Ai #unity2Ros
        self.subsvriber_ = self.create_subscription(String, "/unity2Ros", self.receive_data_from_ros, 10)

        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, '/ros2Unity', 10)  # Ai2ros #ros2Unity

    def publish2Ros(self, data):
        self.data2Ros = Float32MultiArray()
        self.data2Ros.data = data
        self.publisher_Ai2ros.publish(self.data2Ros)

    def receive_data_from_ros(self, msg):
        global unityState
        unityState = msg.data
        # print(unityState)
        # self.msg_from_Unity = msg.data


def spin_pros(ROSnode_transfer_data):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(ROSnode_transfer_data)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)


def returnUnityState():
    while len(unityState) == 0:
        # print(unityState)
        pass
    return unityState

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# path_load_model = os.path.join(os.path.dirname(__file__), 'Model', 'DDPG', '0809_car/model')
# path_save_model = os.path.join(os.path.dirname(__file__), 'Model', 'DDPG', '0809_car/model')
# path_save_result_plot = os.path.join(os.path.dirname(__file__), 'Model', 'DDPG', '0809_car/')
# path_save_log = os.path.join(os.path.dirname(__file__), 'Model', 'DDPG', '0809_car/log')
    
def initialize_envirinment(max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec):
    environment_training = Environment(max_times_in_episode=max_times_in_episode, max_times_in_game=max_times_in_game, end_distance=end_distance
              , stop_target=stop_target,target_fixed_sec=target_fixed_sec)
    return environment_training

def initialize_Agent(load_path, save_path, plot_path, log_path):
    path_load_model = os.path.join(os.path.dirname(__file__), load_path)
    path_save_model = os.path.join(os.path.dirname(__file__), save_path)
    path_save_result_plot = os.path.join(os.path.dirname(__file__), plot_path)
    path_save_log = os.path.join(os.path.dirname(__file__), log_path)
    agent_training = Agent(q_lr, pi_lr, gamma, rho, pretrained=pretrained, new_input_dims=new_input_dims,
                input_dims=input_dims, n_actions=n_actions, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                chpt_dir_load=path_load_model, chpt_dir_save=path_save_model)
    return path_save_result_plot, path_save_log, agent_training

def initialize_UnityAdaptor(action_range, steering_angle_range):
    unity_adaptor_training = UnityAdaptor(action_range=action_range, steering_angle_range=steering_angle_range)
    return unity_adaptor_training

def initialize_state():
    car_state_training = Entity.State(final_target_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         objectUpVector=0,
                         path_closest_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         path_second_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         path_farthest_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_vel=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         car_orientation=0.0,
                         wheel_orientation=Entity.WheelOrientation(left_front=0.0, right_front=0.0),
                         car_angular_vel=0.0,
                         wheel_angular_vel=Entity.WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0,
                                                                  right_front=0.0),
                         min_lidar=[],
                         min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         second_min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         third_min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                         max_lidar=0.0,
                         min_lidar_direciton=[0.0],
                         action_wheel_angular_vel=Entity.WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0,
                                                                         right_front=0.0),
                         action_wheel_orientation=Entity.WheelOrientation(left_front=0.0, right_front=0.0))
    return car_state_training
def initialize_training_environment():
    msg_from_Unity = None
    load_step = load_step
    prev_pos = [0, 0]
    trail_original_pos = [0, 0]
    action_Unity_Unity_adaptor = [0, 0]
    return msg_from_Unity, load_step, prev_pos, trail_original_pos, action_Unity_Unity_adaptor

def connect_Unity(msg_from_Unity, unity_adaptor_training, action_Unity_Unity_adaptor):
    while msg_from_Unity is None:
        msg_from_Unity = returnUnityState()
    car_state_training = unity_adaptor_training.transfer_obs(msg_from_Unity, action_Unity_Unity_adaptor)
    return car_state_training, msg_from_Unity

def setup_newgame(ROSnode_transfer_data, unity_adaptor_training, environment_training, action_Unity_Unity_adaptor):
    new_target = [1.0]
    ROSnode_transfer_data.publish2Ros(new_target)
    print(new_target)
    msg_from_Unity = returnUnityState()
    car_state_training = unity_adaptor_training.transfer_obs(msg_from_Unity, action_Unity_Unity_adaptor)
    environment_training.restart_game(car_state_training)
    return car_state_training, msg_from_Unity

def send_Action_to_Unity(agent_training, car_state_training, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data):
    action_AI_agent = agent_training.choose_actions(car_state_training, prev_pos, trail_original_pos, inference=False)
    print(action_AI_agent)
    action_sent_to_unity, action_Unity_Unity_adaptor = unity_adaptor_training.trasfer_action(action_AI_agent)
    ROSnode_transfer_data.publish2Ros(action_sent_to_unity)

    return action_AI_agent, action_Unity_Unity_adaptor

def receive_State_from_Unity(environment_training, car_state_training, unity_adaptor_training, action_Unity_Unity_adaptor):
    unity_new_obs = returnUnityState()
    new_car_state_training = unity_adaptor_training.transfer_obs(unity_new_obs, action_Unity_Unity_adaptor)
    reward, done, info = environment_training.step(car_state_training, new_car_state_training)
    return new_car_state_training, reward, done, info

def reset_learning_data():
    sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG = (0 for i in range(3))
    loss_training_Critic, loss_training_Actor = ([] for i in range(2))
    loss_Critic_sum = 0
    loss_Actor_sum = 0
    return sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG, loss_training_Critic, loss_training_Actor, loss_Critic_sum, loss_Actor_sum

def Not_connect_to_Unity(msg_from_Unity):
    if msg_from_Unity is None:
        return True
    else:
        return False

def update_learning_data(environment_training, agent_training, loss_training_Critic, loss_training_Actor, reward_history, sum_of_reward_in_one_episode, learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history,
                         critic_loss_history_, actor_loss_history_, reward_history_):
    loss_Critic_sum = 0
    loss_Actor_sum = 0
    for j in range(environment_training.episode_ctr):
        loss_critic_per_round, loss_actor_per_round, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG = agent_training.learn()
        loss_training_Critic.append(loss_critic_per_round)
        loss_Critic_sum += loss_critic_per_round
        loss_training_Actor.append(loss_actor_per_round)
        loss_Actor_sum += loss_actor_per_round
    reward_history.append(sum_of_reward_in_one_episode)
    learning_rate_critic_history.append(learning_rate_Critic_DDPG)
    learning_rate_Avtor_history.append(learning_rate_Actor_DDPG)
    critic_loss_history.append(Utility.mean(loss_training_Critic))
    actor_loss_history.append(Utility.mean(loss_training_Actor))
    critic_loss_history_.append(loss_Critic_sum)
    actor_loss_history_.append(loss_Actor_sum)
    reward_history_.append(np.mean(reward_history[-50:]))
    return learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_, reward_history_

def train_one_episode(done, agent_training, car_state_training, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data, environment_training, sum_of_reward_in_one_episode):
    while (not done):
        # action_AI_agent = agent_training.choose_actions(car_state_training, prev_pos, trail_original_pos, inference=False)
        # action_sent_to_unity, action_Unity_Unity_adaptor = unity_adaptor_training.trasfer_action(action_AI_agent)
        # ROSnode_transfer_data.publish2Ros(action_sent_to_unity)
        action_AI_agent, action_Unity_Unity_adaptor = send_Action_to_Unity(agent_training, car_state_training, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data)

        time.sleep(0.5)

        # unity_new_obs = returnUnityState()
        # new_car_state_training = unity_adaptor_training.transfer_obs(unity_new_obs, action_Unity_Unity_adaptor)
        # reward, done, info = environment_training.step(car_state_training, new_car_state_training)
        new_car_state_training, reward, done, info = receive_State_from_Unity(environment_training, car_state_training, unity_adaptor_training, action_Unity_Unity_adaptor)

        sum_of_reward_in_one_episode += reward
        agent_training.store_transition(car_state_training, action_AI_agent, reward, new_car_state_training, int(done), prev_pos, trail_original_pos)

        prev_pos = info['prev pos']
        trail_original_pos = info['trail original pos']

        car_state_training = new_car_state_training
        return action_AI_agent, prev_pos, trail_original_pos, car_state_training, reward, sum_of_reward_in_one_episode, action_Unity_Unity_adaptor
    
def start_next_episode(msg_from_Unity, car_state_training, environment_training, ROSnode_transfer_data, unity_adaptor_training, action_Unity_Unity_adaptor):
    restart_game = environment_training.restart_episode()
    if restart_game:
        car_state_training, msg_from_Unity = setup_newgame(ROSnode_transfer_data, unity_adaptor_training, environment_training, action_Unity_Unity_adaptor)
    return car_state_training, msg_from_Unity

# 命名規則： 物理意義_在哪被使用到/被產生_哪一種類別
def main(mode):
    print('The mode is:', mode)
    car_state_training = initialize_state()
    environment_training = initialize_envirinment(max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec)
    path_save_result_plot, path_save_log, agent_training = initialize_Agent(load_path, save_path, plot_path, log_path)
    unity_adaptor_training = initialize_UnityAdaptor(action_range, steering_angle_range)
    

    # start subscriber  !! cannot call by a function !!
    rclpy.init()
    ROSnode_transfer_data = AiNode()
    ROSnode_thread = threading.Thread(target=spin_pros, args=(ROSnode_transfer_data,))
    ROSnode_thread.start()
    # start subscriber

    reward_history, reward_history_ = ([] for i in range(2))
    try:
        if mode == 'train':
            learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_ = ([] for i in range(6))
            msg_from_Unity, load_step, prev_pos, trail_original_pos, action_Unity_Unity_adaptor = initialize_training_environment()

            for i in range(load_step + 1, load_step + epoch + 1):
                if Not_connect_to_Unity(msg_from_Unity):
                # start training, first time connect to Unity
                    car_state_training, msg_from_Unity = connect_Unity(msg_from_Unity, unity_adaptor_training, action_Unity_Unity_adaptor)
                    environment_training.restart_game(car_state_training)

                else:
                    print(car_state_training)
                    car_state_training, msg_from_Unity = start_next_episode(msg_from_Unity, car_state_training, environment_training, ROSnode_transfer_data, unity_adaptor_training, action_Unity_Unity_adaptor)

                done = False
                sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG, loss_training_Critic, loss_training_Actor, loss_Critic_sum, loss_Actor_sum = reset_learning_data()

                action_AI_agent, prev_pos, trail_original_pos, car_state_training, reward, sum_of_reward_in_one_episode, action_Unity_Unity_adaptor = train_one_episode(done, agent_training, car_state_training, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data, environment_training, sum_of_reward_in_one_episode)

                learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_, reward_history_ = update_learning_data(environment_training, agent_training, loss_training_Critic, loss_training_Actor, reward_history, sum_of_reward_in_one_episode, learning_rate_critic_history, 
                                                                                                                                                                                                      learning_rate_Avtor_history, critic_loss_history, actor_loss_history, 
                                                                                                                                                                                                      critic_loss_history_, actor_loss_history_, reward_history_)

                if (i) % 200 == 0:
                    agent_training.save_models(i)
                    Utility.plot(reward_history_, learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history,
                                 actor_loss_history, i, path=path_save_result_plot)
                print('episode:', i,
                      ',reward:%.2f' % (sum_of_reward_in_one_episode),
                      ',avg reward:%.2f' % (reward_history_[-1]),
                      ',critic loss:%.2f' % (critic_loss_history[-1]),
                      ',actor loss:%.2f' % (actor_loss_history[-1]),
                      ',ctr:', environment_training.episode_ctr, )

        elif mode == 'test':
            # agent_training.load_models(9000)
            agent_training.eval()
            msg_from_Unity, load_step, prev_pos, trail_original_pos, action_Unity_Unity_adaptor = initialize_training_environment()

            for i in range(load_step + 1, load_step + epoch + 1):
                if Not_connect_to_Unity(msg_from_Unity):
                # start training, first time connect to Unity
                    car_state_training, msg_from_Unity = connect_Unity(msg_from_Unity, unity_adaptor_training, action_Unity_Unity_adaptor)
                    environment_training.restart_game(car_state_training)

                else:
                    car_state_training, msg_from_Unity= start_next_episode(msg_from_Unity, car_state_training, environment_training, ROSnode_transfer_data, unity_adaptor_training, action_Unity_Unity_adaptor)

                done = False
                sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG, loss_training_Critic, loss_training_Actor, loss_Critic_sum, loss_Actor_sum = reset_learning_data()

                action_AI_agent, prev_pos, trail_original_pos, car_state_training, reward, sum_of_reward_in_one_episode, action_Unity_Unity_adaptor = train_one_episode(done, agent_training, car_state_training, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data, environment_training, sum_of_reward_in_one_episode)

                learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_, reward_history_ = update_learning_data
                (environment_training, agent_training, loss_training_Critic, loss_training_Actor, reward_history, sum_of_reward_in_one_episode, learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, 
                 critic_loss_history_, actor_loss_history_, reward_history_)

                print('test run: ', i, ',reward: %.2f,' % (sum_of_reward_in_one_episode), 'ctr: ', environment_training.game_ctr)

    except KeyboardInterrupt:
        print("keyboard")


if __name__ == '__main__':
    # TODO paramaterization
    # mode = 'train'
    mode = 'train'
    main(mode)

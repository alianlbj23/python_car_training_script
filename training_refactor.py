import torch
import numpy as np

import Utility
from Environment import Environment
from AgentDDPG import Agent
from UnityAdaptor import UnityAdaptor
import random
import time


from AINode import AINode
from ROS2NodeManager import ROS2NodeManager
from entity.State import State
from config import PARAMETER

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def initialize_training_environment():
    msg_from_Unity = None
    prev_pos = [0, 0]
    trail_original_pos = [0, 0]
    action_Unity_Unity_adaptor = [0, 0]
    return msg_from_Unity, prev_pos, trail_original_pos, action_Unity_Unity_adaptor

def setup_new_game(ROSnode_transfer_data, state, environment_training, action_Unity_Unity_adaptor):
    new_target = [1.0]
    ROSnode_transfer_data.publish2Ros(new_target)
    print(new_target)
    state.update(ROSnode_transfer_data.unityState, action_Unity_Unity_adaptor)
    environment_training.restart_game(state.current_car_state_training)

def send_Action_to_Unity(agent_training, state, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data):
    action_AI_agent = agent_training.choose_actions(state.current_car_state_training, prev_pos, trail_original_pos, inference=False)
    print(action_AI_agent)
    action_sent_to_unity, action_Unity_Unity_adaptor = unity_adaptor_training.transfer_action(action_AI_agent)
    ROSnode_transfer_data.publish2Ros(action_sent_to_unity)

    return action_AI_agent, action_Unity_Unity_adaptor

def receive_State_from_Unity(ROSnode_transfer_data, environment_training, state, action_Unity_Unity_adaptor):
    unity_new_obs = ROSnode_transfer_data.return_unity_state()
    state.update(unity_new_obs, action_Unity_Unity_adaptor)
    reward, done, info = environment_training.step(state.prev_car_state_training, state.current_car_state_training)
    return reward, done, info

def reset_learning_data():
    sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG = (0 for i in range(3))
    loss_training_Critic, loss_training_Actor = ([] for i in range(2))
    loss_Critic_sum = 0
    loss_Actor_sum = 0
    return sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG, loss_training_Critic, loss_training_Actor, loss_Critic_sum, loss_Actor_sum

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

def train_one_episode(done, agent_training, state, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data, environment_training, sum_of_reward_in_one_episode):
    while (not done):
        action_AI_agent, action_Unity_Unity_adaptor = send_Action_to_Unity(agent_training, state, prev_pos, trail_original_pos, unity_adaptor_training, ROSnode_transfer_data)

        time.sleep(0.5)
        reward, done, info = receive_State_from_Unity(ROSnode_transfer_data, environment_training, state, action_Unity_Unity_adaptor)

        sum_of_reward_in_one_episode += reward
        agent_training.store_transition(state, action_AI_agent, reward, int(done), prev_pos, trail_original_pos)

        prev_pos = info['prev pos']
        trail_original_pos = info['trail original pos']

        return action_AI_agent, prev_pos, trail_original_pos, reward, sum_of_reward_in_one_episode, action_Unity_Unity_adaptor
    
def start_next_episode(state, environment_training, ROSnode_transfer_data, action_Unity_Unity_adaptor):
    is_restart_game = environment_training.restart_episode()
    if is_restart_game:
        setup_new_game(ROSnode_transfer_data, state, environment_training, action_Unity_Unity_adaptor)

# 命名規則： 物理意義_在哪被使用到/被產生_哪一種類別
def main(mode):
    print('The mode is:', mode)
    
    state = State()
    environment_training = Environment()
    agent_training = Agent()
    unity_adaptor_training = UnityAdaptor()
    
    nodeManager = ROS2NodeManager()
    ROSnode_transfer_data = AINode()
    nodeManager.add_node(ROSnode_transfer_data)
    nodeManager.start()

    reward_history, reward_history_ = ([] for i in range(2))
    try:
        if mode == 'train':
            learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_ = ([] for i in range(6))
            msg_from_Unity, prev_pos, trail_original_pos, action_Unity_Unity_adaptor = initialize_training_environment()
            load_step = 0
            for i in range(load_step + 1, load_step + PARAMETER["epoch"] + 1):
                if ROSnode_transfer_data.not_connect_to_Unity():
                    state.current_car_state_training = ROSnode_transfer_data.connect_Unity(state, action_Unity_Unity_adaptor)

                    # unityState = ROSnode_transfer_data.unityState
                    # while len(unityState) == 0:
                    #     print("Waiting for Unity state...")
                    #     time.sleep(1)
                    #     unityState = ROSnode_transfer_data.unityState
                    # state.update(unityState, action_Unity_Unity_adaptor)

                    environment_training.restart_game(state.current_car_state_training)
                else:
                    print(state.current_car_state_training)
                    start_next_episode(state, environment_training, ROSnode_transfer_data, action_Unity_Unity_adaptor)

                done = False
                sum_of_reward_in_one_episode, learning_rate_Critic_DDPG, learning_rate_Actor_DDPG, \
                loss_training_Critic, loss_training_Actor, loss_Critic_sum, loss_Actor_sum = reset_learning_data()

                action_AI_agent, prev_pos, trail_original_pos, \
                reward, sum_of_reward_in_one_episode, action_Unity_Unity_adaptor = train_one_episode(done, agent_training, state, prev_pos, 
                                                                                                     trail_original_pos, unity_adaptor_training, 
                                                                                                     ROSnode_transfer_data, environment_training, 
                                                                                                     sum_of_reward_in_one_episode)

                learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history, \
                actor_loss_history, critic_loss_history_, actor_loss_history_, reward_history_ = update_learning_data(environment_training, agent_training, loss_training_Critic, 
                                                                                                                      loss_training_Actor, reward_history, sum_of_reward_in_one_episode, 
                                                                                                                      learning_rate_critic_history,  learning_rate_Avtor_history, 
                                                                                                                      critic_loss_history, actor_loss_history, critic_loss_history_, 
                                                                                                                      actor_loss_history_, reward_history_)

                if (i) % 200 == 0:
                    agent_training.save_models(i)
                    Utility.plot(reward_history_, learning_rate_critic_history, learning_rate_Avtor_history, critic_loss_history,
                                 actor_loss_history, i, path=agent_training.path_save_result_plot)
                print('episode:', i,
                      ',reward:%.2f' % (sum_of_reward_in_one_episode),
                      ',avg reward:%.2f' % (reward_history_[-1]),
                      ',critic loss:%.2f' % (critic_loss_history[-1]),
                      ',actor loss:%.2f' % (actor_loss_history[-1]),
                      ',ctr:', environment_training.episode_ctr, )

    except KeyboardInterrupt:
        print("keyboard")


if __name__ == '__main__':
    mode = 'train'
    main(mode)

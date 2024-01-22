import RL.DDPG as DDPG
import RL.Utility as Utility
from stable_baselines3.common.noise import NormalActionNoise
import torch.nn.functional as F
import numpy as np
import torch
import abc
import os
from RL.config import AGENT, PATH


class Agent():
    def __init__(self):
        self.path_load_model = os.path.join(os.path.dirname(__file__), PATH["LOAD_PATH"])
        self.path_save_model = os.path.join(os.path.dirname(__file__), PATH["SAVE_PATH"])
        self.path_save_result_plot = os.path.join(os.path.dirname(__file__), PATH["PLOT_PATH"])
        self.path_save_log = os.path.join(os.path.dirname(__file__), PATH["LOG_PATH"])
        self.n_actions = AGENT["n_actions"]
        
        self.q_lr = AGENT["q_lr"]
        self.pi_lr = AGENT["pi_lr"]
        self.rho = AGENT["rho"]
        self.gamma = AGENT["gamma"]
        self.batch_size = AGENT["batch_size"]
        self.input_dims = AGENT["input_dims"]
        self.layer1_size = AGENT["layer1_size"]
        self.layer2_size = AGENT["layer2_size"]

        self.chpt_dir_load = self.path_load_model
        self.chpt_dir_save = self.path_save_model

        # TODO: use DDPG Gaussian as NormalActionNoise  
        self.noise = NormalActionNoise(mean=np.zeros(
            AGENT["n_actions"]), sigma=0.1 * np.ones(AGENT["n_actions"]))
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.updates = 0
        self.pretrained = AGENT["pretrained"]
        self.new_input_dims = AGENT["new_input_dims"]

        if not os.path.exists('Model/DDPG'):
            os.makedirs('Model/DDPG')

        if self.pretrained:
            self.memory = DDPG.ReplayBuffer(input_dims=self.new_input_dims, n_actions=self.n_actions)

            self.pretrained_critic = DDPG.CriticNetwork(self.q_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                        self.chpt_dir_load, self.chpt_dir_save, name='Crirtic_')
            self.pretrained_actor = DDPG.ActorNetwork(self.pi_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                      self.chpt_dir_load, self.chpt_dir_save, name='Actor_')
            self.pretrained_target_critic = DDPG.CriticNetwork(self.q_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                               self.chpt_dir_load, self.chpt_dir_save, name='TargetCrirtic_')
            self.pretrained_target_critic.load_state_dict(self.pretrained_critic.state_dict())
            self.pretrained_target_actor = DDPG.ActorNetwork(self.pi_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                             self.chpt_dir_load, self.chpt_dir_save, name='TargetActor_')
            self.pretrained_target_actor.load_state_dict(self.pretrained_actor.state_dict())

            self.critic = DDPG.PretrainedCriticNetwork(self.pretrained_critic, self.q_lr, self.new_input_dims, self.input_dims,
                                                       self.chpt_dir_load, self.chpt_dir_save, name='Crirtic_')
            self.actor = DDPG.PretrainedActorNetwork(self.pretrained_actor, self.pi_lr, self.new_input_dims, self.input_dims,
                                                     self.chpt_dir_load, self.chpt_dir_save, name='Actor_')
            self.target_critic = DDPG.PretrainedCriticNetwork(self.pretrained_target_critic, self.q_lr, self.new_input_dims,
                                                              self.input_dims, self.chpt_dir_load, self.chpt_dir_save,
                                                              name='TargetCrirtic_')
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor = DDPG.PretrainedActorNetwork(self.pretrained_target_actor, self.pi_lr, self.new_input_dims,
                                                            self.input_dims, self.chpt_dir_load, self.chpt_dir_save,
                                                            name='TargetActor_')
            self.target_actor.load_state_dict(self.actor.state_dict())
        else:
            self.memory = DDPG.ReplayBuffer(input_dims=self.input_dims, n_actions=self.n_actions)

            self.critic = DDPG.CriticNetwork(self.q_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                             self.chpt_dir_load, self.chpt_dir_save, name='Crirtic_')

            self.actor = DDPG.ActorNetwork(self.pi_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                           self.chpt_dir_load, self.chpt_dir_save, name='Actor_')

            self.target_critic = DDPG.CriticNetwork(self.q_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                    self.chpt_dir_load, self.chpt_dir_save, name='TargetCrirtic_')
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor = DDPG.ActorNetwork(self.pi_lr, self.input_dims, self.layer1_size, self.layer2_size, self.n_actions, \
                                                  self.chpt_dir_load, self.chpt_dir_save, name='TargetActor_')
            self.target_actor.load_state_dict(self.actor.state_dict())

        self.update_network_parameters(rho=1)

    def update_network_parameters(self, rho=None):
        if rho is None:
            rho = self.rho
        # rho = self.rho
        else:
            print(rho)
        critic_params = self.critic.named_parameters()
        actor_params = self.actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params_dict = dict(critic_params)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict = dict(target_critic_params)
        target_actor_params_dict = dict(target_actor_params)

        for name in critic_params_dict:
            critic_params_dict[name] = rho * critic_params_dict[name].clone() + \
                                       (1 - rho) * target_critic_params_dict[name].clone()
        self.target_critic.load_state_dict(critic_params_dict)

        for name in actor_params_dict:
            actor_params_dict[name] = rho * actor_params_dict[name].clone() + \
                                      (1 - rho) * target_actor_params_dict[name].clone()
        self.target_actor.load_state_dict(actor_params_dict)

    def learn(self):
        self.updates += 1
        critic_losses, actor_losses = 0, 0

        state, actions, reward, new_state, d = self.memory.sample_buffer(
            self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.device)
        d = torch.tensor(d).to(self.device)

        # if self.pretrained:
        #     for param in self.pretrained_critic.parameters():
        #         param.requires_grad = False
        # print("before no_grad", list(self.pretrained_critic.parameters())[-1])

        with torch.no_grad():
            # print("after no_grad", list(self.pretrained_critic.parameters())[-1])
            self.target_actor.eval()
            target_actions = self.target_actor.forward(new_state)
            self.target_critic.eval()
            critic_value_ = self.target_critic.forward(
                new_state, target_actions)  # batch size

            target = []
            for i in range(self.batch_size):
                target.append(reward[i] + (1 - d[i]) *
                              self.gamma * critic_value_[i])
            target = torch.tensor(target, dtype=torch.float).to(self.device)
            target = target.view(self.batch_size, 1)

        self.critic.eval()
        critic_value = self.critic.forward(state, actions)

        critic_loss = F.mse_loss(target, critic_value)
        critic_losses = critic_loss.item()

        self.critic.train()
        self.critic.optimizer.zero_grad()  # clean the previous grdient
        critic_loss.backward()  # claculate gradient
        self.critic.optimizer.step()  # update paramters

        lr_c = self.get_lr(self.critic.optimizer)

        if self.updates % 1 == 0:
            self.actor.eval()
            actions = self.actor.forward(state)

            self.critic.eval()
            actor_loss = -self.critic.forward(state, actions)
            actor_loss = torch.mean(actor_loss)
            actor_losses = actor_loss.item()

            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            lr_a = self.get_lr(self.actor.optimizer)

            self.update_network_parameters()

        return critic_losses, actor_losses, lr_c, lr_a

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_models(self, tag):
        self.critic.save_checkpoint(tag)
        self.actor.save_checkpoint(tag)
        self.target_critic.save_checkpoint(tag)
        self.target_actor.save_checkpoint(tag)

    def load_models(self, tag):
        if self.pretrained:
            self.pretrained_critic.load_checkpoint(tag)
            self.pretrained_actor.load_checkpoint(tag)
            self.pretrained_target_critic.load_checkpoint(tag)
            self.pretrained_target_actor.load_checkpoint(tag)
        else:
            self.critic.load_checkpoint(tag)
            self.actor.load_checkpoint(tag)
            self.target_critic.load_checkpoint(tag)
            self.target_actor.load_checkpoint(tag)

    def eval(self):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()

    def store_transition(self, state, actions, rewards, done, prev_pos):
        self.memory.store_transition(self.processFeature(state.prev_car_state_training, prev_pos), actions, rewards,
                                     self.processFeature(state.current_car_state_training, prev_pos), done)

    # override
    def choose_actions(self, obs, prev_pos, inference):
        self.actor.eval()
        obs = self.processFeature(obs, prev_pos)
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        with torch.no_grad():
            actions = self.actor.forward(obs).to(self.device)
        self.actor.train()

        # noice
        if not inference:
            actions = actions + torch.tensor(self.noise(), dtype=torch.float).to(self.device)

        return actions.cpu().detach().numpy()

    def processFeature(self, state, prev_pos):
        feature = []

        # distance between car and target
        feature.append(state.car_pos.x - state.final_target_pos.x)
        feature.append(state.car_pos.y - state.final_target_pos.y)

        # angle in radian between up(0, 1)vector and car to target
        rad = Utility.radFromUp([state.car_pos.x, state.car_pos.y], \
                                [state.final_target_pos.x, state.final_target_pos.y])
        feature.append(Utility.decomposeCosSin(rad))  # cos(radian), sin(radian) *2

        # car orientation(eular angles in radians)
        feature.append(Utility.decomposeCosSin(state.car_orientation))

        feature.append(state.car_vel.x)
        feature.append(state.car_vel.y)

        # car angular velocity in radians(eular angles in radians)
        feature.append(state.car_angular_vel)

        feature.append(state.wheel_angular_vel.left_back)
        feature.append(state.wheel_angular_vel.right_back)

        feature.append(state.action_wheel_angular_vel.left_back)
        feature.append(state.action_wheel_angular_vel.right_back)

        feature = Utility.flatten(feature)

        return feature
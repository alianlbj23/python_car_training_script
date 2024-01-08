import torch
import Utility
import math
# from CustomThread import CustomThread
import os
import json
from datetime import datetime
from Entity import State
import Entity


class Environment():
    def __init__(self, max_times_in_episode, max_times_in_game, end_distance=(1, 15), \
                 save_log=False, stop_target=True, target_fixed_sec=8, min_angle_diff=10):
        self.devie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pos = [0, 0]
        self.target_pos = None
        self.real_target = None
        self.episode_ctr = 0
        self.game_ctr = 0
        self.total_ctr = 0
        self.prev_pos = [0, 0]
        self.orientation = None
        self.prevOrientation = 0

        self.prev_time = datetime.now().second
        self.time = datetime.now().second

        self.max_times_in_episode = max_times_in_episode
        self.max_times_in_game = max_times_in_game

        self.save_log = save_log
        if save_log:
            self.log = {"obs": []}
            with open(os.path.join(os.path.dirname(__file__), "state_log_virtual_edge.txt"), "w") as f:
                json.dump(self.log, f)

        self.target_fixed_sec = target_fixed_sec
        self.stop_target = stop_target
        self.end_distance = end_distance
        self.epsilon = 0.0001
        self.min_angle_diff = min_angle_diff

        self.distance_out = False
        self.game_finished = False
        self.reach_goal = False

    def calculate_orientation_diff(self, car_orientation, target_orientation):
        diff = abs(target_orientation - car_orientation)
        if diff > 180:
            diff = 360 - diff
        reward = diff

        return reward

    def calculate_distance(self, car_pos, target_pos):
        distance = math.dist(car_pos, target_pos)
        return distance

    # def radToPositiveDeg(self, rad):
    #     # left +, right -, up 0, down 180 => clockwise: 0 - 359
    #     deg = rad / DEG2RAD
    #     if deg < 0:
    #         deg = -deg
    #     elif deg > 0:
    #         deg = 360 - deg

    #     return deg

    def save_log(self):
        f = open(os.path.join(os.path.dirname(__file__), "state_log_virtual_edge.txt"), 'w')
        log = json.dumps((self.log))
        f.seek(0)
        f.write(log)
        f.truncate()
        f.close()

    def restart_episode(self):
        if (self.save_log):
            self.save_log()

        self.episode_ctr = 0

        restart_game = False

        if (self.distance_out == True or self.reach_goal == True or self.game_finished == True):
            if self.distance_out == True:
                print("distance is out")
            if self.reach_goal == True:
                print("reaches goal")
            if self.game_finished == True:
                print("game is finished")
            restart_game = True

        if (self.stop_target == True):
            self.game_ctr = 0
            if (self.time < self.prev_time):
                self.prev_time -= 60
            if (self.time - self.prev_time) >= self.target_fixed_sec:
                self.prev_time = self.time
            restart_game = True

        return restart_game

    def restart_game(self, state: State):
        self.init = 1
        self.game_ctr = 0
        self.pos = [state.car_pos.x, state.car_pos.y]
        self.inital_pos = self.pos
        # self.pos = [0., 0.]
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]
        # self.target_pos = [self.pos[0] + obs['final target pos']['x'], self.pos[1] + obs['final target pos']['y']]

        self.trail_original_pos = [state.path_closest_pos.x, state.path_closest_pos.y]
        self.distance = self.calculate_distance(self.pos, self.target_pos)

    # override
    def check_termination(self, state):
        try:
            print(min(state.min_lidar))
            collision = min(state.min_lidar) < 0.1
        except:
            pass
        # print(state.min_lidar_direciton)
        self.turnover = (state.objectUpVector < 0)
        self.pos = [state.car_pos.x, state.car_pos.y]
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]
        # print("target position: ", self.target_pos)
        # print("car position: ", self.pos)

        distance = math.dist(self.pos, self.target_pos)

        self.reach_goal = ((abs(self.carOrientation - self.targetOrientation) <= 20) \
                           and distance <= self.end_distance[0])

        # self.reach_goal = (abs(self.carOrientation - self.targetOrientation) < 5) 
        print("distance", distance)
        self.distance_out = distance >= self.end_distance[1] or distance <= self.end_distance[0]
        try:
            self.game_finished = self.game_ctr >= self.max_times_in_game or collision or self.turnover

        except:
            pass
        if self.reach_goal:
            print("reach_goal!!!!!!!!!!!!!!!!!!!")
        # if self.episode_ctr >= self.max_times_in_episode:
        #     print("episode ctr >= {}".format(self.max_times_in_episode))
        if self.game_finished:
            print("game_ctr >= {}".format(self.max_times_in_game))
        if distance <= self.end_distance[0]:
            print("distance <= {}".format(self.end_distance[0]))
        if distance >= self.end_distance[1]:
            print("distance >= {}".format(self.end_distance[1]))

        done = self.reach_goal or self.distance_out \
               or self.episode_ctr >= self.max_times_in_episode \
               or self.game_finished
        return done, self.reach_goal

    def calculate_reward(self, state: Entity.State, new_state: Entity.State):
        reward = 0

        self.pos = [new_state.car_pos.x, new_state.car_pos.y]
        self.prev_pos = [state.car_pos.x, state.car_pos.y]

        self.carOrientation = Utility.rad2deg(new_state.car_orientation)
        prevCarOrientation = Utility.rad2deg(state.car_orientation)

        target_pos = [state.final_target_pos.x, state.final_target_pos.y]

        self.targetOrientation = Utility.rad2deg(Utility.radFromUp(self.pos, target_pos))

        ### distance to final target
        prevTargetDist = self.calculate_distance(self.prev_pos, target_pos)
        distanceToTarget = self.calculate_distance(self.pos, target_pos)

        distanceDiff = distanceToTarget - prevTargetDist

        # if distanceDiff > 5:
        #     reward -= distanceDiff*800
        # elif distanceDiff > 3:
        #     reward -= distanceDiff*300

        # elif distanceDiff < 3:
        #     reward += distanceDiff*200
        # elif distanceDiff < 2.5:
        #     reward += distanceDiff*400

        if distanceDiff > 0:
            distanceDiff *= 2
            distanceDiff *= 400
            reward -= distanceDiff
        elif distanceDiff < 0:
            reward += 100 * -(distanceDiff)

        ### angle gap to target
        prevTargetOrientation = Utility.rad2deg(Utility.radFromUp(self.prev_pos, target_pos))

        prevAngleGapToTarget = self.calculate_orientation_diff(prevCarOrientation, prevTargetOrientation)
        TargetOrientation = Utility.rad2deg(Utility.radFromUp(self.pos, target_pos))
        angleGapToTarget = self.calculate_orientation_diff(self.carOrientation, TargetOrientation)
        targetAngleDiff = angleGapToTarget - prevAngleGapToTarget
        # print('targetAngle: ', targetAngleDiff)
        targetAngleDiff *= 4
        if targetAngleDiff > 0:
            targetAngleDiff *= 4
        reward += -targetAngleDiff

        if ((self.game_ctr - 1) // 5) == ((self.game_ctr - 2) // 5):
            if (state.action_wheel_angular_vel.left_back < 0 and new_state.action_wheel_angular_vel.left_back > 0) or \
                    (state.action_wheel_angular_vel.left_back > 0 and new_state.action_wheel_angular_vel.left_back < 0):
                self.stucked_count += 1
                # print("count ", self.stucked_count, self.game_ctr)
        else:
            self.stucked_count = 0

        if self.stucked_count > 1:
            reward += -200 * self.stucked_count
            print("count ", self.stucked_count)

        # print("total ", reward)
        # print("----------------------")

        return reward
    def step(self, state, new_state):
        self.episode_ctr += 1
        self.game_ctr += 1
        self.total_ctr += 1
        # print("new_state: ", new_state.final_target_pos)
        reward = self.calculate_reward(state, new_state)

        done, reachGoal = self.check_termination(state)  # self.trailOrientation

        if reachGoal:
            reward += 400

        info = {'prev pos': []}
        info['prev pos'] = self.prev_pos
        info['trail original pos'] = [0, 0]

        return reward, done, info

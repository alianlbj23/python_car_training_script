import torch
import Utility
import math
import os
import json
from datetime import datetime
from config import ENVIRONMENT

class Environment():
    def __init__(self, save_log=False, min_angle_diff=10):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

        self.max_times_in_episode = ENVIRONMENT["MAX_TIMES_IN_EPISODE"]
        self.max_times_in_game = ENVIRONMENT["MAX_TIMES_IN_GAME"]

        self.save_log = save_log
        if save_log:
            self.log = {"obs": []}
            with open(os.path.join(os.path.dirname(__file__), "state_log_virtual_edge.txt"), "w") as f:
                json.dump(self.log, f)

        self.target_fixed_sec = ENVIRONMENT["TARGET_FIXED_SEC"]
        self.stop_target = ENVIRONMENT["STOP_TARGET"]
        self.end_distance = ENVIRONMENT["END_DISTANCE"]
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

        is_restart_game = False

        if (self.distance_out == True or self.reach_goal == True or self.game_finished == True):
            if self.distance_out == True:
                print("distance is out")
            if self.reach_goal == True:
                print("reaches goal")
            if self.game_finished == True:
                print("game is finished")
            is_restart_game = True

        if (self.stop_target == True):
            self.game_ctr = 0
            if (self.time < self.prev_time):
                self.prev_time -= 60
            if (self.time - self.prev_time) >= self.target_fixed_sec:
                self.prev_time = self.time
            is_restart_game = True
        return is_restart_game

    def restart_game(self, state):
        self.init = 1
        self.game_ctr = 0
        self.pos = [state.car_pos.x, state.car_pos.y]
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]
        self.distance = self.calculate_distance(self.pos, self.target_pos)

    # override
    def check_termination(self, state):
        try:
            # print(min(state.min_lidar))
            collision = min(state.min_lidar) < 0.1
        except:
            pass

        self.pos = [state.car_pos.x, state.car_pos.y]
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]
        distance = math.dist(self.pos, self.target_pos)

        self.reach_goal = ((abs(self.carOrientation - self.targetOrientation) <= 20) \
                           and distance <= self.end_distance[0])

        # print("distance", distance)
        self.distance_out = distance >= self.end_distance[1] or distance <= self.end_distance[0]
        try:
            self.game_finished = self.game_ctr >= self.max_times_in_game or collision

        except:
            pass
        if self.reach_goal:
            print("reach_goal!!!!!!!!!!!!!!!!!!!")
        if self.game_finished:
            if collision == True:
                print("collide with walls")
            else:
                print("game_ctr >= {}".format(self.max_times_in_game))
        if distance <= self.end_distance[0]:
            print("distance <= {}".format(self.end_distance[0]))
        if distance >= self.end_distance[1]:
            print("distance >= {}".format(self.end_distance[1]))

        done = self.reach_goal or self.distance_out \
               or self.episode_ctr >= self.max_times_in_episode \
               or self.game_finished
        return done, self.reach_goal

    def calculate_reward(self, state, new_state):
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

        return reward
    
    def step(self, state, new_state):
        self.episode_ctr += 1
        self.game_ctr += 1
        self.total_ctr += 1
        reward = self.calculate_reward(state, new_state)

        done, reachGoal = self.check_termination(state)  # self.trailOrientation

        if reachGoal:
            reward += 400

        return reward, done
    

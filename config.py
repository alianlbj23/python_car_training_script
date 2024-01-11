DEG2RAD = 0.01745329251
# ----------------environment--------------- #
# users can change: max_times_in_episode, max_times_in_game, end_distance
max_times_in_episode=30
max_times_in_game=210
# (reach goal, distance out)
end_distance=(0.2, 7)
stop_target=False
target_fixed_sec=12
# ----------------environment--------------- #

# ----------- paths ------------------#
# users should change the path to match yourself's one
load_path = './Model/DDPG/0809_car/model'
save_path = './Model/DDPG/0809_car/model'
plot_path = './Model/DDPG/0809_car'
log_path = './Model/DDPG/0809_car/log'
# ------------paths -------------------#

# -------------------------Agent -------------------#
q_lr=0.001
pi_lr=0.001
gamma=0.99
rho=0.005
pretrained=False
new_input_dims=17
input_dims=13
n_actions=2
batch_size=100
layer1_size=400
layer2_size=300
load_step = 0
# ------------------------ Agent -------------------#

# --------------UnityAdaptor -------------#
action_range=600
steering_angle_range=20
# --------------UnityAdaptor -------------#
epoch = 5000
unityState = ''
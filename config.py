DEG2RAD = 0.01745329251
# ----------------environment--------------- #

ENVIRONMENT = {
    "MAX_TIMES_IN_EPISODE": 30,
    "MAX_TIMES_IN_GAME": 210,
    "END_DISTANCE": (0.2, 7), 
    "STOP_TARGET":False,
    "TARGET_FIXED_SEC":12
}

# ----------------environment--------------- #

# ----------- paths ------------------#
# users should change the path to match yourself's one
PATH = {
    "LOAD_PATH": './Model/DDPG/0809_car/model',
    "SAVE_PATH": './Model/DDPG/0809_car/model',
    "PLOT_PATH": './Model/DDPG/0809_car',
    "LOG_PATH":'./Model/DDPG/0809_car/log'
}
# ------------paths -------------------#

# -------------------------Agent -------------------#
AGENT = {
    "q_lr": 0.001,
    "pi_lr": 0.001,
    "gamma" :0.99,
    "rho" :0.005,
    "pretrained": False,
    "new_input_dims": 17,
    "input_dims": 13,
    "n_actions": 2,
    "batch_size": 100,
    "layer1_size": 400,
    "layer2_size": 300,
    "load_step": 400
}
# ------------------------ Agent -------------------#

UNITY = {
    "action_range": 600,
    "steering_angle_range":20
}

PARAMETER = {
    "epoch": 5000
}

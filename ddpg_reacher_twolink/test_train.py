from agent import Controller

# Load controller
# E10000: (S?/R0/A?0, S1/R1/A80 ,S2/R2/A90, S1/R3/A90)
# E5000: (S4/R0/A80, S4/R1/A70 ,S4/R2/A60, S4/R3/A10)

SEED = 4
REWARD_TYPE = 2
controller = Controller(rand_seed = SEED, rew_type = REWARD_TYPE)

# Training
NUM_EPISODES = 10000
MAX_TIME_STEPS = 150
MODEL_SAVE_NAME = 'reacher'

# controller.train(num_episodes = NUM_EPISODES, max_timesteps = MAX_TIME_STEPS, model_name = MODEL_SAVE_NAME)

# Testing
NUM_TESTS = 10
MODEL_LOAD_NAME = 'reacher_' + str(NUM_EPISODES) + '_' + str(REWARD_TYPE)

controller.test(num_test = NUM_TESTS, max_timesteps = MAX_TIME_STEPS, model_name = MODEL_LOAD_NAME)
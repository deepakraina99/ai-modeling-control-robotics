# AI-based Modeling and Control of Robotic Systems

This ProjectSubmitted to Advances in Robotics (AIR) 2021 conference.
This repository helps in the  installation,  running of the python codes for the AI based modeling and control of a 2-link robotic manipulator.
The user can work and implement on the Neural Network based modeling and Reinforcement learning based control of robot arm in OpenAI gym. This [repository](https://github.com/deepakraina99/ai-modeling-control-robotics) contains a sub folder for modeling and two sub folders for control. The control task is the target reaching of the robot using two specific algorithms like DDPG (Deep Deterministic Gradient Policy)and PPO (Proximal Policy Gradient). The modeling task is the NN based modeling of the robot to get the desired torque values for the input trajectory(cycloidal). 
***

## AI based modeling of the 2-link arm
### PyTorch Installation (Basic Installation)
PyTorch is a GPU and CPU optimised tensor library for deep learning. Refer [here](https://pytorch.org/docs/stable/index.html)

```bash
pip install torch
```
It is recommeneded to use Python 3.6 (although other Python 3.x versions may still work). You can either download [Python 3.6 here](https://www.python.org/downloads/), or use [pyenv](https://github.com/pyenv/pyenv) to install Python 3.6 in a local directory, e.g. `pyenv install 3.6.5; pyenv local 3.6.5`
### Getting started
**1.** _File _description_: The [folder](https://github.com/deepakraina99/ai-modeling-control-robotics/tree/main/2link-model-learning) contains three python files  and one csv file exists. The Python files are comprised of the main controller, analytical model and the train_test file. The csv file is the dataset needed to train the model. This is comprised of the trajectory inputs needed for the model i.e. the position, velocity and acceleration which is fed to the Neural Network model to generate the desired torque values for the joint1 and joint 2 of the robot arm.

**2.** _Training_the_NN_: To run and see the output, use the train_test.py file. The class has input called Epoch which needs to be mentioned. This file is used to set the epoch to train and run the main project. The output observed is the Epoch, Iter, Training loss, Test loss.

**3.** _Testing_the_NN_: Use the same train_test.py file to test the model after the successful completion of training. For testing the model use test_traj_class.csv dataset. The output observed is the computation time for analytical model and Network model (These classes are defined in the models.py file ).
#### Running the model
Use the train_test.py file to run the NN model with specified number of epochs, Controller.train(). For testing the model, use the same file and the function controller.test() to test the model.
An example for testing the model is given below:

```python
# Robot controller
controller = RobotController()
EPOCHS = 10
MODEL_FILE_LOC = 'models/trained_nn_model_' + str(EPOCHS)

## Testing
controller.test(model_fileloc = MODEL_FILE_LOC, num_test=1)
```
***
## AI based control
OpenAI Gym is a toolkit for constructing and evaluating reinforcement learning algorithms. The gym library is a set of research problems (environments) that we can use to fine-tune our reinforcement learning algorithms. These environments share a gui, making it possible to write general algorithms. Refer [here](https://gym.openai.com/) for further details. 
### OpenAI Gym Installation(Basic Installation)

```bash
pip install gym
```
The two common algorithms used in target reaching problem in RL based Control are:- DDPG (Deep Deterministic Gradient 
Policy)and PPO (Proximal Policy Gradient). 
*
### DDPG based Target reaching of 2-link robot
**1.** _File_description_: First, the [folder](https://github.com/deepakraina99/ai-modeling-control-robotics/tree/main/ddpg_reacher_twolink) contains the python files agent, model, twolinkarm_env, plot_reward, test_train. The main environment is represented in twolinkarm_env i.e. the two link arm and the ball. The model file contains the actor-critic network model. The agent.py file contains the agent, Ornstein-Uhlenbeck process, controller, replay buffer classes.

**2.** _Training_the_Model_: Use the file test_train.py for training the model. The controller class takes inputs such as seed and reward type which is given as integer values. The controller.train() takes inputs like number of episodes, maximum time steps. The output is the episode, length, reward, average reward.

**3.** _Testing_the_model_: Use the same file to test the model. The output is the 2-link arm reaching the target ball. The controller.test() takes inputs like the number of episodes, maximum timesteps and number of tests to be run.

An example for testing the model using DDPG algorithm is given below:
```python
SEED, REWARD_TYPE =4,2
controller = Controller(rand_seed = SEED, rew_type = REWARD_TYPE)
# Testing
NUM_TESTS = 10
MODEL_LOAD_NAME = 'reacher_' + str(NUM_EPISODES) + '_' + str(REWARD_TYPE)

controller.test(num_test = NUM_TESTS, max_timesteps = MAX_TIME_STEPS, model_name = MODEL_LOAD_NAME)
```
***
### PPO based target reaching
**1.** _File_description_: The [folder](https://github.com/deepakraina99/ai-modeling-control-robotics/tree/main/ppo_reacher_twolink) contains the python files  model, twolinkarm_env, test_reacher, train. The main environment is represented in twolinkarm_env i.e. the two link arm and the ball. The model file contains the actor-critic network model. 

**2.** _Training_the_Model_: Use the file train.py for training the model. 

**3.** _Testing_the_model_: Use the file test_reacher.py for the visualization of target reaching by robot using PPO methodology. The output is the robot reaching the target ball in each tests.  The output is the length, reward, average reward for the specified number of episodes.

For testing the model,
type the command:
```bash
python test_reacher.py
```

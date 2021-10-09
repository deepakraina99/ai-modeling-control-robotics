# AI-based Modeling and Control of Robotic Systems

This repository provides the python codes for the AI based modeling and control of a 2-link robotic manipulator.
The user can develop the Deep Neural Network (DNN) based inverse dynamics model and Reinforcement Learning (RL) based target reaching controller of a robotic arm. 
***

## AI based modeling of the 2-link arm
The modeling task is the DNN based modeling of the 2-link robotic arm to estimate the desired torque values for the given input joint trajectory.
### Installation
* Install PyTorch, which is a GPU and CPU optimised tensor library for deep learning. Refer [here](https://pytorch.org/docs/stable/index.html)

```
pip install torch
```
* Install Python 3.6 (although other Python 3.x versions may still work). You can either download [Python 3.6 here](https://www.python.org/downloads/), or use [pyenv](https://github.com/pyenv/pyenv) to install Python 3.6 in a local directory.
```
pip install python=3.6
```
### Getting started
**1. Training the NN**: To train the DNN for model learning, use the *train_test.py* file. The class has input called Epoch which needs to be mentioned.  The training and test loss is observed at the end of training.

**3. Testing the NN**: Use the same *train_test.py* file to test the model after the successful completion of training. The output observed is the computation time for analytical model and DNN model.

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
The two state-of-the-art RL algorithms i.e  DDPG (Deep Deterministic Gradient Policy) and PPO (Proximal Policy Gradient) are used in learning a target reaching controller of robotic arm.
### Installation
* Install OpenAI gym
```
pip install gym
```
### Getting started
**1. Training the Model**: Use the file *test_train.py* for training the model. The controller class takes seed and reward type as input. The controller.train() class function takes inputs like number of episodes and maximum time steps.

**2. Testing the model**: Use the same file to test the model. The output is the 2-link arm reaching the target ball. The controller.test() takes inputs as the maximum timesteps and number of tests to be run.

An example for testing the model using DDPG algorithm is given below:
```python
SEED, REWARD_TYPE =4,2
controller = Controller(rand_seed = SEED, rew_type = REWARD_TYPE)
# Testing
NUM_TESTS = 10
MODEL_LOAD_NAME = 'reacher_' + str(NUM_EPISODES) + '_' + str(REWARD_TYPE)
controller.test(num_test = NUM_TESTS, max_timesteps = MAX_TIME_STEPS, model_name = MODEL_LOAD_NAME)
```

# ############################################
# Two link robot arm simulation environment
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.4
# ############################################

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

#Environment class
class TwoLinkArmEnv(gym.Env):
    metadata = {'render.modes' : ['human', 'rgb_array'],
                'video.frames_per_second' : 20
               }
    
    def __init__(self, reward_type = 0):
        self.target = np.array([0.0, 0.0]) #[x,y]
        self.state = np.array([0.0]*4) #[theta1, theta2, thetadot1, thetadot2]
        self.obs_state = None

        self.viewer = None 
        self.arm_length = 1.0
        self.target_radius = 0.1
        self.max_speed = np.pi #rad/sec
        self.dt = 0.05 #time step
        self.max_time = 5
        self.step_n = 0
        self.reward_ver = reward_type
        
        # Target max-min limit
        self.target_high = [2, 2]
        self.target_low = [-1*i for i in self.target_high]
        # Joint max-min limit
        self.joint_high = [np.pi, np.pi]
        self.joint_low = [-1*i for i in self.joint_high]

        if self.reward_ver == 3:
            self.joint_low = [0.0, 0.0] #joint min limit 
            self.target_low = [-2, 0] #target min limit 

        #attributes
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32)
        self.observation_high = np.array(self.target_high*3 + [self.max_speed]*2)
        self.observation_space = spaces.Box(low=-self.observation_high, high=self.observation_high, dtype=np.float32)
        self.seed() #initialize a seed
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self,action):
        self.step_n += 1
        theta1, theta2, thetadot1, thetadot2 = self.state
        x_target, y_target = self.target
        action = np.clip(action, a_min=-self.max_speed, a_max=self.max_speed)
        self.last_action = action
        #update state
        new_theta1 = theta1 + action[0] * self.dt
        new_theta2 = theta2 + action[1] * self.dt
        # new_theta1 = np.clip(new_theta1, self.joint_low[0], self.joint_high[0])
        # new_theta2 = np.clip(new_theta2, self.joint_low[1], self.joint_high[1])
        new_state = np.array([new_theta1, new_theta2, action[0], action[1]])
        self.state = new_state
        # print('state: ', new_state)
        #get an observation of state
        obs_state = self.get_obs()
        
        #check to terminate
        reached = False
        terminated = False
        distance_to_target = self.euclidian_distance(self.target, [obs_state[4], obs_state[5]])
        # print('distance-to-target %.2f:' % (distance_to_target))
        if distance_to_target < self.target_radius:
            print('Reached Goal !')
            reached = True
        #calculate the reward
        reward_dist = -distance_to_target
        reward_ctrl = -np.square(action).sum()
        # Reward: 0
        if self.reward_ver == 0:
            reward = reward_dist
        # Reward: 1
        if self.reward_ver == 1:
            reward = reward_dist + reward_ctrl
        # Reward: 2 or 3
        if self.reward_ver == 2 or self.reward_ver == 3:
            if not reached:
                reward = - 1.0 + reward_ctrl
            else:
                reward = 1000
        return obs_state, reward, (reached or terminated), {}

    def reset(self):
        self.last_action = None
        self.step_n = 0
        #selecting random initial configuration
        state_high = np.array([self.joint_high[0], self.joint_high[1], 0.0, 0.0], dtype=np.float32)
        state_low = np.array([self.joint_low[0], self.joint_low[1], -0.0, -0.0], dtype=np.float32)
        self.state = self.np_random.uniform(low=state_low, high=state_high)
        #selecting random target location
        target_high = np.array(self.target_high)
        target_low = np.array(self.target_low)
        self.target = self.np_random.uniform(low=target_low, high=target_high)

        return self.get_obs()
    
    def get_obs(self):
        # print('get_obs')
        theta1 , theta2 , thetadot1 , thetadot2 = self.state
        x1 = self.arm_length * np.cos(theta1)
        y1 = self.arm_length * np.sin(theta1)
        x2 = x1 + self.arm_length * np.cos(theta1 + theta2)
        y2 = y1 + self.arm_length * np.sin(theta1 + theta2)
        self.obs_state = np.append(self.target, np.array([x1, y1, x2, y2, thetadot1, thetadot2], dtype=np.float32))
        # print(self.obs_state)
        return self.obs_state

    def render(self, mode='human',close=False):
        #close the viewer when needed
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        #first time render is called on a new viewer, it has to be initialized
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            
            #initialize viewer
            self.viewer = rendering.Viewer(700,700)
            self.viewer.set_bounds(-4.2, 4.2, -4.2, 4.2)
            
            #create target circle
            target = rendering.make_circle(self.target_radius)
            target.set_color(1,0,0)
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            self.viewer.add_geom(target)
            
            #create first arm segment
            link1 = rendering.make_capsule(self.arm_length,0.2)
            link1.set_color(0.5, 0.5, 0.5)
            self.link1_transform = rendering.Transform()
            link1.add_attr(self.link1_transform)
            self.viewer.add_geom(link1)
            
            #create first joint
            joint1 = rendering.make_circle(0.1)
            joint1.set_color(0, 0, 0)
            # joint1.add_attr(self.link1_transform)
            self.viewer.add_geom(joint1)

            #create second arm segment
            link2 = rendering.make_capsule(self.arm_length,0.2)
            link2.set_color(0.65, 0.65, 0.65)
            self.link2_transform = rendering.Transform()
            link2.add_attr(self.link2_transform)
            self.viewer.add_geom(link2)
            
            #create second joint
            joint2 = rendering.make_circle(0.1)
            joint2.set_color(0, 0, 1)
            joint2.add_attr(self.link2_transform)
            self.viewer.add_geom(joint2)
            
            #create end-effector circle
            end_effector = rendering.make_circle(0.1)
            end_effector.set_color(0,1,0)
            self.end_effector_transform = rendering.Transform()
            end_effector.add_attr(self.end_effector_transform)
            self.viewer.add_geom(end_effector)
            
        obs_state = self.get_obs()
        
        #set the viewer in the object according to current state
        self.link1_transform.set_rotation(self.state[0])
        self.link2_transform.set_translation(obs_state[2],obs_state[3])
        self.link2_transform.set_rotation(self.state[0] + self.state[1])
        self.end_effector_transform.set_translation(obs_state[4],obs_state[5])
        
        # if self.last_action is None:
        self.target_transform.set_translation(self.target[0],self.target[1])
            
        return self.viewer.render(return_rgb_array = mode == 'rgb_array')
    
    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def close(self):
        # print('close')
        if self.viewer:
            self.viewer.close()
            self.viewer = None
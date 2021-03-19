import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# Analytical model of 2 link robot arm
class AnalyticalModel():
    def __init__(self, tf = 50, step = 0.1, num_traj = 1, record_file = 'logs/traj.csv'):
        self.m1, self.m2 = 1, 1 # mass of links
        self.l1, self.l2 = 1, 1 # length of links
        self.lc1, self.lc2 = self.l1*0.5, self.l2*0.5 # COM of links
        self.I1z, self.I2z = (self.m1*self.l1*self.l1)/3, (self.m2*self.l2*self.l2)/3; #Inertia of links
        self.g = -9.81 # acceleration due to gravity
        self.tf = tf # toral time of trajectory
        self.max_angle = 90 # maximum angle
        self.thi = np.random.rand(2,1)*np.deg2rad(self.max_angle) # initial joint angles
        self.thf = np.random.rand(2,1)*np.deg2rad(self.max_angle) # final joint angles
        self.tstep = step # time step
        self.record_file = record_file

    # Function to get trajectory [th, dth, ddth]: cycloidal trajectory
    def getTrajectory(self, t, tf, thi, thf):
        th=thi+((thf-thi)/tf)*(t-(tf/(2*np.pi))*np.sin((2*np.pi/tf)*t))
        dth=((thf-thi)/tf)*(1-np.cos((2*np.pi/tf)*t))
        ddth=(2*np.pi*(thf-thi)/(tf*tf))*np.sin((2*np.pi/tf)*t)
        return th, dth, ddth

    # Function to get torque for given trajectory
    def getTorque(self, th, dth, ddth):
        th1, dth1, th2, dth2 = th[0], dth[0], th[1], dth[1]

        # State equations
        # Mass matrix [M]
        m11 = self.m1*self.lc1*self.lc1 + self.m2*(self.l1*self.l1 + self.lc2*self.lc2 + 2*self.l1*self.lc2*np.cos(th2))+ self.I1z + self.I2z
        m12 = self.m2*(self.lc2*self.lc2 + self.l1*self.lc2*np.cos(th2)) + self.I2z
        m21 = m12
        m22 = self.m2*self.lc2*self.lc2 + self.I2z
        M = np.array([[m11, m12],
                      [m21, m22]], dtype=np.float32)

        # Coupling Vector [C]
        c1 = -self.m2*self.l1*self.l2*np.sin(th2)*dth2*(dth1+(dth2/2))
        c2 = self.m2*self.l1*self.lc2*(dth1**2)*np.sin(th2)
        C = np.array([c1,
                      c2], dtype=np.float32)

        # Gravity Vector [G]
        g1 = self.m1*self.lc1*self.g*np.sin(th1) + self.m2*self.l1*self.g*np.sin(th1) + self.m2*self.lc2*self.g*np.sin(th1+th2)
        g2 = self.m2*self.lc2*self.g*np.sin(th1+th2)
        G = np.array([g1,
                      g2], dtype=np.float32)

        # Torque vector [tau]
        tau = np.matmul(M,ddth) + C + G
        return tau

    # Function to solve analytical model
    def run(self, num_traj, test=False):
        for i in range(num_traj):
            self.thi = np.random.rand(2,1)*np.deg2rad(self.max_angle) # initial joint angles
            self.thf = np.random.rand(2,1)*np.deg2rad(self.max_angle) # final joint angles
            # self.thi = np.array([[0.59207397],[0.86151783]])
            # self.thf = np.array([[0.92306365],[0.23544595]])
            ts, ths, dths, ddths, taus = [], [], [], [], []
            for t in np.arange(0,self.tf,self.tstep):
                th, dth, ddth = self.getTrajectory(t, self.tf, self.thi, self.thf)
                tau = self.getTorque(th, dth, ddth)
                ts.append(t)
                ths.append(th)
                dths.append(dth)
                ddths.append(ddth)
                taus.append(tau)
            ts = np.asarray(ts).squeeze()
            ts = np.reshape(ts, (ts.shape[0],1))
            ths = np.asarray(ths).squeeze()
            dths = np.asarray(dths).squeeze()
            ddths = np.asarray(ddths).squeeze()
            taus = np.asarray(taus).squeeze()
            traj_tau = np.hstack((ts,ths,dths,ddths,taus))
            if (test): 
                return traj_tau
            else: 
                self.writeTraj(self.record_file, traj_tau)
                # self.plotTraj(traj_tau)

    # Function to write trajectory to a .csv file
    def writeTraj(self,filename,arr):
        pd.DataFrame(arr).to_csv(filename, mode='a',header=None, index=None)

    # Function to plot trajectory and torques
    def plotTraj(self,traj):
        fig = plt.figure(figsize=(16,3))
        t = traj[:,0]
        plt.subplot(1,4,1)
        plt.plot(t,traj[:,1],label='$\\theta_1$')
        plt.plot(t,traj[:,2],label='$\\theta_2$')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Joint angle (rad)')
        
        plt.subplot(1,4,2)
        plt.plot(t,traj[:,3],label='$\dot{\\theta}_1$')
        plt.plot(t,traj[:,4],label='$\dot{\\theta}_2$')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Joint velocity (rad/s)')

        plt.subplot(1,4,3)
        plt.plot(t,traj[:,5],label='$\ddot{\\theta}_1$')
        plt.plot(t,traj[:,6],label='$\ddot{\\theta}_2$')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Joint acceleration ($rad/s^2$)')

        plt.subplot(1,4,4)
        plt.plot(traj[:,0],traj[:,7],label='$\\tau_1$')
        plt.plot(traj[:,0],traj[:,8],label='$\\tau_2$')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N-m)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('logs/model_traj.pdf')
        D
# Neural network model of 2 link arm
class NetworkModel(nn.Module):
    def __init__(self,num_input,num_output, seed ,hidden_layer1=32, hidden_layer2=64):
        super(NetworkModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.layer1 = nn.Sequential(nn.Linear(num_input, hidden_layer1), nn.ReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(hidden_layer1, hidden_layer2), nn.ReLU(), nn.Dropout(0.2))
        self.output = nn.Linear(hidden_layer2, num_output)
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.output(x)
        return x

# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    am = AnalyticalModel()
    am.run(num_traj=1)
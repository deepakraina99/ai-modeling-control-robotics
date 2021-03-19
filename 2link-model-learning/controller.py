import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from models import AnalyticalModel, NetworkModel

class RobotController():
    def __init__(self, dof=2, seed=5, datasetFile='logs/traj.csv', recordData = False, epochs=500):
        self.dof = dof
        self.input_dim = self.dof*3
        self.output_dim = self.dof
        self.seed = seed
        
        # Neural netwok based model
        self.nn_model = NetworkModel(self.input_dim, self.output_dim, self.seed)
        self.criterion = nn.MSELoss()
        self.opimizer = optim.SGD(self.nn_model.parameters(), lr=0.001)
        # Analytical model
        self.anal_model = AnalyticalModel()
        self.dataset_file = datasetFile
        self.record_data = recordData
        
    # Function for loading dataset
    def datasetLoader(self, filename):
        dataset = pd.read_csv(filename, header=None).values
        print(len(dataset))

        train_size = int(0.80*len(dataset))
        val_size = int(0.20*len(dataset))
        print('train_size: ', train_size, 'val_size: ', val_size)
        train_data,val_data= random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(dataset=train_data, batch_size=500, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_data, batch_size=500, shuffle=True, num_workers=0)
        return train_loader, val_loader

    # Function for training the network model
    def train(self, epochs):
        if (self.record_data):
            self.anal_model.run(num_traj=500)
        train_loader, val_loader = self.datasetLoader(self.dataset_file)
        print('Training started')
        loss_train, loss_test = [], []
        start_time = time.time()
        for epoch in range(epochs):
            loss_temp = []
            for mini_batch_num, data in enumerate(train_loader):
                data = data.type(torch.FloatTensor)
                inputs, true_torques = data[:,1:7], data[:,7:9]
                self.nn_model.train()
                pred_torques = self.nn_model(inputs)
                loss = self.criterion(pred_torques, true_torques)
                self.opimizer.zero_grad()
                loss.backward()
                loss_temp.append(loss.item())
                self.opimizer.step()

                if mini_batch_num % 50 == 0:
                    print('Epoch {}/{}; Iter {}/{}; Loss: {:.4f}'.format(epoch+1,epochs,mini_batch_num+1,len(train_loader),loss.item()))
            
            # validation of model
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    data = data.type(torch.FloatTensor)
                    self.nn_model.eval()
                    inputs, true_torques = data[:,1:7], data[:,7:9]
                    pred_torques = self.nn_model(inputs)
                    loss_test_temp = self.criterion(pred_torques, true_torques)
                
                print('Epoch {}/{}; Iter {}/{}; Training Loss: {:.4f}, Test Loss: {}'.format(epoch+1,epochs,mini_batch_num+1,len(train_loader),                                                                                    round(np.mean(np.array(loss_temp)),3), round(loss_test_temp.item(),3)))
            if epoch >= 1:
                loss_train.append(np.mean(np.array(loss_temp)))
                loss_test.append(loss_test_temp.item())
            if epoch % 100 == 0:
                fname = 'models/trained_nn_model_' + str(epochs)   
                self.saveModel(filename=fname)

        print('Training End !')
        print('Time taken for training = {}'.format(time.time()-start_time))
        self.plot_loss(loss_train, loss_test)

    # Function for plotting loss values
    def plot_loss(self, train_loss, test_loss):
        plt.plot(train_loss, c="red", label="Training Loss")
        plt.plot(test_loss, c="green", label="Validation Loss")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig('logs/train_test_loss.pdf')
        plt.show()

    # Function for saving the model
    def saveModel(self, filename):
        torch.save(self.nn_model.state_dict(), filename)
        print('Network model saved')

    # Function for loading the model
    def loadModel(self, filename):
        self.nn_model.load_state_dict(torch.load(filename, map_location = torch.device('cpu')))
        print('Network model loaded')

    # Function for testing the model
    def test(self, model_fileloc='models/trained_nn_model_500', num_test=5):
        print('Testing the model')
        st = time.time()
        traj_tau = self.anal_model.run(num_traj=1, test=True)
        print('Computation time: Analytic Model : {}'.format(time.time()-st))
        pd.DataFrame(traj_tau).to_csv('test_traj_class.csv', mode='a',header=None, index=None)

        # Solving trained model
        joint_traj = traj_tau[:,1:7]
        joint_traj = Variable(torch.from_numpy(joint_traj))
        st = time.time()
        self.loadModel(model_fileloc)
        self.nn_model.eval()
        pred_torques = self.nn_model(joint_traj.float())
        print('Computation time: Network Model : {}'.format(time.time()-st))

        pred_torques = pred_torques.detach().numpy()

        self.plot_comparison(traj_tau, pred_torques)

    def plot_comparison(self, traj, pred_torques):
        # fig = plt.figure(figsize=(12,6))
        t = traj[:,0]
        # plt.subplot(2,2,1)
        # plt.plot(t,traj[:,1],label='$\\theta_1$')
        # plt.plot(t,traj[:,2],label='$\\theta_2$')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint angle (rad)')
        
        # plt.subplot(2,2,2)
        # plt.plot(t,traj[:,3],label='$\dot{\\theta}_1$')
        # plt.plot(t,traj[:,4],label='$\dot{\\theta}_2$')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint velocity (rad/s)')
        # plt.subplot(2,2,3)
        # plt.plot(t,traj[:,5],label='$\ddot{\\theta}_1$')
        # plt.plot(t,traj[:,6],label='$\ddot{\\theta}_2$')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joint acceleration ($rad/s^2$)')
        # plt.subplot(2,2,4)
        plt.plot(t, traj[:,7], 'k', label='Actual')
        plt.plot(t, pred_torques[:,0], 'b--', label='Predicted')
        plt.plot(t, traj[:,8], 'k')
        plt.plot(t,pred_torques[:,1], 'b--')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N.m)")
        timestr = time.strftime("%H%M%S")
        plt.savefig('logs/model_compare_'+timestr+'.pdf')
        # plt.savefig('logs/model_comparison_'+timestr+'.pdf')
        plt.show()

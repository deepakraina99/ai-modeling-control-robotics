import pandas
import matplotlib.pyplot as plt
import numpy as np

rewards_file = 'logs/rewards.csv'
# Read data
R = pandas.read_csv(rewards_file, header=None).values
num_eps = len(R)
mean_over = 100
s=0
R_mean = []
for e in range(mean_over,num_eps):
    R_list = R[s:e]
    R_mean_item = sum(R_list)/mean_over
    R_mean.append(R_mean_item)
    s+=1

R_mean = np.asarray(R_mean)
fig = plt.figure()
# plt.plot(np.arange(len(R)), R)
plt.plot(np.arange(len(R_mean)), R_mean)

plt.ylabel('Average reward')
plt.xlabel('# Episode')
plt.savefig('logs/avg_reward.pdf')
plt.show()

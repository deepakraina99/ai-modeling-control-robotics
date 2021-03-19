import torch
import torch.nn as nn

class ActorCritic():
    def __init__(self, state_dim, action_dim, action_std, fc1_units = 64, fc2_units = 32):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc1_units),
            nn.Tanh(),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Linear(fc2_units, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, fc1_units),
            nn.Tanh(),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Linear(fc2_units, 1)
        )

        self.actor_var = torch.full((action_dim,), action_std*action_std)

    def forward(self, x):
        raise NotImplementedError
        
    

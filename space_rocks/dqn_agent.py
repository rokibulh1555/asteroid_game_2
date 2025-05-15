import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._calculate_conv_output(h, w), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _calculate_conv_output(self, h, w):
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        h = conv2d_size_out(h, 8, 4)
        h = conv2d_size_out(h, 4, 2)
        h = conv2d_size_out(h, 3, 1)
        w = conv2d_size_out(w, 8, 4)
        w = conv2d_size_out(w, 4, 2)
        w = conv2d_size_out(w, 3, 1)
        return h * w * 64

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_shape, num_actions, device):
        self.device = device
        self.num_actions = num_actions
        self.policy_net = DQNetwork(input_shape, num_actions).to(device)
        self.target_net = DQNetwork(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_steps = 1000
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

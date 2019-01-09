import numpy as np
import random
import pdb
from collections import namedtuple, deque
from recordclass import recordclass

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate
UPDATE_EVERY = 4            # how often to update the network
PRIORITISATION = 1 # 0.6        # 0 for uniform sampling of replay buffer
PRIORITY_OFFSET = 0 # int(1e-2) # Small number to ensure we visit edge-case transitions
IMPORTANCE_SAMPLING = 1 # 0.4   # 0 for no importance sample weighting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (e, s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        es, states, actions, rewards, next_states, dones, weights = experiences

        # Compute estimated outputs (action values) from the current network
        # Use gather to only select output neuron for the chosen action
        q_estimates = self.qnetwork_local(states).gather(1, actions)

        # Compute target outputs (action values) from the experience batch
        # Use max to only select output neuron for the optimal next action
        # Bootstrap from the special target network to promote stable learning
        next_action_values = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_action_values * (1 - dones))

        # Compute importance sampling ratio
        probabilities = weights / self.memory.weight_sum()
        is_ratios = ((1/len(self.memory)) * (1/probabilities)) ** IMPORTANCE_SAMPLING

        # Compute TD error and update the experience priority
        #td_errors = q_targets - q_estimates
        #for e, td_error in zip(es, td_errors):
        #    e.weight = (td_error.item() + PRIORITY_OFFSET) ** PRIORITISATION

        # Compute loss, backprop and update weights
        self.optimizer.zero_grad()
        loss = F.mse_loss(is_ratios * q_estimates, is_ratios * q_targets)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = recordclass("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done", "weight"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):

        """Add a new experience to memory."""
        priority = self.weight_max() ** PRIORITISATION
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def weight_sum(self):
        return sum([e.weight for e in self.memory])

    def weight_max(self):
        if len(self.memory) == 0:
            return 1
        return max([e.weight for e in self.memory])

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # Calculate the sample probabilities, based on experience priority
        weights = [e.weight for e in self.memory]

        # Sample
        experiences = random.choices(self.memory, k=self.batch_size, weights=weights)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(
            np.vstack([e.weight for e in experiences if e is not None])).float().to(device)

        return (experiences, states, actions, rewards, next_states, dones, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

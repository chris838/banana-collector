### Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.


### DQN (Deep Q-Network)

#### Algorithm

We use a simplified form of Deep Q-Network inspired by https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

The agent interacts with the environment, selecting actions and recording experience tuples of the form (s,a,r,s') in a replay buffer. Here we use a replay buffer of size 10^5, discarding older samples once we fill the buffer. We'll cover the method for selecting actions later.

Once we have enough experience samples to fill a minibatch (here we use minibatch size of 64) we can start learning. A learning iteration is performed once per every 4 interactions with the environment. Each iteration random samples a minibatch of 64 experience tuples from the replay buffer, then uses these samples to update an estimate of the action-value (Q-value) function.

Our Q-value function estimate is formed from a neural network that, given a state as input, predicts the value of each of the four possible actions. We use neural-net formed from 37 inputs, two fully-connected hidden layers each with 64 neurons, and 4 outputs.

    QNetwork(
      (fc1): Linear(in_features=37, out_features=64, bias=True)
      (fc2): Linear(in_features=64, out_features=64, bias=True)
      (fc3): Linear(in_features=64, out_features=4, bias=True)
    )

This means our model only has (37+1) * 64 + (64+1) * 64 + (64+1) * 4 = 6,852 model parameters to train.

Training is performed using Adam (https://arxiv.org/pdf/1412.6980.pdf), an alternative to stochastic gradient descent. We define the loss as the mean-squared error, across the minibatch sample, between targets and predictions. Recall that each experience tuple consists of (s,a,r,s'). The predicted action-value is the value of action a in state s, Q(s,a) as predicted by the current model. The target action-value is defined as r + γQ(s',a'), where a' is chosen to maximise Q(s',a') and γ is a discount factor set to 0.99.

As suggested in the DQN paper, we actually maintain two models during training - a local model and a target model. The local model is updated directly after each learning step, whereas the target model is 'soft updated' - it is interpolated with the local model according to a factor τ, which we set to 0.001.  We use the target model to generate target action-values and the local model for predicted action-values, which promotes more stable and more sample-efficient learning.

Finally, we select our agent's actions according to an ε-greedy policy, with ε initially at 1 and decaying with each episode by a factor of 0.995, to a minimum of 0.01. When asked to select an action, with probability ε we simply choose a random action, and with probability (1 - ε) we choose the highest valued action according to our local model's predicted action values for the current state.



#### Results

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

Solving the environment requires reaching an average score of 13 over 100 episodes. Our agent reached this target after 525 episodes. A plot of the rewards per episode is included below.


### Double-DQN

#### Algorithm

#### Results


### Prioritised Replay

#### Algorithm

#### Results


# Ideas for Future Work

The submission has concrete future ideas for improving the agent's performance.

# Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.


# DQN (Deep Q-Network)

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



## Results

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

Solving the environment requires reaching an average score of 13 over 100 episodes. Our agent reached this target after 525 episodes. A plot of the rewards per episode is included below.

![Reward Graph of DQN](https://github.com/chris838/banana-collector/blob/master/results/dqn_64_64.png)



# Double-DQN

Inspired by results from this paper https://arxiv.org/pdf/1509.06461.pdf, Double DQN improves action-value estimates by removing the overestimation bias that comes from the maximisation step. We maintain two different models, each with an independent estimate of action values. We then use one model to find the maximising action, then evaluate that action with the other model.

This is particularly easy to implement in our case, since we already have two versions of the model (our target and local models) which should be sufficiently different from one another.

## Results

![Reward Graph of Double DQN](https://github.com/chris838/banana-collector/blob/master/results/double_dqn_64_64.png)



# Prioritised Replay

Prioritised replay is described in this paper https://arxiv.org/pdf/1511.05952.pdf. Here, instead of sampling uniformly from our sample buffer, we sample according to each sample's priority, which is proportional to the TD-error of that sample when it was last used.

This requires adjusting our loss function to take into account the sampling bias, using something called an importance sampling ratio.

In the paper, the author recommends using a sum-tree data structure to generate sample minibatches efficiently. For the purposes of testing, we instead use a simple queue as in the original implementation. This drastically increases the amount of time it takes to run the algorithm, however since the problem is small anyway we can still run an entire test within 30 minutes on a laptop.


## Results

![Reward Graph of Prioritised Replay](https://github.com/chris838/banana-collector/blob/master/results/prioritised_dqn_64_64.png)



# Ideas for Future Work

Neither the prioritised replay, nor the double DQN method appeared to have a significant effect on the learning rate, although more tests would need to be performed to confirm this. In particular, prioritised has a number of sample parameters that could be adjusted. It might be prudent to re-implement prioritised replay using sum-trees, as this would permit running more tests in a shorter time.

The rainbow paper https://arxiv.org/pdf/1710.02298.pdf compares a number of additional extensions to DQN that have proved successful in improving learning rates. We might implement one or more of these and test their effects.

Another source of optimisation is the model itself. Here, we've used a very simple two layer model with 64 neurons in each. We could increases the number of layers and/or the number of neurons. This might also prompt the use of regularisation and other techniques to reduce overfitting - e.g. the use of dropout or residual neural nets. However, given that our simple model already performs reasonably well, I would be reluctant to make it more complex without justification.

We could also try approaches that differ quite substantially from DQN, such as policy-gradient methods and actor-critic methods, as these are considered state of the art concerning the Atari games that the DQN paper was originally applied to.

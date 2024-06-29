"""
Carlos Luz
CS181
"""

# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
#from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900



class Learner(object):
    """
    This agent jumps randomly.
    We can do better by implementing Q-Learning.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.init_decay = 1

        # Model hyper-params
        self.epsilon = 0.1
        self.learning_rate = 0.8
        self.gamma = .75
        self.decay_rate = .9


        # Initialize Q-table with zeros
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    # Reset the state of the agent for each new game.
    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    # We want to discretize the state space to make it easier to learn
    # We can do this by binning the state space into discrete bins
    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    # Implement the Q-Learning algorithm
    # 1 -> monkey jumps
    # 0 -> monkey does not jump
    def action_callback(self, state):

        # Discretize the state
        current_state = self.discretize_state(state)

        # The initial state is None when first starting -> pick action via epsilon greedy
        if (self.last_state == None):
            self.last_action = self.determine_action(current_state)
            self.last_state = current_state
            self.init_decay -= self.decay_rate
            return self.last_action
        else:

            """
            Q <- Q + LR*(gamma * max_current(Q) - Q_prior)
            Update Q value of prior state based on the reward and the max Q value of the current state
            """

            # Find max_current(Q): find max Q across either action given our current state
            max_Q_val = np.max(self.Q[:, current_state[0], current_state[1]])

            # Q value of prior state
            prior_Q_val = self.Q[self.last_action, self.last_state[0], self.last_state[1]]

            # Update
            self.Q[self.last_action, self.last_state[0], self.last_state[1]] = prior_Q_val + self.learning_rate * (self.last_reward + self.gamma * max_Q_val - prior_Q_val)

        # Determine action via Epsilon-Greedy:
        self.last_action = self.determine_action(state=current_state)

        # "Archive" this current state
        self.last_state = current_state

        # Update epsilon
        self.init_decay *= self.decay_rate
        return self.last_action
    
    # Determine action based on epsilon greedy
    def determine_action(self, state):
        if (np.random.rand() < self.epsilon * self.init_decay):
            next_action = np.random.randint(0,2)
        else:
            next_action = np.argmax(self.Q[:, state[0], state[1]])
        
        return next_action

    # Update the reward
    def reward_callback(self, reward):
        self.last_reward = reward


# Run the game, and save the history of the scores
def run_games(learner, hist, iters=100, t_len=100):

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    agent = Learner()
    hist = []
    run_games(agent, hist, 100, 100)
    print(hist)

    # Data saved in the .npy file -> used to make graphs
    np.save('hist', np.array(hist))

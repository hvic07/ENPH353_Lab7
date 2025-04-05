import random
import pickle
import os
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                q_table = pickle.load(f)
            print(f"Loaded best policy from {filename}")
            return q_table
        else:
            print("No previous best policy found. Starting fresh.")
            return None  # Return None if there's no saved policy

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 
        if random.uniform(0, 1) < self.epsilon:
        # Exploration: Pick a random action
            return random.choice(self.actions)
        else:
            # Exploitation: Pick the best action based on Q-values
            q_values = {a: self.q.get((state, a), 0) for a in self.actions}
            return max(q_values, key=q_values.get)  # Action with highest Q-value

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        if (state1, action1) not in self.q:
            self.q[(state1, action1)] = 0  # Initialize unseen state-action pairs

        # Find max(Q(s2, a)) over all possible actions in state2
        max_future_q = max(
            [self.q.get((state2, a), 0) for a in ["L", "R", "F"]]  # Default Q=0 if unseen
        )

        # Bellman update equation
        self.q[(state1, action1)] += self.alpha * (reward + self.gamma * max_future_q - self.q[(state1, action1)])

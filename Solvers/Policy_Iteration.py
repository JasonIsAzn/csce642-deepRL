# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
from Solvers.Abstract_Solver import AbstractSolver, Statistics

def get_random_policy(num_states, num_actions):
    policy = np.zeros([num_states, num_actions])
    for s_idx in range(num_states):
        action = s_idx % num_actions
        policy[s_idx, action] = 1
    return policy

class PolicyIteration(AbstractSolver):

    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)
        # Start with a random policy
        # self.policy[s,a] denotes \pi(a|s)
        # Note: Policy is determistic i.e., only one element in self.policy[s,:] is 1 rest are 0
        self.policy = get_random_policy(env.observation_space.n, env.action_space.n)

    def train_episode(self):
        """
            Run a single Policy iteration. Evaluate and improve the policy.

            Use:
                self.policy: [S, A] shaped matrix representing the policy.
                             self.policy[s,a] denotes \pi(a|s)
                             Note: Policy is determistic i.e., only one element in self.policy[s,:] is 1 rest are 0
                self.env: OpenAI environment.
                    env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                    env.observation_space.n is the number of states in the environment.
                    env.action_space.n is the number of actions in the environment.
                self.options.gamma: Gamma discount factor.
                np.eye(self.env.action_space.n)[action]
        """

        # Evaluate the current policy
        self.policy_eval()

        # For each state...
        for s in range(self.env.observation_space.n):
            # Find the best action by one-step lookahead
            # Ties are resolved in favor of actions with lower indexes (Hint: use max/argmax directly).
            q_sa = self.one_step_lookahead(s)
            a = q_sa.argmax()
            
            self.policy[s, :] = 0
            self.policy[s, a] = 1

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Policy Iteration"

    def one_step_lookahead(self, state):
        """
        Helper function to calculate the value for all actions from a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A


    def policy_eval(self):
        # We want to solve Ax=B where
        # A = (I - y * P_pi)
        # B = R_pi
        # x = V_pi
        
        s_size = self.env.observation_space.n
        a_size = self.env.action_space.n

        # Compute P_pi and R_pi
        P_pi = np.zeros((s_size, s_size))
        R_pi = np.zeros(s_size)
        
        for s in range(s_size):
            for a in range(a_size):
                policy_sa = self.policy[s,a]
                if policy_sa == 0:
                    continue
                
                for prob, next_state, reward, done in self.env.P[s][a]:
                    R_pi[s] += policy_sa * prob * reward
                    
                    P_pi[s,next_state] += policy_sa * prob
        
        # Solve (I - y * P_pi) V = R_pi
        A = np.eye(s_size) - self.options.gamma * P_pi
        
        # Solve Ax = B
        self.V = np.linalg.solve(A, R_pi)


    def create_greedy_policy(self):
        """
        Return the currently known policy.

        Returns:
            A function that takes an observation as input and greedy action as integer
        """
        def policy_fn(state):
            return np.argmax(self.policy[state])

        return policy_fn

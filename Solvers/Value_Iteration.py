
import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics


class ValueIteration(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)

    def train_episode(self):
        for each_state in range(self.env.observation_space.n):
            A = self.one_step_lookahead(each_state)
            v_s = A.max()
            self.V[each_state] = v_s
            
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def one_step_lookahead(self, state: int):
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A

    def create_greedy_policy(self):
        def policy_fn(state):
            A = self.one_step_lookahead(state)
            pi_s = np.argmax(A)
            return pi_s
        
        return policy_fn


class AsynchVI(ValueIteration):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s] - best_action_value))
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):
        """
        What is this?
            same as other `train_episode` function above, but for Asynch value iteration

        New Inputs:

            self.pq.update(state, priority)
                priority is a number BUT more-negative == higher priority

            state = self.pq.pop()
                this gets the state with the highest priority

        Update:
            self.V
                this is still the same as the previous
        """
        for _ in range(self.env.observation_space.n):            
            s = self.pq.pop()   
            
            A = self.one_step_lookahead(s)
            v_s = A.max()
            self.V[s] = v_s
            
            # Update priorities
            for p in self.pred[s]:
                A_p = self.one_step_lookahead(p)
                v_p = A_p.max()
                self.pq.update(p, -abs(self.V[p] - v_p))

        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

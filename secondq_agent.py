import random
from collections import defaultdict
import numpy as np

class QAgent:
    def _init_(self, num_workers=3, state_buckets=5, alpha=0.6, gamma=0.9,
                 eps_start=0.3, eps_min=0.02, eps_decay=0.999):
        self.num_workers = num_workers
        self.state_buckets = state_buckets
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = eps_start    # exploration rate
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(num_workers, dtype=float))

    def state_from_queues(self, queue_lengths):
        """Convert queue lengths into discrete bucketed state."""
        buckets = []
        for l in queue_lengths:
            norm = min(l, 20) / 20.0
            bucket = int(norm * (self.state_buckets - 1))
            buckets.append(bucket)
        return tuple(buckets)

    def choose_worker(self, state):
        """Choose a worker using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.randrange(self.num_workers)
        else:
            qvals = self.Q[state]
            maxv = qvals.max()
            best = [i for i, v in enumerate(qvals) if v == maxv]
            action = random.choice(best)
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return action

    def update(self, state, action, reward, next_state):
        """Standard Q-learning update rule."""
        q_current = self.Q[state][action]
        q_next_max = self.Q[next_state].max()
        self.Q[state][action] = q_current + self.alpha * (
            reward + self.gamma * q_next_max - q_current
        )

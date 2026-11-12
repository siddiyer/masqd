import numpy as np
import torch


class ReplayBuffer:
    """
    Per-agent replay buffer. Besides the usual transition, each entry stores two behavior descriptors:
    d_prime, the descriptor the transition is labeled with, and d, the descriptor the rollout actually expressed.
    """

    def __init__(self, state_dim, action_dim, bd_size, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.d_prime = np.zeros((max_size, bd_size))
        self.d = np.zeros((max_size, bd_size))

    def add(self, state, action, next_state, reward, done, d_prime, d):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.d_prime[self.ptr] = d_prime
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Returns (state, action, next_state, reward, not_done, d_prime, d) as float tensors."""
        ind = np.random.randint(0, self.size, size=batch_size)
        assert self.size >= batch_size, "Batch size larger than experiences in the replay buffer: " + str(self.size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind]),
            torch.FloatTensor(self.d_prime[ind]),
            torch.FloatTensor(self.d[ind])
        )

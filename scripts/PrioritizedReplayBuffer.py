'''
Author: Xiaoxun Zhang
Date: 2025-03-24 20:19:48
LastEditTime: 2025-03-24 20:37:26
Description: 
'''

import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = int(capacity)
        self.alpha = alpha  # 优先级系数
        self.beta = beta    # 重要性采样系数
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(self.capacity)
        self.position = 0
        self.size = 0

    def push(self, experience, td_error=None):
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append(experience)
            self.priorities[self.position] = max_priority
            self.size += 1
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.size < self.capacity:
            probs = self.priorities[:self.size]
        else:
            probs = self.priorities

        # 计算采样概率
        probs = probs ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性权重
        total = self.size if self.size < self.capacity else self.capacity
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return [self.memory[idx] for idx in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return self.size

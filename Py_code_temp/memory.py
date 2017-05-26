import random

from collections import deque

class ExperienceMemory(object):
    #here experience is observation frame
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = deque()
        self.current_elbo = 0

    def add_experience(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.max_size:
            self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))
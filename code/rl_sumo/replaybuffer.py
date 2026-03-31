
import random
import torch

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.capacity = args.capacity
        self.buffer = []
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            "None可以用作占位符，以后可能会被实际值替换。例如，在初始化一个固定大小的列表时，可以先用None填充，然后在稍后分配具体值。"
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # state = torch.tensor(np.array(state), device=self.args.device, dtype=torch.float32)
        # action = torch.tensor(np.array(action), device=self.args.device, dtype=torch.int64)
        # reward = torch.tensor(reward, device=self.args.device, dtype=torch.float32)
        # next_state = torch.tensor(np.array(next_state), device=self.args.device, dtype=torch.float32)
        # done = torch.tensor(done, dtype=torch.float, device=self.args.device)
        # edge_index = torch.tensor(edge_index, dtype=torch.float, device=self.args.device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from torch.optim.lr_scheduler import StepLR

# Configurations
reward_shaping = True
showFigures = True
printDetails = True

GAMMA = 0.01  # Changed to a more common value for future reward consideration
BATCH_SIZE = 64
TARGET_UPDATE = 100
MEMORY_CAPACITY = 15000  # Increased capacity
EPS_Ratio = 1
EPS_MIN = 0.0001
EPS_DECAY_ratio = 0.00004
LearnStep = 0
episodes = 30
SEQ_LENGTH = 5

# Setup the environment and device
env = model.model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Priority Experience Replay
Transition = namedtuple('Transition', ('state_seq', 'action', 'reward', 'next_state_seq', 'priority'))

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(max_priority)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) ** beta
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=sample_probs)
        transitions = [self.memory[idx] for idx in indices]
        return transitions, indices, sample_probs[indices]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# Neural Network class with LSTM
class Net(nn.Module):
    def __init__(self, STATE_SIZE, ACTION_SIZE):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(STATE_SIZE, 512, batch_first=True)
        self.fc = nn.Linear(512, ACTION_SIZE)

    def forward(self, x):
        x, _ = self.lstm(x)
        if x.dim() == 3:
            x = x[:, -1, :]
        else:
            x = x
        x = F.relu(x)
        return self.fc(x)

class Agent1:
    def __init__(self, STATE_SIZE, ACTION_SIZE, action_space, seq_length=SEQ_LENGTH):
        self.STATE_SIZE = STATE_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.action_space = action_space
        self.seq_length = seq_length
        self.policy_net = Net(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_net = Net(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.learn_counter = 0
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.loss_func = nn.MSELoss()
        self.best_reward = float('-inf')  # Track the best reward
        self.state_sequence = deque(maxlen=seq_length)
        self.lr_min = 0.00001

        # 添加最大步数限制
        self.beta = 0.4
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.steps_max = 2 * MEMORY_CAPACITY  # 设置最大步数为经验回放区大小

    def update_beta(self):
        # 更新 beta 值
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * (self.learn_counter / self.steps_max)
        self.beta = min(self.beta, self.beta_end)  # 确保 beta 不超过 1.0

    def select_action(self, state):
        global EPS_Ratio, EPS_DECAY_ratio
        EPS_Ratio = max(EPS_MIN, EPS_Ratio - EPS_DECAY_ratio)
        self.state_sequence.append(state)
        if len(self.state_sequence) < self.seq_length:
            return random.randrange(self.ACTION_SIZE)

        state_seq = np.array(self.state_sequence)
        state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(device)

        if random.random() > EPS_Ratio:
            with torch.no_grad():
                action_values = self.policy_net(state_seq)
            action_index = torch.argmax(action_values).item()
        else:
            action_index = random.randrange(self.ACTION_SIZE)
        return action_index

    def store_transition(self, s, a, r, s1, priority):
        self.memory.push(s, a, r, s1, priority)

    def learn(self):
        self.update_beta()  # 每次学习时更新 beta
        if self.learn_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1

        if len(self.memory) < BATCH_SIZE:
            return

        transitions, indices, probs = self.memory.sample(BATCH_SIZE, self.beta)  # 使用更新后的 beta
        batch = Transition(*zip(*transitions))

        # Convert state sequence batch to numpy array before creating tensor
        state_seq_array = np.array(batch.state_seq)
        state_seq_batch = torch.FloatTensor(state_seq_array).to(device)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).view(-1, 1).to(device)
        next_state_seq_array = np.array(batch.next_state_seq)
        next_state_seq_batch = torch.FloatTensor(next_state_seq_array).to(device)

        Q = self.policy_net(state_seq_batch).gather(1, action_batch)

        next_state_actions = self.policy_net(next_state_seq_batch).max(1)[1].view(-1, 1)
        Q1 = self.target_net(next_state_seq_batch).gather(1, next_state_actions).detach()

        target = reward_batch + GAMMA * Q1
        loss = F.smooth_l1_loss(Q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        priorities = (loss.detach().cpu().numpy() + 1e-5).reshape(-1)  # Ensure priorities is a list or array
        self.memory.update_priorities(indices, priorities.tolist())  # Convert to list

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.lr_min)

        return loss

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def save_best_model(self, reward, path):
        if reward > self.best_reward:
            self.best_reward = reward
            self.save_model(path)
            print(f"New best model saved with reward: {reward}")


class DRQN:
    def __init__(self):
        self.action = np.load("action_all.npy", allow_pickle=True)

    def print_model_parameters(self):
        Agent = Agent1(3, 25410, self.action)
        print(Agent.policy_net)

    def train(self):
        Agent = Agent1(3, 25410, self.action, seq_length=SEQ_LENGTH)  # Create instance of choose_tchws
        rewards_list = []

        with open("DDQN+DRQN-5-001.txt", "a") as file:
            for episode in range(episodes):
                s = env.reset()
                Agent.state_sequence.clear()
                r = 0

                for _ in range(Agent.seq_length - 1):
                    Agent.state_sequence.append(s)

                while True:
                    global LearnStep
                    LearnStep += 1
                    action_index = Agent.select_action(s)

                    T_chws = self.action[action_index][0]
                    f_tower = self.action[action_index][1]
                    f_cw_pump = self.action[action_index][2]
                    f_chw_pump = self.action[action_index][3]

                    S_, CLc, Done, P_chiller, P_tower, R, R_p, R_c, T_chwr, T_outdoor, P_total, Chiller_number, T_cwr = env.step(
                        self.action[action_index])

                    # Convert state sequence to tensor
                    state_seq_array = np.array(list(Agent.state_sequence))
                    state_seq_tensor = torch.FloatTensor(state_seq_array).unsqueeze(0).to(device)

                    # Correct usage of target_net and policy_net
                    with torch.no_grad():
                        policy_net_output = Agent.policy_net(state_seq_tensor)
                        target_net_output = Agent.target_net(state_seq_tensor)

                    td_error = abs(R + GAMMA * target_net_output.max().item() - policy_net_output.max().item())
                    Agent.store_transition(s, action_index, R, S_, td_error)
                    r += R

                    if len(Agent.memory) > BATCH_SIZE:
                        loss = Agent.learn()

                    if Done:
                        print(f"Episode: {episode}, Reward: {r}, EPS_Ratio: {EPS_Ratio:.4f}")
                        Agent.save_best_model(r, "best_model.pth")
                        break
                    s = S_

                    file.write(f"{episode}, {LearnStep}, {P_total:.3f}, {P_chiller:.3f}, {P_tower:.3f}, {CLc:.3f}, "
                               f"{T_outdoor}, {Chiller_number},{T_chws},{f_chw_pump},{f_cw_pump},{f_tower},{R_p:.3f},"
                               f"{R_c:.3f},{R:.3f},{T_cwr:.3f},{T_chwr:.3f}\n")

                    rewards_list.append(r)

        if showFigures:
            plt.figure(figsize=(10, 5))
            plt.plot(rewards_list, label="Rewards per Episode")
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title('Training Convergence')
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    train = DRQN()
    train.train()

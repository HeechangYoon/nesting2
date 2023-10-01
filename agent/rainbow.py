import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from collections import deque
from agent.network_iqn import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, batch_size, gamma=0.99, n_step=1, alpha=0.6, beta_start=0.4, beta_steps=10000):
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps

        self.frame = 1  # for beta calculation
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for i in range(self.n_step):
            Return += self.gamma ** i * n_step_buffer[i][3]

        return n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2], Return, n_step_buffer[-1][4], n_step_buffer[-1][5]

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_steps)

    def add(self, state, angle, position, reward, next_state, done):
        # n_step calc
        self.n_step_buffer.append((state, angle, position, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, angle, position, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, angle, position, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, angle, position, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, angles, positions, rewards, next_states, dones = zip(*samples)
        return states, angles, positions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self, state_size, x_action_size, a_action_size, capacity=1000, alpha=0.6, beta_start=0.4, beta_steps=100000,
                 n_step=3, batch_size=64, lr=0.0000001, lr_step=2000, lr_decay=0.9, tau=0.001, gamma=0.9, N=8):
        self.state_size = state_size
        self.x_action_size = x_action_size
        self.a_action_size = a_action_size
        self.tau = tau
        self.N = N
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        self.gamma = gamma

        self.batch_size = batch_size
        self.n_step = n_step
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = Network(state_size, x_action_size, a_action_size, N).to(device)
        self.qnetwork_target = Network(state_size, x_action_size, a_action_size, N).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=lr_step, gamma=lr_decay)

        # Replay memory
        self.memory = PrioritizedReplay(capacity, self.batch_size, gamma=self.gamma, n_step=n_step,
                                        alpha=alpha, beta_start=beta_start, beta_steps=beta_steps)

    def step(self, state, angle, position, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, angle, position, reward, next_state, done)

        loss = None
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences)

        return loss

    def get_angle(self, state, possible_a, eps=0.0, noisy=True):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)

        if random.random() >= eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues_a(state.permute(0, 3, 1, 2), noisy=noisy)
            action_values = action_values.cpu().data.numpy()
            mask = np.ones_like(action_values)
            for i in range(len(possible_a)):
                mask[i, possible_a[i]] = 0.0
            const = 1.5 * (np.max(action_values) - np.min(action_values))
            action_values = action_values - const * mask
            angle = np.argmax(action_values, axis=1)
        else:
            angle = [random.choices(candidate)[0] for candidate in possible_a]

        return angle[0]

    def get_position(self, state, possible_x, eps=0.0, noisy=True):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)

        if random.random() >= eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues_x(state.permute(0, 3, 1, 2), noisy=noisy)
            action_values = action_values.cpu().data.numpy()
            mask = np.ones_like(action_values)
            for i in range(len(possible_x)):
                mask[i, possible_x[i]] = 0.0
            const = 1.5 * (np.max(action_values) - np.min(action_values))
            action_values = action_values - const * mask
            position = np.argmax(action_values, axis=1)
        else:
            position = [random.choices(candidate)[0] for candidate in possible_x]

        return position[0]

    def learn(self, experiences):
        self.optimizer.zero_grad()

        states, angles, positions, rewards, next_states, dones, idx, weights = experiences
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        angles = torch.LongTensor(angles).to(device).unsqueeze(1)
        positions = torch.LongTensor(positions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        Q_targets_next_a, _ = self.qnetwork_target.forward_a(next_states, self.N)
        Q_targets_next_a = Q_targets_next_a.detach()  # (batch, num_tau, actions)
        q_t_n_a = Q_targets_next_a.mean(dim=1)

        Q_targets_next_x, _ = self.qnetwork_target.forward_x(next_states, self.N)
        Q_targets_next_x = Q_targets_next_x.detach()  # (batch, num_tau, actions)
        q_t_n_x = Q_targets_next_x.mean(dim=1)

        # calculate log-pi
        logsum_a = torch.logsumexp((Q_targets_next_a - Q_targets_next_a.max(2)[0].unsqueeze(-1)) / self.entropy_tau, 2).unsqueeze(-1)
        assert logsum_a.shape == (self.batch_size, self.N, 1), "log pi next has wrong shape"
        tau_log_pi_next_a = Q_targets_next_a - Q_targets_next_a.max(2)[0].unsqueeze(-1) - self.entropy_tau * logsum_a
        pi_target_a = F.softmax(q_t_n_a / self.entropy_tau, dim=1).unsqueeze(1)
        Q_target_a = (self.gamma ** self.n_step * (pi_target_a * (Q_targets_next_a - tau_log_pi_next_a) * (1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
        assert Q_target_a.shape == (self.batch_size, 1, self.N)

        logsum_x = torch.logsumexp((Q_targets_next_x - Q_targets_next_x.max(2)[0].unsqueeze(-1)) / self.entropy_tau, 2).unsqueeze(-1)
        assert logsum_x.shape == (self.batch_size, self.N, 1), "log pi next has wrong shape"
        tau_log_pi_next_x = Q_targets_next_x - Q_targets_next_x.max(2)[0].unsqueeze(-1) - self.entropy_tau * logsum_x
        pi_target_x = F.softmax(q_t_n_x / self.entropy_tau, dim=1).unsqueeze(1)
        Q_target_x = (self.gamma ** self.n_step * (pi_target_x * (Q_targets_next_x - tau_log_pi_next_x) * (1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
        assert Q_target_x.shape == (self.batch_size, 1, self.N)

        q_k_target_a = self.qnetwork_target.get_qvalues_a(states, noisy=True).detach()
        v_k_target_a = q_k_target_a.max(1)[0].unsqueeze(-1)
        tau_log_pik_a = q_k_target_a - v_k_target_a - self.entropy_tau * torch.logsumexp((q_k_target_a - v_k_target_a) / self.entropy_tau, 1).unsqueeze(-1)
        assert tau_log_pik_a.shape == (self.batch_size, self.a_action_size), "shape instead is {}".format(tau_log_pik_a.shape)
        munchausen_addon_a = tau_log_pik_a.gather(1, angles)

        q_k_target_x = self.qnetwork_target.get_qvalues_x(states, noisy=True).detach()
        v_k_target_x = q_k_target_x.max(1)[0].unsqueeze(-1)
        tau_log_pik_x = q_k_target_x - v_k_target_x - self.entropy_tau * torch.logsumexp((q_k_target_x - v_k_target_x) / self.entropy_tau, 1).unsqueeze(-1)
        assert tau_log_pik_x.shape == (self.batch_size, self.x_action_size), "shape instead is {}".format(tau_log_pik_x.shape)
        munchausen_addon_x = tau_log_pik_x.gather(1, positions)

        # calc munchausen reward:
        munchausen_reward_a = (rewards + self.alpha * torch.clamp(munchausen_addon_a, min=self.lo, max=0)).unsqueeze(-1)
        assert munchausen_reward_a.shape == (self.batch_size, 1, 1)
        # Compute Q targets for current states
        Q_targets_a = munchausen_reward_a + Q_target_a

        munchausen_reward_x = (rewards + self.alpha * torch.clamp(munchausen_addon_x, min=self.lo, max=0)).unsqueeze(-1)
        assert munchausen_reward_x.shape == (self.batch_size, 1, 1)
        # Compute Q targets for current states
        Q_targets_x = munchausen_reward_x + Q_target_x

        # Get expected Q values from local model
        q_k_a, taus_a = self.qnetwork_local.forward_a(states, self.N, noisy=True)
        Q_expected_a = q_k_a.gather(2, angles.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected_a.shape == (self.batch_size, self.N, 1)

        q_k_x, taus_x = self.qnetwork_local.forward_x(states, self.N, noisy=True)
        Q_expected_x = q_k_x.gather(2, positions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected_x.shape == (self.batch_size, self.N, 1)

        # Quantile Huber loss
        td_error_a = Q_targets_a - Q_expected_a
        assert td_error_a.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l_a = calculate_huber_loss(td_error_a, 1.0)
        quantil_l_a = abs(taus_a - (td_error_a.detach() < 0).float()) * huber_l_a / 1.0

        td_error_x = Q_targets_x - Q_expected_x
        assert td_error_x.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l_x = calculate_huber_loss(td_error_x, 1.0)
        quantil_l_x = abs(taus_x - (td_error_x.detach() < 0).float()) * huber_l_x / 1.0

        loss_a = quantil_l_a.sum(dim=1).mean(dim=1, keepdim=True) * weights  # , keepdim=True if per weights get multipl
        loss_x = quantil_l_x.sum(dim=1).mean(dim=1, keepdim=True) * weights  # , keepdim=True if per weights get multipl
        loss = loss_a + loss_x
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update priorities
        td_error = 0.5 * (td_error_a.sum(dim=1).mean(dim=1, keepdim=True) + td_error_x.sum(dim=1).mean(dim=1, keepdim=True)) # not sure about this -> test
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, episode, file_dir):
        torch.save({"episode": episode,
                    "model_state_dict": self.qnetwork_target.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "frame-%d.pt" % episode)

def calculate_huber_loss(td_errors, k=1.0):
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss
import torch
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from agent.network import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, x_action_size, a_action_size, learning_rate, gamma, lmbda, eps_clip, K_epoch):
        self.state_size = state_size
        self.x_action_size = x_action_size
        self.a_action_size = a_action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch

        self.data = []
        self.network = Network(state_size, x_action_size, a_action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, x_a_lst, a_a_lst, r_lst, s_prime_lst, x_prob_a_lst, a_prob_a_lst, x_mask_lst, a_mask_lst, done_lst = [], [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done = transition

            s_lst.append(s)
            x_a_lst.append([x_a])
            a_a_lst.append([a_a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            x_prob_a_lst.append([x_prob_a])
            a_prob_a_lst.append([a_prob_a])
            x_mask_lst.append(x_mask)
            a_mask_lst.append(a_mask)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask,  a_mask, done = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(x_a_lst).to(device), \
                                                                                 torch.tensor(a_a_lst).to(device), \
                                                                                 torch.tensor(r_lst, dtype=torch.float).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                                                                 torch.tensor(x_prob_a_lst).to(device),\
                                                                                 torch.tensor(a_prob_a_lst).to(device), torch.tensor(x_mask_lst).to(device), \
                                                                                 torch.tensor(a_mask_lst).to(device), \
                                                                                 torch.tensor(done_lst, dtype=torch.float).to(device)

        self.data = []
        return s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done

    def get_action(self, s, possible_actions):
        s = torch.from_numpy(s).float().to(device).unsqueeze(0)

        a_logit = self.network.a_pi(s.permute(0, 3, 1, 2))

        mask = np.ones(self.a_action_size)
        mask[possible_actions] = 0.0
        a_logit = a_logit - 1e8 * torch.from_numpy(mask).float().to(device)

        a_prob = torch.softmax(a_logit, dim=-1)[0]

        a_m = Categorical(a_prob)
        angle = a_m.sample().item()

        a = angle
        prob = a_prob[angle].item()

        return a, prob, mask

    def get_position(self, s, possible_x):
        s = torch.from_numpy(s).float().to(device).unsqueeze(0)
        x_logit = self.network.x_pi(s.permute(0, 3, 1, 2))
        mask_x = np.ones(self.x_action_size)
        mask_x[possible_x] = 0.0
        x_logit = x_logit - 1e8 * torch.from_numpy(mask_x).float().to(device)

        x_prob = torch.softmax(x_logit, dim=-1)[0]

        x_m = Categorical(x_prob)
        x = x_m.sample().item()

        a_x = x
        prob_x = x_prob[x].item()

        return a_x, prob_x, mask_x

    def train(self):
        s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done = self.make_batch()
        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.network.v(s_prime.permute(0, 3, 1, 2)) * done
            delta = td_target - self.network.v(s.permute(0, 3, 1, 2))
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            x_logit = self.network.x_pi(s.permute(0, 3, 1, 2))
            x_logit = x_logit.float() - 1e8 * x_mask.float()
            x_pi = torch.softmax(x_logit, dim=-1)
            x_pi_a = x_pi.gather(1, x_a)
            x_ratio = torch.log(x_pi_a) - torch.log(x_prob_a)  # a/b == exp(log(a)-log(b))
            x_ratio.clamp_(max=88).exp()

            x_surr1 = x_ratio * advantage
            x_surr2 = torch.clamp(x_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            x_loss = -torch.min(x_surr1, x_surr2)

            a_logit = self.network.a_pi(s.permute(0, 3, 1, 2))
            a_logit = a_logit.float() - 1e8 * a_mask.float()
            a_pi = torch.softmax(a_logit, dim=-1)
            a_pi_a = a_pi.gather(1, a_a)
            a_ratio = torch.log(a_pi_a) - torch.log(a_prob_a)  # a/b == exp(log(a)-log(b))
            a_ratio.clamp_(max=88).exp()

            a_surr1 = a_ratio * advantage
            a_surr2 = torch.clamp(a_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            a_loss = -torch.min(a_surr1, a_surr2)

            loss = x_loss + a_loss + 0.2 * F.smooth_l1_loss(self.network.v(s.permute(0, 3, 1, 2)), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "model.pt")
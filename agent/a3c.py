import torch
import torch.multiprocessing as mp

from torch.distributions import Categorical
from agent.network import *
from environment.env import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    def __init__(self, name, n_episode, t_max, gamma, lmbda, global_network, shared_optimizer, lock,
                 global_episode, global_episode_reward, global_efficiency, global_batch_rate, global_loss, res_queue):
        super(Worker, self).__init__()
        self.name = 'worker-%02i' % name
        self.n_episode = n_episode
        self.t_max = t_max
        self.gamma = gamma
        self.lmbda = lmbda
        self.global_network = global_network
        self.shared_optimizer = shared_optimizer
        self.lock = lock
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.global_efficiency = global_efficiency
        self.global_batch_rate = global_batch_rate
        self.global_loss = global_loss
        self.res_queue = res_queue

        self.data = []
        self.env = HiNEST(look_ahead=5)
        self.local_network = Network(self.env.state_size, self.env.x_action_size, self.env.a_action_size).to(device)

    def get_action(self, s, possible_actions):
        s = torch.from_numpy(s).float().to(device).unsqueeze(0)
        with torch.no_grad():
           a_logit = self.local_network.a_pi(s.permute(0, 3, 1, 2))

        mask = np.ones(self.env.a_action_size)
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
        with torch.no_grad():
            x_logit = self.local_network.x_pi(s.permute(0, 3, 1, 2))
        mask_x = np.ones(self.env.x_action_size)
        mask_x[possible_x] = 0.0
        x_logit = x_logit - 1e8 * torch.from_numpy(mask_x).float().to(device)

        x_prob = torch.softmax(x_logit, dim=-1)[0]

        x_m = Categorical(x_prob)
        x = x_m.sample().item()

        a_x = x
        prob_x = x_prob[x].item()

        return a_x, prob_x, mask_x

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, x_a_lst, a_a_lst, r_lst, s_prime_lst, x_prob_a_lst, a_prob_a_lst, x_mask_lst, a_mask_lst, done_lst \
            = [], [], [], [], [], [], [], [], [], []
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

        s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done \
            = (torch.tensor(s_lst, dtype=torch.float).to(device),
               torch.tensor(x_a_lst).to(device),
               torch.tensor(a_a_lst).to(device),
               torch.tensor(r_lst, dtype=torch.float).to(device),
               torch.tensor(s_prime_lst, dtype=torch.float).to(device),
               torch.tensor(x_prob_a_lst).to(device),
               torch.tensor(a_prob_a_lst).to(device),
               torch.tensor(x_mask_lst).to(device),
               torch.tensor(a_mask_lst).to(device),
               torch.tensor(done_lst, dtype=torch.float).to(device))

        self.data = []

        return s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done

    def train(self):
        s, x_a, a_a, r, s_prime, x_prob_a, a_prob_a, x_mask, a_mask, done = self.make_batch()

        with torch.no_grad():
            td_target = r + self.gamma * self.local_network.v(s_prime.permute(0, 3, 1, 2)) * done
            delta = td_target - self.local_network.v(s.permute(0, 3, 1, 2))
            delta = delta.cpu().detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

        x_logit = self.local_network.x_pi(s.permute(0, 3, 1, 2))
        x_logit = x_logit.float() - 1e8 * x_mask.float()
        x_pi = torch.softmax(x_logit, dim=-1)
        x_m = Categorical(x_pi)

        a_logit = self.local_network.a_pi(s.permute(0, 3, 1, 2))
        a_logit = a_logit.float() - 1e8 * a_mask.float()
        a_pi = torch.softmax(a_logit, dim=-1)
        a_m = Categorical(a_pi)

        loss = (- (x_m.log_prob(x_a) + a_m.log_prob(a_a)) * advantage
                + 0.5 * F.smooth_l1_loss(self.local_network.v(s.permute(0, 3, 1, 2)), td_target.detach())
                - 0.01 * (x_m.entropy().unsqueeze(-1) + a_m.entropy().unsqueeze(-1)))

        self.shared_optimizer.zero_grad()
        loss.mean().backward()
        for lp, gp in zip(self.local_network.parameters(), self.global_network.parameters()):
            gp._grad = lp.grad
        self.shared_optimizer.step()
        self.local_network.load_state_dict(self.global_network.state_dict())

        avg_loss = loss.mean().item()
        return avg_loss

    def run(self):
        total_step = 1
        while self.global_episode.value < self.n_episode:
            s = self.env.reset()
            episode_reward = 0.0

            while True:
                possible_actions = self.env.get_possible_actions()
                a, prob, mask = self.get_action(s, possible_actions)
                possible_x = self.env.get_possible_positions(a)
                a_x, prob_x, mask_x = self.get_position(s, possible_x)

                s_prime, r, efficiency, batch_rate, done, overlap, temp = self.env.step((a_x, a))
                self.put_data((s, a_x, a, r, s_prime, prob_x, prob, mask_x, mask, done))

                s = s_prime
                episode_reward += r

                if total_step % self.t_max == 0 or done:  # update global and assign to local net
                    # sync
                    loss = self.train()

                    if done:  # done and print information
                        with self.lock:
                            self.global_episode.value += 1

                            if self.global_episode_reward.value == 0.0:
                                self.global_episode_reward.value = episode_reward
                            else:
                                self.global_episode_reward.value = self.global_episode_reward.value * 0.99 + episode_reward * 0.01

                            if self.global_efficiency.value == 0.0:
                                self.global_efficiency.value = efficiency
                            else:
                                self.global_efficiency.value = self.global_efficiency.value * 0.99 + efficiency * 0.01

                            if self.global_batch_rate.value == 0.0:
                                self.global_batch_rate.value = batch_rate
                            else:
                                self.global_batch_rate.value = self.global_batch_rate.value * 0.99 + batch_rate * 0.01

                            if self.global_loss.value == 0.0:
                                self.global_loss.value = loss
                            else:
                                self.global_loss.value = self.global_loss.value * 0.99 + loss * 0.01

                        self.res_queue.put([self.global_episode.value, self.global_episode_reward.value,
                                            self.global_efficiency.value, self.global_batch_rate.value, self.global_loss.value])

                        print(self.name,
                              "Episode:", self.global_episode.value,
                              "| Episode_reward: %.2f" % self.global_episode_reward.value,
                              "| Efficiency: %.2f" % self.global_efficiency.value,
                              "| Batch_rate: %.2f" % self.global_batch_rate.value,
                              "| Loss: %.2f" % self.global_loss.value)

                        break

                total_step += 1
        self.res_queue.put(None)


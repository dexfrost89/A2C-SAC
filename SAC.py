from torch.distributions import Normal
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

cuda = torch.device("cuda:0")
cpu = torch.device("cpu")

class MyReplayBuffer:
    def __init__(self, s_size, action_size, max_size, batch_size):
        self.state_buf = np.zeros((max_size, s_size), dtype=np.float32)
        self.next_state_buf = np.zeros((max_size, s_size), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_size), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.max_size = max_size
        self.pointer = 0
        self.batch_size = batch_size
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        self.state_buf[self.pointer] = state
        self.next_state_buf[self.pointer] = next_state
        self.reward_buf[self.pointer, 0] = reward
        self.action_buf[self.pointer] = action
        self.done_buf[self.pointer, 0] = done

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return {"s": torch.Tensor(self.state_buf[indices]), \
                "next_s": torch.Tensor(self.next_state_buf[indices]), \
                "r": torch.Tensor(self.reward_buf[indices]), \
                "a": torch.Tensor(self.action_buf[indices]), \
                "done": torch.Tensor(self.done_buf[indices])}

    def __len__(self):
        return self.size

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_mean_layer = nn.Linear(hidden_dim, output_dim)
        self.output_std_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        out = F.relu(self.hidden_layer(out))

        out1 = self.output_mean_layer(out)
        out2 = F.softplus(self.output_std_layer(out))

        dist = Normal(out1, out2)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.input_layer1 = nn.Linear(state_dim, hidden_dim // 2)
        self.input_layer2 = nn.Linear(action_dim, hidden_dim // 2)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):

        inp1, inp2 = F.relu(self.input_layer1(s)), F.relu(self.input_layer2(a))

        inp = torch.concat((inp1, inp2), dim=-1)
        print(inp.mean())

        out = F.relu(self.hidden_layer(inp))
        print(out.mean())
        #out = F.relu(self.hidden_layer(out))

        return self.output_layer(out)

class SAC:
    def __init__(self):

        #Инитим среду
        self.env = gym.make('MountainCarContinuous-v0')

        #Инитим Агентов
        self.actor = Actor(2, 128, 1).to(cuda)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.ro = 0.99

        self.alpha_log = torch.Tensor(np.log([1.0])).to(cuda)
        self.alpha_log.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.alpha_log], lr=3e-4)
        self.desired_entropy = -1

        self.critic1 = Critic(2, 1, 128).to(cuda)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2 = Critic(2, 1, 128).to(cuda)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1 = Critic(2, 1, 128).to(cuda)
        self.target_critic2 = Critic(2, 1, 128).to(cuda)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.ReplayBuffer = MyReplayBuffer(2, 1, 50000, 128)

        self.writer = SummaryWriter()
        self.epoch = 0
        self.Q1_losseses = []
        self.Q2_losseses = []
        self.Actor_losseses = []
        self.alpha_losseses = []
        self.targets = []
        self.min_reward = 10000
        self.max_reward = -10000



    def train(self):
        for epoch in range(0):
            self.epoch = epoch
            obs, _ = self.env.reset()

            episode_reward = 0.0

            d = False

            while not d:

                actions, _ = self.get_action(obs)

                actions = actions.detach().to(cpu).numpy()

                next_obs, r, d, _, _ = self.env.step(actions)

                self.min_reward = min(r, self.min_reward)
                self.max_reward = max(r, self.max_reward)

                episode_reward += np.mean(r)

                self.ReplayBuffer.store(obs, actions, r, next_obs, d)

                obs = np.copy(next_obs)

        episode_reward = 0.0
        for epoch in range(10000):
            self.epoch = epoch
            obs, _ = self.env.reset()


            d = False

            step = 0

            while (not d) and step < 999:
                step += 1

                actions, _ = self.get_action(torch.Tensor(obs))

                actions = actions.detach().to(cpu).numpy()

                next_obs, r, d, _, _ = self.env.step(actions)

                if r > 50:
                    r -= 100
                    r = -(0.1 + r)
                else:
                    r = -(0.1 + r)
                    r -= 0.1

                r *= 5

                episode_reward += np.mean(r)

                self.min_reward = min(r, self.min_reward)
                self.max_reward = max(r, self.max_reward)

                self.ReplayBuffer.store(obs, actions, r, next_obs, d)

                obs = np.copy(next_obs)

            self.writer.add_scalar("Reward", episode_reward, epoch)

            if self.ReplayBuffer.size >= 5000:
                for update_num in range(20):

                    batch = self.ReplayBuffer.sample()

                    self.update_Q(batch)

                    self.update_Actor(batch)

                    self.update_target_Q()

            if epoch % 1 == 0:
                print('Loss/Q1', np.mean(self.Q1_losseses))
                print('Loss/Q2', np.mean(self.Q2_losseses))
                print('Loss/Actor', np.mean(self.Actor_losseses))
                print('Loss/alpha', np.mean(self.alpha_losseses))
                print('targets', np.mean(self.targets))
                print('min reward', self.min_reward)
                print('max reward', self.max_reward)
                print(epoch, episode_reward, self.alpha_log.exp())
                episode_reward = 0.0
                self.Q1_losseses = []
                self.Q2_losseses = []
                self.Actor_losseses = []
                self.alpha_losseses = []
                self.targets = []


    def get_action(self, s):
        actions, log_probs = self.actor(s.to(cuda))
        return actions, log_probs

    def get_Qs(self, s, a):
        Q1 = self.critic1(s.to(cuda), a)
        Q2 = self.critic2(s.to(cuda), a)
        return Q1, Q2

    def get_target_Qs(self, s, a):
        Q1 = self.target_critic1(s.to(cuda), a)
        Q2 = self.target_critic1(s.to(cuda), a)
        return Q1, Q2

    def update_Q(self, batch):

        alpha = self.alpha_log.exp()
        #print('alpha', alpha.item(), sep='\t')

        next_actions, next_actions_log_probas = self.get_action(batch['next_s'])
        print('batch["s"]', batch['s'].mean(), sep='\t')
        #print('next_actions', next_actions, sep='\t')
        #print('next_actions_log_probas', next_actions_log_probas, sep='\t')

        next_actions_log_probas = torch.sum(next_actions_log_probas, dim=-1).reshape(-1, 1)
        #print('next_actions_log_probas_sumed', next_actions_log_probas, sep='\t')

        target_Q1, target_Q2 = self.get_target_Qs(batch['next_s'], next_actions)
        #print('batch["next_s"]', batch['next_s'], sep='\t')
        print('target_Q1', target_Q1.mean(), sep='\t')
        print('target_Q2', target_Q2.mean(), sep='\t')

        min_target_Q = torch.min(target_Q1, target_Q2).reshape(-1, 1)
        print('min_target_Q', min_target_Q.mean(), sep='\t')

        #print(batch['r'].shape)
        #print((1 - batch['done']).shape)
        #print(min_target_Q.shape)
        #print(next_actions_log_probas.shape)

        target = (batch['r']).to(cuda) + (self.gamma * (1 - batch['done'])).to(cuda) * \
                                                         (min_target_Q - alpha * next_actions_log_probas)
        #print('batch["r"]', batch['r'], sep='\t')
        #print('batch["done"]', batch['done'], sep='\t')
        target = target.detach()

        print('target', target.mean(), sep='\t')

        self.targets.append(target.to(cpu).mean())

        Q1, Q2 = self.get_Qs(batch['s'], batch['a'].to(cuda))

        loss = nn.SmoothL1Loss()

        Q1_loss = loss(Q1, target.reshape(-1, 1))
        Q2_loss = loss(Q2, target.reshape(-1, 1))

        self.writer.add_scalar("Loss/Q1", Q1_loss.item(), self.epoch)
        self.writer.add_scalar("Loss/Q2", Q2_loss.item(), self.epoch)

        self.Q1_losseses.append(Q1_loss.item())
        self.Q2_losseses.append(Q2_loss.item())

        #print("Loss/Q1", Q1_loss.item(), self.epoch)
        #print("Loss/Q2", Q2_loss.item(), self.epoch)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        Q1_loss.backward()
        Q2_loss.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

    def update_Actor(self, batch):
        actions, log_probas = self.get_action(batch['s'])
        log_probas = torch.sum(log_probas, dim=-1)

        Q1, Q2 = self.get_Qs(batch['s'], actions)

        min_Q = torch.min(Q1, Q2).reshape(-1)

        Actor_loss = torch.mean(-min_Q + self.alpha_log.exp() * log_probas)

        self.writer.add_scalar("Loss/Actor", Actor_loss.item(), self.epoch)

        #print("Loss/Actor", Actor_loss.item(), self.epoch)
        self.Actor_losseses.append(Actor_loss.item())

        self.actor_optimizer.zero_grad()

        Actor_loss.backward()

        self.actor_optimizer.step()

        self.update_alpha(log_probas)

    def update_target_Q(self):
        target_critic1_state_dict = self.target_critic1.state_dict()
        critic1_state_dict = self.critic1.state_dict()
        for key in critic1_state_dict:
            target_critic1_state_dict[key] = critic1_state_dict[key] * (1 - self.ro) + target_critic1_state_dict[key] * self.ro
        self.target_critic1.load_state_dict(target_critic1_state_dict)

        target_critic2_state_dict = self.target_critic2.state_dict()
        critic2_state_dict = self.critic2.state_dict()
        for key in critic2_state_dict:
            target_critic2_state_dict[key] = critic2_state_dict[key] * (1 - self.ro) + target_critic2_state_dict[key] * self.ro
        self.target_critic2.load_state_dict(target_critic2_state_dict)

    def update_alpha(self, log_probas):
        loss = torch.mean(-self.alpha_log.exp() * (log_probas + self.desired_entropy).detach())

        self.writer.add_scalar("Loss/alpha", loss.item(), self.epoch)

        #print("Loss/alpha", loss.item(), self.epoch)
        self.alpha_losseses.append(loss.item())

        self.alpha_optimizer.zero_grad()

        loss.backward()

        self.alpha_optimizer.step()

agent = SAC()
agent.train()

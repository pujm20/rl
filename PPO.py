import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

def orthogonal_init(layer, gain=1.0):#初始化权重和偏置
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
 

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mu = nn.Linear(hidden_dim, action_dim)
        self.fc3_sigma = nn.Parameter(torch.zeros(action_dim))  # 使用nn.Parameter定义log_std
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3_mu, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.fc3_mu(x))  #tanh——>[-1, 1] or sigmoid——>[0, 1]; softmax——>输出和为1，用于分类，离散空间
        sigma = torch.exp(self.fc3_sigma)
        return mu, sigma
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# class ActorCritic:
#     def __init__(self,state_dim, action_dim, hidden_dim=128):
#         self.actor = Actor(state_dim, action_dim, hidden_dim=128)
#         self.critic = Critic(state_dim, action_dim, hidden_dim=128)

class RolloutBuffer():#
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.dws = []
        self.lod_prob = []
    def add(self, transition):
        state, action, reward, next_state, done, dw, log_prob = transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.dws.append(dw)
        self.lod_prob.append(log_prob)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.dws = []
        self.lod_prob = []

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
 
    def update(self, x):#动态更新平均值和标准差可以用到在线算法（online algorithm），其中最常见的方法是Welford的算法
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
 
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
 
        return x

#连续动作空间，在此环境只有一个维度，而mu，sigma
class PPO_agent:
    def __init__(self, args):
        self.device = args['device']
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.gamma = args['gamma']
        self.lamda = args['lamda']
        self.eps_clip = args['eps_clip']
        self.entropy_coef = args['entropy_coef']
        self.entropy_decay = args['entropy_decay']

        #actor输出动作采取概率分布，不同于价值方法DQN，mlp(state)输出动作Q值
        self.actor = Actor(args['state_dim'], args['hidden_dim'], args['action_dim']).to(self.device)
        self.critic = Critic(args['state_dim'], args['hidden_dim']).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args['lr_critic'])
        self.memory = RolloutBuffer()


    def take_action(self, state):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        with torch.no_grad():            
            mu, sigma = self.actor(state)
        action_dist = Normal(mu, sigma)
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True).item()
        action = action.detach().cpu().numpy().flatten()
        
        return action, log_prob

    def predict_action(self, state):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        with torch.no_grad():            
            mu, sigma = self.actor(state)
        action_dist = Normal(mu, sigma)
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)
        action = action.detach().cpu().numpy().flatten()
        return action
    
    def update(self):
        states = torch.tensor(np.array(self.memory.states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(self.memory.actions), device=self.device, dtype=torch.float32)
        rewards = torch.tensor(np.array(self.memory.rewards), device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(self.memory.next_states), device=self.device, dtype=torch.float32)
        dws = torch.tensor(np.float32(self.memory.dws), device=self.device).unsqueeze(1)
        dones = torch.tensor(np.float32(self.memory.dones), device=self.device).unsqueeze(1)
        log_probs = torch.tensor(np.array(self.memory.lod_prob), device=self.device, dtype=torch.float32).unsqueeze(1)
        # print(f"states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, next_states: {next_states.shape}, dones: {dones.shape}, dws: {dws.shape}, log_probs: {log_probs.shape}")
        # exit()
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)#标准化奖励

        with torch.no_grad(): 
            #计算advantages
            vs = self.critic(states)
            vs_ = self.critic(next_states)
            #td_target = rewards + self.gamma * vs_ * (1-dws)
            deltas = rewards + self.gamma * vs_ * (1.0 - dws) - vs
            advantages = torch.zeros_like(deltas).to(self.device)
            last_advantage = 0.0
            length = len(states)
            for t in reversed(range(length)):
                advantages[t] = deltas[t] + self.gamma * self.lamda * last_advantage * (1 - dones[t])
                last_advantage = advantages[t]
            td_target = advantages + vs
            advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8))#标准化优势
        

        self.entropy_coef *= self.entropy_decay#衰减熵系数，鼓励探索

        #训练
        indics = np.arange(length)#生成数组
        for _ in range(self.epochs):
            np.random.shuffle(indics)
            for start in range(0, length, self.batch_size):
                end = start + self.batch_size
                batch_indics = indics[start:end]
                states_batch = states[batch_indics]
                actions_batch = actions[batch_indics]
                log_probs_old_batch = log_probs[batch_indics]
                td_target_batch = td_target[batch_indics]

                mu_batch, sigma_batch = self.actor(states_batch)
                action_dist_batch = Normal(mu_batch, sigma_batch)
                log_probs_new_batch = action_dist_batch.log_prob(actions_batch)
                log_probs_new_batch = log_probs_new_batch.sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_probs_new_batch - log_probs_old_batch)
                surr1 = ratio * advantages[batch_indics]
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages[batch_indics]
                entropy = action_dist_batch.entropy().mean()#计算熵，鼓励探索
                actor_loss = -torch.min(surr1,surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # 可选: 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(td_target_batch, self.critic(states_batch))
                # 可选: L2 正则化
                # l2_loss = 0
                # for param in self.critic.parameters():
                #     l2_loss += torch.sum(param**2)
                # critic_loss += self.l2_reg * l2_loss

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # 可选: 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)         
                self.critic_optimizer.step()

        self.memory.clear()
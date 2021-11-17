import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F
from torch.optim import Adam
import tqdm
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 2 * action_dim)  # for mu and sigma
        ).to(device)
        self.sigma = None
        self.device = device
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        distrib = self.get_action_distribution(state)
        return distrib.log_prob(action).sum(-1)

    def get_action_distribution(self, state: torch.Tensor):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu, log_sigma = torch.chunk(self.model(state.to(self.device)), 2, dim=-1)
        sigma = torch.exp(log_sigma)
        return Normal(mu, sigma)  # batch_size x action_size
        
    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        distrib = self.get_action_distribution(state)
        action = distrib.rsample()
        action_tahn = torch.tanh(action)
        return action_tahn, action, distrib
        
class Critic(nn.Module):
    def __init__(self, state_dim, device):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        ).to(device)
        self.device = device
        
    def get_value(self, state):
        return self.model(state.to(self.device)).squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, device):
        self.actor = Actor(state_dim, action_dim, device)
        self.critic = Critic(state_dim, device)
        self.device = device
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)
        self.beta = 0.4
        self.d_target = 1000.
        self.clipping_e = 0.2

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float() # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float() # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx]).float().to(self.device) # Estimated by generalized advantage estimation
            
            # TODO: Update actor here            
            # TODO: Update critic here
            critic_loss = F.mse_loss(self.critic.get_value(s), v.to(self.device))
            p = torch.exp(self.actor.compute_proba(s, a))

            r = p / op.to(self.device)
            clamped_r = torch.clamp(r, 1.0 - self.clipping_e, 1.0 + self.clipping_e)
            actor_loss = torch.min(r * adv, clamped_r * adv).mean()

            # kullback_leibler = torch.kl_div(p, op)
            # self.update_beta(kullback_leibler.mean())
            # actor_loss = p / op * adv - self.beta * kullback_leibler

            loss = actor_loss + critic_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")

    def update_beta(self, kl):
        if kl > self.d_target * 1.5:
            self.beta *= 2.0
        elif kl < self.d_target / 1.5:
            self.beta /= 2.0



def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)

if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=DEVICE)
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    for i in tqdm.trange(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            ppo.save()

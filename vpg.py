import numpy as np
import gym
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical


def mlp(sizes, activation=nn.Tanh, last_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i != len(sizes) - 2:
            layers.append(activation())
    
    layers.append(last_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_acts, hidden_dims=[30]):
        super().__init__()
        self.pi = mlp(sizes=[obs_dim] + hidden_dims + [num_acts])
        self.v = mlp(sizes=[obs_dim] + hidden_dims + [1])
        self.oracle = {
            0: 1, 2: 1, 2: 1, 3: 0,
            4: 1, 5: 0, 6: 1, 7: 0,
            8: 2, 9: 2, 10: 1, 11: 0,
            12: 0, 13: 2, 14: 2, 15: 2,
        }
    
    def step(self, obs):
        logits = self.pi(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        values = self.v(obs)
        log_p = torch.log(dist.probs)
        return actions, values, log_p

    def get_dist(self, obs):
        logits = self.pi(obs)
        return Categorical(logits=logits)


class TrajectoryBuffer:
    def __init__(self, num_tau, T, obs_dim=1):
        self.num_tau = num_tau
        self.T = T
        self.obs_dim = obs_dim

        # Initialize buffers
        self.observations = torch.zeros((self.num_tau, self.T, self.obs_dim), dtype=torch.float32)
        self.rewards = torch.zeros((self.num_tau, self.T), dtype=torch.float32)
        self.actions = torch.zeros((self.num_tau, self.T), dtype=torch.float32)
        self.mask = torch.zeros((self.num_tau, self.T))
        self.pointer = 0

    def clear(self):
        self.observations.zero_()
        self.rewards.zero_()

    def reset_pointer(self):
        self.pointer = 0

    def put(self, tau_i, obs, r):
        assert tau_i <= self.num_tau
        assert self.pointer <= self.T

        self.observations[tau_i, self.pointer] = obs
        self.rewards[tau_i, self.pointer] = r
        self.mask[tau_i, self.pointer] = 1

        self.pointer += 1

    def compute_rewards(self):
        return torch.sum(self.rewards, dim=1)


class VPGTrainer:
    def __init__(self, env, actor_critic):
        self.env = env
        self.actor_critic = actor_critic
        self.buf = None
        self.optimizer = None

    def train_vpg(self, epochs=10, episodes_per_update=10,
            timesteps_per_episode=100, actor_lr=1e-3, critic_lr=1e-3
        ):
        self.buf = TrajectoryBuffer(num_tau=episodes_per_update, T=timesteps_per_episode, obs_dim=1)
        self.optimizer = torch.optim.Adam(self.actor_critic.pi.parameters(), lr=actor_lr)
        avg_return = []
        losses = []
        for n in range(epochs):
            # Fill buffer with experience
            self.collect_experience(num_tau=episodes_per_update, T=timesteps_per_episode)
            episode_rewards = self.buf.compute_rewards()
            
            self.optimizer.zero_grad()
            batch_loss = self.loss(
                self.buf.observations, self.buf.actions,
                episode_rewards, self.buf.mask
            )
            batch_loss.backward()
            self.optimizer.step()

            avg_return.append(episode_rewards.mean().item())
            losses.append(batch_loss.item())
        
        print(avg_return)
        print(losses)
        
    def collect_experience(self, num_tau, T):
        self.buf.clear()
        for tau_i in range(num_tau):
            self.buf.reset_pointer()
            o = torch.as_tensor([[self.env.reset()]], dtype=torch.float32)
            a, r = 0, 0.
            for t in range(T):
                a, v, log_p = self.actor_critic.step(o)
                a = a.item()
                o_, r, done, _ = self.env.step(a)
                o_ = torch.as_tensor([[o_]], dtype=torch.float32)
                self.buf.put(tau_i, o, r)
                o = o_
                if done:
                    break

    def loss(self, obs, acts, weights, mask):
        dist = self.actor_critic.get_dist(obs)
        log_p = dist.log_prob(acts)
        log_p *= mask
        temp = (log_p.T * weights).T
        return torch.sum(temp, dim=1).mean()


if __name__ == '__main__':
    sizes = [10, 30, 2]
    env = gym.make('FrozenLake-v0', is_slippery=False)
    env.reset()
    actor_critic = ActorCritic(obs_dim=1, num_acts=4)

    trainer = VPGTrainer(env, actor_critic)
    trainer.train_vpg(epochs=5, episodes_per_update=5, timesteps_per_episode=30)
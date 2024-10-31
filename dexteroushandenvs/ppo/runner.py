import os
import time

import numpy as np
import torch

from .utils import get_ppo_yaml_args, update_linear_schedule
from .model import Policy
from .ppo import PPO
from .storage import RolloutStorage

class PPORunner:
    def __init__(self, env, num_envs, output_dir):
        self.env = env
        self.num_envs = num_envs

        self.device = torch.device("cuda:0")

        print("State space:", self.env.state_space)
        print("Action space:", self.env.act_space)

        self.algo_args = get_ppo_yaml_args()
        self.actor_critic = Policy(
            self.env.state_space.shape,
            self.env.act_space,
            base_kwargs={'recurrent': self.algo_args["use_recurrent_policy"]})
        self.actor_critic.to(self.device)

        self.agent = PPO(
            self.actor_critic,
            self.algo_args["clip_param"],
            self.algo_args["ppo_epoch"],
            self.algo_args["num_mini_batch"],
            self.algo_args["value_loss_coef"],
            self.algo_args["entropy_coef"],
            lr=self.algo_args["lr"],
            eps=self.algo_args["eps"],
            max_grad_norm=self.algo_args["max_grad_norm"])
        
        self.rollouts = RolloutStorage(self.algo_args["num_steps"], self.num_envs, self.env.observation_space.shape, self.env.action_space, self.actor_critic.recurrent_hidden_state_size)

        self.output_dir = output_dir

    def run(self):
        obs_dict = self.env.reset()
        obs = obs_dict["states"]
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        episode_return = torch.zeros(self.num_envs, device=self.device)

        num_updates = int(
            self.algo_args["num_env_steps"]) // self.algo_args["num_steps"] // self.num_envs
        for j in range(num_updates):

            if self.algo_args["use_linear_lr_decay"]:
                # decrease learning rate linearly
                update_linear_schedule(self.agent.optimizer, j, num_updates, self.algo_args["lr"])

            for step in range(self.algo_args["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                # Observe reward and next obs
                obs_dict, reward, done, infos = self.env.step(action)
                episode_return += reward

                if torch.all(done):
                    obs_dict, _, _, _ = self.env.step(action)
                    print(f"Updates {j}, mean return {episode_return.mean().item():.2f}, max return {episode_return.max().item():.2f}, min return {episode_return.min().item():.2f}")
                    episode_return = torch.zeros(self.num_envs, device=self.device)

                obs = obs_dict["states"]

                # If done then clean the history of observations.
                masks = 1 - done
                bad_masks = 1 - done
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()


            self.rollouts.compute_returns(next_value, self.algo_args["use_gae"], self.algo_args["gamma"],
                                    self.algo_args["gae_lambda"], self.algo_args["use_proper_time_limits"])

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % self.algo_args["save_interval"] == 0
                    or j == num_updates - 1) and self.output_dir != "":
                torch.save(self.actor_critic, os.path.join(self.output_dir, "actor_critic.pt"))
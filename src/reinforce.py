import random
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import numpy as np


class ReinforceReward:
    def __init__(self, config, rewards, rewards_single):
        # RL params
        self.sample_size = config.rl_sample_size
        self.epsilon = config.epsilon
        self.max_num_sents = config.max_num_sents

        # Reward dictionaries
        self.rewards = rewards
        self.single_rewards = rewards_single

        self.kl_method = config.kl_method

    def train(self, probs, doc_id):
        if self.kl_method == "none":
            self.update_data_instance(probs, doc_id)
            batch_index_and_loss_lists = [self.compute_summary_index_and_loss("sample") for _ in range(self.sample_size)]
            batch_rewards = [self.compute_reward(idx_list[0]) for idx_list in batch_index_and_loss_lists]
            baseline_reward = self.compute_baseline()
            loss = self.compute_batch_loss(batch_index_and_loss_lists, batch_rewards, baseline_reward)
            return loss, baseline_reward
        elif self.kl_method == "kl":
            self.update_data_instance(probs, doc_id)
            pred_dist = Categorical(self.probs.view(-1))
            reward_dist_vals = list(self.curr_single_rewards.values())[:probs.view(-1).shape[0]]
            reward_dist = Categorical(torch.tensor(reward_dist_vals, device=self.probs.device))
            return kl_divergence(reward_dist, pred_dist).unsqueeze(0)
        else:
            raise ValueError("Invalid KL method selected.")

    def greedy_summarize(self, probs):
        self.update_data_instance(probs)
        if probs.view(-1).size(0) < self.max_num_sents:
            return np.array(list(range(probs.view(-1).size(0))))
        return self.compute_summary_index_and_loss("greedy")[0]  # Just return indices so we can write

    def update_data_instance(self, probs, doc_id=None):
        self.probs = torch.clamp(probs, 1e-6, 1 - 1e-6)  # Ensure probs are not 0 or 1
        if doc_id is not None:
            self.curr_rewards = self.rewards[doc_id]
            self.curr_single_rewards = self.single_rewards[doc_id]

    def compute_summary_index_and_loss(self, sample_method):
        if sample_method == "sample":
            probs = self.probs.squeeze()
            if len(probs.size()) == 0: probs = probs.unsqueeze(0)
            assert len(probs.size()) == 1

            summary_index = []
            mask = torch.ones(probs.size(), requires_grad=False).cuda()
            loss_list = []
            for i in range(self.max_num_sents):
                p_masked = probs * mask
                if random.uniform(0, 1) <= self.epsilon:  # explore
                    selected_idx = torch.multinomial(mask, 1)
                else:
                    selected_idx = torch.multinomial(p_masked, 1)
                loss_i = (self.epsilon / mask.sum() + (1 - self.epsilon) * p_masked[selected_idx] / p_masked.sum()).log()
                loss_list.append(loss_i)
                mask = mask.clone()
                mask[selected_idx] = 0
                summary_index.append(selected_idx)
            summary_index = torch.cat(summary_index, dim=0)
            loss = sum(loss_list)
        elif sample_method == "greedy":
            loss = 0
            _, summary_index = torch.topk(self.probs.view(1, -1), k=self.max_num_sents, sorted=False)
        else:
            raise ValueError("Invalid sample method selected for policy gradient loss")

        summary_index, _ = torch.sort(summary_index.squeeze())
        return summary_index.cpu().numpy(), loss

    def compute_reward(self, summary_indices):
        return self.curr_rewards[tuple(summary_indices)]

    def compute_baseline(self):
        greedy_indices, _ = self.compute_summary_index_and_loss("greedy")
        return self.compute_reward(greedy_indices)

    def compute_batch_loss(self, batch_index_and_loss_lists, batch_rewards, baseline_reward):
        loss_list = [
            batch_index_and_loss_lists[i][1] * (baseline_reward - batch_rewards[i])
            for i in range(len(batch_rewards))
        ]
        avg_loss = sum(loss_list) / (float(len(loss_list)) + 1e-8)
        return avg_loss

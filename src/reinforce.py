import random
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from scipy.stats import zscore

import os
from helper import sent_tok

def return_summary_index(probs, sample_method="greedy", max_num_of_sents=3, epsilon=0.1, z_thresh=1.0):
    """
    :param probs: torch tensor of the probablities for all sentences in the doc
    :param sample_method: greedy or sample
    :param max_num_of_sents: max num of sents to be selected
    :return: a list of index for the selected sents
    """
    if sample_method == "sample":
        probs = probs.squeeze()
        if len(probs.size()) == 0: probs = probs.unsqueeze(0)
        assert len(probs.size()) == 1

        # Herke's method
        summary_index = []
        mask = torch.ones(probs.size(), requires_grad=False).cuda()
        loss_list = []
        for i in range(max_num_of_sents):
            p_masked = probs * mask
            if random.uniform(0, 1) <= epsilon:  # explore
                selected_idx = torch.multinomial(mask, 1)
            else:
                selected_idx = torch.multinomial(p_masked, 1)
            loss_i = (epsilon / mask.sum() + (1 - epsilon) * p_masked[selected_idx] / p_masked.sum()).log()
            loss_list.append(loss_i)
            mask = mask.clone()
            mask[selected_idx] = 0
            summary_index.append(selected_idx)
        summary_index = torch.cat(summary_index, dim=0)
        loss = sum(loss_list)
    elif sample_method == "z_score_threshold":
        loss = 0
        z_scores = zscore(probs.cpu().numpy())
        top_vals, summary_index = torch.topk(probs.view(1, -1), k=max_num_of_sents, sorted=True)
        summary_index.squeeze_()
        top_aff = summary_index[0].item()
        last_aff = summary_index[-1].item()
        if z_scores[top_aff] - z_scores[last_aff] > z_thresh:
            summary_index = summary_index[:-1]
    else:  # Greedy
        loss = 0
        _, summary_index = torch.topk(probs.view(1, -1), k=max_num_of_sents, sorted=False)

    summary_index, _ = torch.sort(summary_index.squeeze())
    return summary_index.cpu().numpy(), loss


class ReinforceReward(object):
    def __init__(self, reward_dict, std_rouge=False, rouge_metric="all", b=20, rl_baseline_method="greedy", max_num_of_sents=3,
                 single_reward_dict=None, beta=1.0, kl_method='none', epsilon=0.1, pair_reward_dict=None, use_z=False, z_thresh=3.0):
        """
        :param std_rouge:
        :param rouge_metric:
        :param b:
        :param rl_baseline: "greedy", "global_avg","batch_avg", None
        """
        self.probs_torch = None
        self.doc = None

        self.global_avg_reward = 0.
        self.train_examples_seen = 0.

        self.std_rouge = std_rouge
        self.doc_write_count = 0

        self.rouge_metric = rouge_metric
        self.rl_baseline_method = rl_baseline_method
        self.b = b  # batch_size
        self.epsilon = epsilon # exploration parameter
        self.rewards = reward_dict

        self.max_num_of_sents = max_num_of_sents

        self.kl_method = kl_method
        self.single_reward_dict = single_reward_dict
        self.beta = beta

        self.z_cutoff_count = 0
        self.z_score_decode = use_z
        self.pair_reward_dict = pair_reward_dict
        self.z_thresh = z_thresh


    def train_kl(self, probs, doc_id):
        self.update_data_instance(probs, doc_id)
        greedy_index_list, _ = self.generate_index_list_and_loss("greedy")
        greedy_reward = self.generate_reward(greedy_index_list)
        loss = self.kl_div_loss()
        return self.beta * loss, greedy_reward

    def train_entropy_reg(self, probs, doc_id):
        self.update_data_instance(probs, doc_id)
        pred_dist = Categorical(self.probs_torch.view(-1))
        greedy_index_list, _ = self.generate_index_list_and_loss("greedy")
        greedy_reward = self.generate_reward(greedy_index_list)
        loss = -1*pred_dist.entropy()  # Negate entropy so that we maximize it?
        return self.beta * loss, greedy_reward


    def train(self, probs, doc_id):
        """
        :return: training_loss_of_the current example
        """
        if self.kl_method == "kl_only":
            return self.train_kl(probs, doc_id)
        elif self.kl_method == "entropy_reg":
            return self.train_entropy_reg(probs, doc_id)
        else:  # RL loss
            self.update_data_instance(probs, doc_id)
            self.train_examples_seen += 1
            batch_index_and_loss_lists = self.sample_batch(self.b)
            batch_rewards = [
                self.generate_reward(idx_list[0])
                for idx_list in batch_index_and_loss_lists
            ]
            rl_baseline_reward = self.compute_baseline(batch_rewards)
            loss = self.generate_batch_loss(batch_index_and_loss_lists, batch_rewards, rl_baseline_reward)

            greedy_index_list, _ = self.generate_index_list_and_loss("greedy")
            greedy_reward = self.generate_reward(greedy_index_list)

            return loss, greedy_reward

    def kl_div_loss(self):
        pred_dist = Categorical(self.probs_torch.view(-1))
        reward_dist = Categorical(torch.tensor(list(self.curr_single_r.values())[:100], device=self.probs_torch.device))
        return kl_divergence(reward_dist, pred_dist).unsqueeze(0)

    def validate(self, probs, doc_id, full=False):
        """
        :return: training_loss_of_the current example
        """
        self.update_data_instance(probs, doc_id, full)
        if self.z_score_decode:
            summary_index_list, _ = self.generate_index_list_and_loss("z_score_threshold")
            if summary_index_list.shape[0] == 2:
                self.z_cutoff_count += 1
        else:
            summary_index_list, _ = self.generate_index_list_and_loss("greedy")

        lead_index_list, _ = self.generate_index_list_and_loss("lead3")

        greedy_reward = self.generate_reward(summary_index_list)
        lead_reward = self.generate_reward(lead_index_list)

        # Record KL loss
        pred_dist = Categorical(self.probs_torch.view(-1))
        reward_dist = Categorical(torch.tensor(list(self.curr_single_r.values())[:100], device=self.probs_torch.device))
        kl_loss = kl_divergence(reward_dist, pred_dist)

        return greedy_reward, lead_reward, kl_loss.item()

    def validate_std(self, probs, corenlp, doc):
        # Get greedy sentences
        self.update_data_instance(probs)
        num_sents = min(self.max_num_of_sents, probs.size(0))

        summary_index_list,_ = self.generate_index_list_and_loss("greedy", max_sents=num_sents)

        # Construct hypothesis
        sents = sent_tok(corenlp, doc)
        hypothesis = "\n".join([sents[i] for i in summary_index_list])

        # Write doc
        hyp_file = 'hyp.' + str(self.doc_write_count).rjust(5, '0') + '.txt'
        with open(os.path.join('result', 'model', hyp_file), 'w') as f:
            f.write(hypothesis)

        self.doc_write_count += 1
        return summary_index_list


    def update_data_instance(self, probs, doc_id=None, full=False):
        # self.probs_torch = probs
        # self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero
        self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero

        if doc_id is not None:
            reward_idx = doc_id if full else 'x{}'.format(doc_id)
            self.curr_rewards = self.rewards[reward_idx]
            self.curr_single_r = self.single_reward_dict[doc_id]

            if self.z_score_decode:
                self.curr_pair_r = self.pair_reward_dict[doc_id]

    def generate_index_list_and_loss(self, sample_method, max_sents=3):
        """
        :param sample_method: "lead3,leadk,sample, greedy"
        :return: return a list of sentence indexes for next step of computation
        """
        if sample_method == "lead3":
            return range(self.max_num_of_sents), 0
        else:  # either "sample" or "greedy" based on the prob_list
            return return_summary_index(self.probs_torch, sample_method, max_sents, self.epsilon, z_thresh=self.z_thresh)

    def generate_reward(self, summary_index_list):
        if isinstance(summary_index_list, np.ndarray) and summary_index_list.shape[0] == 2:
            return self.curr_pair_r[tuple(summary_index_list)]
        else:
            return self.curr_rewards[tuple(summary_index_list)]

    def generate_summary(self, summary_index_list):
        return [self.doc.content[i] for i in summary_index_list]

    def sample_batch(self, b):
        batch_index_and_loss_lists = [self.generate_index_list_and_loss("sample") for _ in range(b)]
        return batch_index_and_loss_lists

    def compute_baseline(self, batch_rewards):
        def running_avg(t, old_mean, new_score):
            return (t - 1) / t * old_mean + new_score / t

        batch_avg_reward = np.mean(batch_rewards)
        self.global_avg_reward = running_avg(self.train_examples_seen, self.global_avg_reward, batch_avg_reward)

        if self.rl_baseline_method == "batch_avg":
            return batch_avg_reward
        elif self.rl_baseline_method == "global_avg":
            return self.global_avg_reward
        elif self.rl_baseline_method == "greedy":
            summary_index_list, _ = self.generate_index_list_and_loss("greedy")
            return self.generate_reward(summary_index_list)
        else:  # none
            return 0

    def generate_batch_loss(self, batch_index_and_loss_lists, batch_rewards, rl_baseline_reward):
        loss_list = [
            batch_index_and_loss_lists[i][1] * (rl_baseline_reward - batch_rewards[i])
            for i in range(len(batch_rewards))
        ]
        avg_loss = sum(loss_list) / (float(len(loss_list)) + 1e-8)
        return avg_loss

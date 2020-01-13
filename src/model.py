# coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from helper import strip_zeros
from reinforce import ReinforceReward
import numpy as np
torch.manual_seed(233)


class SummaRuNNer(nn.Module):
    def __init__(self, config):
        super(SummaRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(400, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x): # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data #ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0] # ex. N sentes
        #print("seq_num", sequence_length)
        # word level LSTM
        word_features = self.word_embedding(x) # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features) #output: word_outputs (N,W,h)

        # sentence level LSTM
        sent_features = self._avg_pooling(word_outputs, sequence_length) #output:(N,h)
        sent_outputs, _ = self.sent_LSTM(sent_features.view(1, -1, self.sent_input_size)) #input (1,N,h)
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)]) #output:(1,h)
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.view(-1, 2 * self.sent_LSTM_hidden_units)

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)


class GruRuNNer(nn.Module):
    def __init__(self, config):
        super(GruRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_GRU_hidden_units = config.word_GRU_hidden_units
        self.sent_GRU_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_GRU = nn.GRU(
            input_size=self.word_input_size,
            hidden_size=self.word_GRU_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_GRU = nn.GRU(
            input_size=self.sent_input_size,
            hidden_size=self.sent_GRU_hidden_units,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(400, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):
        sequence_length = torch.sum(torch.sign(x), dim=1).data
        sequence_num = sequence_length.size()[0]

        # word level GRU
        word_features = self.word_embedding(x)
        word_outputs, _ = self.word_GRU(word_features)
        # sentence level GRU
        sent_features = self._avg_pooling(word_outputs, sequence_length)
        sent_outputs, _ = self.sent_GRU(sent_features.view(1, -1, self.sent_input_size))
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)])
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.view(-1, 2 * self.sent_GRU_hidden_units)

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)



class SimpleRNN(nn.Module):
    def __init__(self, config, reward_train, reward_dev, reward_train_single, reward_dev_single, reward_dev_pairs=None, corenlp=None):
        super(SimpleRNN, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        # self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # self.encoder = nn.Sequential(nn.Linear(800, 400), nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(400, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

        # Computing loss and rewards
        self.reinforce_train = ReinforceReward(reward_dict=reward_train, std_rouge=config.std_rouge, rouge_metric=config.rouge_metric,
                                               b=config.sample_batch_size, rl_baseline_method=config.rl_baseline_method,
                                               single_reward_dict=reward_train_single,
                                               beta=config.kl_weight, kl_method=config.kl_method, epsilon=config.epsilon)
        self.reinforce_eval = ReinforceReward(reward_dict=reward_dev, std_rouge=config.std_rouge, rouge_metric=config.rouge_metric,
                                               b=config.sample_batch_size, rl_baseline_method=config.rl_baseline_method,
                                              single_reward_dict=reward_dev_single,
                                              beta=config.kl_weight, kl_method=config.kl_method,
                                              pair_reward_dict=reward_dev_pairs, use_z=config.use_z, z_thresh=config.z_thresh)
        self.reinforce = self.reinforce_train
        self.full_rewards = config.full_rewards
        self.std_rouge = config.std_rouge
        self.corenlp = corenlp

    def update_doc_id(self, doc_id):
        self.doc_id = doc_id

    def update_doc(self, doc):
        self.doc = doc

    def set_reinforce(self, config, reward_train, reward_train_single, reward_dev, reward_dev_single):
        self.reinforce_train = ReinforceReward(reward_dict=reward_train, std_rouge=config.std_rouge,
                                               rouge_metric=config.rouge_metric,
                                               b=config.sample_batch_size, rl_baseline_method=config.rl_baseline_method,
                                               single_reward_dict=reward_train_single,
                                               beta=config.kl_weight, kl_method=config.kl_method,
                                               epsilon=config.epsilon)
        self.reinforce_eval = ReinforceReward(reward_dict=reward_dev, std_rouge=config.std_rouge,
                                              rouge_metric=config.rouge_metric,
                                              b=config.sample_batch_size, rl_baseline_method=config.rl_baseline_method,
                                              single_reward_dict=reward_dev_single,
                                              beta=config.kl_weight, kl_method=config.kl_method,
                                              pair_reward_dict=None, use_z=config.use_z,
                                              z_thresh=config.z_thresh)
        self.reinforce = self.reinforce_train
        self.full_rewards = config.full_rewards
        self.std_rouge = config.std_rouge


    def update_aux_params_and_hook(self, cosines, aux_grads):
        self.cosines = cosines
        self.aux_grads = aux_grads

        hooks = []
        for name, param in self.named_parameters():
            hook = param.register_hook(lambda main_grad: self._cos_update(name, main_grad))
            hooks.append(hook)
        return hooks

    def halve_grads(self):
        hooks = []
        for param in self.parameters():
            hook = param.register_hook(lambda grad: (0.5)*grad)
            hooks.append(hook)
        return hooks


    def _cos_update(self, name, grad):
        param_aux_grads = self.aux_grads[name] # List of auxiliary gradients for this specific parameter
        valid_count = 0.
        acc_grad = torch.zeros(param_aux_grads[0].shape, device=param_aux_grads[0].device)
        for cos, aux_grad in zip(self.cosines, param_aux_grads):
            if cos > 0:
                valid_count += 1.
                acc_grad.add_(cos*aux_grad)
        grad_weight = 1 / (valid_count + 1.)
        return torch.add(input=grad, alpha=grad_weight, other=acc_grad)


    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]]
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sents

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num, self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)

        prob = self.decoder(enc_output)
        outputs = prob.view(sequence_num, 1)

        if self.training:
            loss, reward = self.reinforce.train(outputs, self.doc_id)
        else:
            if self.std_rouge:
                return self.reinforce.validate_std(outputs, self.corenlp, self.doc)
            else:
                return self.reinforce.validate(outputs, self.doc_id, self.full_rewards)
        return loss, reward



    def eval(self):
        super(SimpleRNN, self).eval()
        self.reinforce = self.reinforce_eval

    def train(self, mode=True):
        super(SimpleRNN, self).train(mode=mode)
        self.reinforce = self.reinforce_train

class SimpleRuNNer(nn.Module):
    def __init__(self, config):
        super(SimpleRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x): # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data #ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0] # ex. N sentes
        #print("seq_num", sequence_length)
        # word level LSTM
        word_outputs = self.word_embedding(x) # Input: LongTensor (N, W), Output: (N, W, embedding_dim) (49*30*100)
        sent_features = self._avg_pooling(word_outputs, sequence_length)  #output N*h
        sent_outputs = sent_features.view(1, sequence_num,-1) #output:(N,h) (49*100)
        #sent_outputs  = sent_features.unsqueenze(1, -1)) #input (1,N,h)
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)]) #output:(1,h)
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.squeeze()

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)

class RNES(nn.Module):
    def __init__(self, config):
        super(RNES, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.out_channel = 50  # args.kernel_num
        self.kernel_sizes = range(0, 8)  # args.kernel_sizes[1,2,...,7]
        self.hidden_state = 400
        self.sent_input_size = 400

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.conv = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.out_channel,
                                             K * 2 + 1, padding=K) for K in self.kernel_sizes])

        # self.dropout = nn.Dropout(args.dropout)
        # reverse order LSTM
        self.sent_GRU = nn.GRU(
            input_size=self.sent_input_size,
            hidden_size=self.hidden_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.doc_encoder = nn.Sequential(nn.Linear(self.hidden_state * 2, self.hidden_state),
                                         nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(self.hidden_state * 4, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

        self.redundancy = nn.Sequential(nn.Linear(self.hidden_state * 2, self.hidden_state),
                                        nn.Tanh())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x, num_of_sent=3, greedy=False):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        conv_input = word_features.transpose(1, 2)
        sent_features_list = []
        for i in self.kernel_sizes:
            sent_features_list.append(self.conv[i](conv_input))
        sent_features = torch.cat(sent_features_list, dim=1).transpose(1, 2)

        sent_features = self._avg_pooling(sent_features, sequence_length).view(1, sequence_num,
                                                                               self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_GRU(sent_features)
        enc_output = enc_output.squeeze(0)

        doc_features = self.doc_encoder(enc_output.mean(dim=0))

        g = Variable(doc_features.data.new(self.hidden_state).zero_())

        prob_list = []
        sample_list = []
        for i in range(sequence_num):
            prob_i = self.decoder(torch.cat([enc_output[i], g, doc_features], dim=-1))

            if num_of_sent <= 0 or len(sample_list) < num_of_sent:
                if greedy:
                    sample_i = (prob_i > 0.5).float().data.cpu().numpy()[0]
                else:
                    sample_i = prob_i.bernoulli().data.cpu().numpy()[0]
            else:
                sample_i = 0

            if sample_i == 1:
                prob_list.append(prob_i)
                sample_list.append(i)
                g += self.redundancy(enc_output[i])
            else:
                prob_list.append(1 - prob_i)

        prob = torch.cat(prob_list, dim=0).squeeze() * 0.99999 + 0.000005
        logp = prob.log().sum()

        return logp, sample_list
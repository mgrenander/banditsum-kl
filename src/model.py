import torch
import torch.nn as nn
from reinforce import ReinforceReward
torch.manual_seed(233)


class SimpleRNN(nn.Module):
    def __init__(self, config, rewards):
        super(SimpleRNN, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_LSTM_hidden_units
        self.sent_LSTM_hidden_units = config.sent_LSTM_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

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

        self.decoder = nn.Sequential(nn.Linear(400, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

        # Computing loss and rewards
        self.reinforce_train = ReinforceReward(config, rewards['train'], rewards['train_single'])
        self.reinforce_eval = ReinforceReward(config, rewards['dev'], rewards['dev_single'])
        self.reinforce = self.reinforce_train

    def update_doc_id(self, doc_id):
        self.doc_id = doc_id

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
            return self.reinforce.train(outputs, self.doc_id)
        else:
            return self.reinforce.greedy_summarize(outputs)

    def eval(self):
        super(SimpleRNN, self).eval()
        self.reinforce = self.reinforce_eval

    def train(self, mode=True):
        super(SimpleRNN, self).train(mode=mode)
        self.reinforce = self.reinforce_train

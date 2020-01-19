import pickle
import numpy as np
import re
import torch

REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')

def process_text(tok_client, text):
    ann = tok_client.annotate(text)
    text_sents = []
    for i in range(len(ann.sentence)):
        ann_text = [ann.sentence[i].token[j].word for j in range(len(ann.sentence[i].token))]
        ann_text = strip_punc(ann_text)
        if ann_text:  # The above process may result in empty lists which we don't want
            text_sents.append(ann_text)
    return text_sents


def strip_punc(tokens):
    stripped = []
    for t in tokens:
        if not REMOVE_CHAR_PATTERN.match(t):
            stripped += re.sub(REMOVE_CHAR_PATTERN, " ", t.lower()).split()
    return stripped


def convert_tokens_to_ids(doc, args):
    max_len = len(max(doc, key=lambda x: len(x)))
    sent_list = []
    for i in range(len(doc)):
        words = doc[i]
        sent = [args.word2id[word] if word in args.word2id else 1 for word in words]
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        sent_list.append(sent)
    return torch.tensor(sent_list).long()


def config(args, vocab):
    args.vocab_size = vocab.embedding.shape[0],
    args.embedding_dim = vocab.embedding.shape[1],
    args.position_size = 500,
    args.position_dim = 50,
    args.word_input_size = 100,
    args.sent_input_size = 2 * args.hidden,
    args.word_LSTM_hidden_units = args.hidden,
    args.sent_LSTM_hidden_units = args.hidden,
    args.pretrained_embedding = vocab.embedding,
    args.word2id = vocab.w2i,
    args.id2word = vocab.i2w


class Vocab():
    def __init__(self):
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab(self, vocab_file="../data/vocab/raw_vocab.txt"):
        with open(vocab_file, "r") as f:
            for line in f:
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("Read {} words from vocab file".format(len(self.word_list)))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile="../data/glove.6B/glove.6B.100d.txt", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("Processed {} data".format(len(model)))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file".format((len(model), len(self.word_list))))

    def pickle_vocab(self, save_path="../data/vocab/vocab_100d.p"):
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

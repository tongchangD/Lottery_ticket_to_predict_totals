import random
import os
import numpy as np
# EOS_token = 1

def get_couplets(data_dir):
    print("Reading lines...")
    in_lines = open(os.path.join(data_dir, "in.txt"),
                    encoding="UTF-8").read().strip().split('\n')
    out_lines = open(os.path.join(data_dir, "out.txt"),
                     encoding="UTF-8").read().strip().split('\n')
    #assert len(in_lines) == len(out_lines) #, "上下联长度不匹配" 对对联需要上下句 词语相同

    pairs = []
    for i, in_line in enumerate(in_lines):
        pairs.append([in_line, out_lines[i]])
    return pairs

class Language:
    def __init__(self, vocab_file):
        self.index2word = self.read_vocab(vocab_file)
        self.word2index = dict((c, i) for i, c in enumerate(self.index2word))
        self.n_words = len(self.index2word)  # Count SOS and EOS

    def read_vocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            vocabs = [line.decode('utf8').strip() for line in f]
        return vocabs

class PairsLoader():
    def __init__(self, language, pairs, batch_size, max_length,target_length):
        self.language = language
        self.pairs = pairs
        self.batch_size = batch_size
        self.max_length = max_length
        self.target_length=target_length
        self.position = 0

    def load_single_pair(self):
        if self.position >= len(self.pairs):
            random.shuffle(self.pairs)
            self.position = 0
        single_pair = self.pairs[self.position]
        self.position += 11
        return single_pair

    def indexes_from_sentence(self, sentence):
        # indexes = [self.language.word2index[word] if word != "" and word in self.language.word2index.keys() else self.language.word2index[' '] for word in sentence.split(' ') ]
        # indexes = [
        #     self.language.word2index[word] if word in self.language.word2index.keys() else self.language.n_words+1 for word in sentence.split(' ')]
        indexes = [
            self.language.word2index[word] if word in self.language.word2index.keys() else self.language.n_words + 1 for
            word in sentence.split(' ')]
        for word in sentence.split(' '):
            if word not in self.language.word2index.keys():
                print(sentence)

        if len(indexes) > self.max_length:
            indexes = indexes[:self.max_length]
        # indexes.append(EOS_token)
        return indexes

    def indexes_from_pair(self, pair):
        input_indexes = self.indexes_from_sentence(pair[0])
        output_indexes = self.indexes_from_sentence(pair[1])
        return input_indexes, output_indexes

    def pad_train_sentence(self, sentence):
        # pad on the right
        results = [0 for i in range(self.max_length)]
        for i in range(len(sentence)):
            results[i] = sentence[i]
        return results

    def pad_traget_sentence(self, sentence):
        # pad on the right
        results = [0 for i in range(self.target_length)]
        for i in range(len(sentence)):
            results[i] = sentence[i]
        return results

    def load(self):
        while True:
            input_batch = []
            output_batch = []
            # mask_batch = np.zeros([self.batch_size, self.max_length+1])  # tcd
            mask_batch = np.zeros([self.batch_size, self.target_length]) #
            for i in range(self.batch_size):
                pair = self.load_single_pair()
                input_indexes, output_indexes = self.indexes_from_pair(pair)
                input_batch.append(self.pad_train_sentence(input_indexes))
                output_batch.append(self.pad_traget_sentence(output_indexes))

            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, np.array(output_batch))))

            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            # print(input_batch, output_batch, mask_batch)
            yield input_batch, output_batch, mask_batch

    def load_evl(self):
        input_batch = []
        output_batch = []
        mask_batch = np.zeros([len(self.pairs), self.target_length])  #
        for i in range(len(self.pairs)):
            pair = self.load_single_pair()
            input_indexes, output_indexes = self.indexes_from_pair(pair)
            input_batch.append(self.pad_train_sentence(input_indexes))
            output_batch.append(self.pad_traget_sentence(output_indexes))
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, np.array(output_batch))))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        return input_batch, output_batch, mask_batch

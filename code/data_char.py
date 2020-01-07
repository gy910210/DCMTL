import codecs
import numpy as np
import os


class DataLoader(object):

    def __init__(self, max_seq_len, vocab_index_dict,
                 seg_label_index_dict=None,
                 coarse_label_index_dict=None,
                 fine_label_index_dict=None):

        self.max_seq_len = max_seq_len
        self.vocab_index_dict = vocab_index_dict
        self.seg_label_index_dict = seg_label_index_dict
        self.coarse_label_index_dict = coarse_label_index_dict
        self.fine_label_index_dict = fine_label_index_dict
        self.data = list()
        self.data_size = 0

    def decode_line(self, line):
        spt = line.strip().split("\t")
        input, seg_label, coarse_label, fine_label = spt[0], spt[1], spt[2], spt[3]
        # default = 0 means not found
        input_idx = [self.vocab_index_dict.get(term, 0) for term in input.strip().split(" ")]

        if self.seg_label_index_dict is None:
            seg_label = [int(i) for i in seg_label.strip().split(" ")]
        else:
            seg_label = [self.seg_label_index_dict[term] for term in seg_label.strip().split(" ")]

        if self.coarse_label_index_dict is None:
            coarse_label = [int(i) for i in coarse_label.strip().split(" ")]
        else:
            coarse_label = [self.coarse_label_index_dict[term] for term in coarse_label.strip().split(" ")]

        if self.fine_label_index_dict is None:
            fine_label = [int(i) for i in fine_label.strip().split(" ")]
        else:
            fine_label = [self.fine_label_index_dict[term] for term in fine_label.strip().split(" ")]

        # actual length of input
        input_len = len(input_idx)
        # padding
        if input_len < self.max_seq_len:
            for _ in range(self.max_seq_len - input_len):
                input_idx.append(0)  # the last one in vocab is zero-vec
                seg_label.append(0)  # O tag
                coarse_label.append(0)  # O tag
                fine_label.append(0)  # O tag

        if input_len > self.max_seq_len:
            input_idx = input_idx[0:self.max_seq_len]
            seg_label = seg_label[0:self.max_seq_len]
            coarse_label = coarse_label[0:self.max_seq_len]
            fine_label = fine_label[0:self.max_seq_len]
            input_len = self.max_seq_len

        return input_idx, input_len, seg_label, coarse_label, fine_label


    def load(self, data_file, encoding):
        if os.path.isfile(data_file):
            input_idxs, input_lens, seg_labels, coarse_labels, fine_labels = list(), list(), list(), list(), list()
            cnt = 0
            with codecs.open(data_file, 'r', encoding=encoding) as fin:
                for line in fin:
                    spt = line.strip().split("\t")
                    input_idxs.append([int(idx) for idx in spt[0].split(" ")])
                    input_lens.append(int(spt[1]))
                    seg_labels.append([int(idx) for idx in spt[2].split(" ")])
                    coarse_labels.append([int(idx) for idx in spt[3].split(" ")])
                    fine_labels.append([int(idx) for idx in spt[4].split(" ")])
                    cnt += 1
        else:
            txt_data_file = data_file + ".name"
            input_idxs, input_lens, seg_labels, coarse_labels, fine_labels = list(), list(), list(), list(), list()
            cnt = 0
            fout = codecs.open(data_file, 'w', encoding=encoding)
            with codecs.open(txt_data_file, 'r', encoding=encoding) as fin:
                for line in fin:
                    input_idx, input_len, seg_label, coarse_label, fine_label = self.decode_line(line)
                    fout.write(" ".join([str(x) for x in input_idx]) + "\t" +
                               str(input_len) + "\t" +
                               " ".join([str(x) for x in seg_label]) + "\t" +
                               " ".join([str(x) for x in coarse_label]) + "\t" +
                               " ".join([str(x) for x in fine_label]) + "\n")
                    input_idxs.append(input_idx)
                    input_lens.append(input_len)
                    seg_labels.append(seg_label)
                    coarse_labels.append(seg_label)
                    fine_labels.append(fine_label)
                    cnt += 1
            fout.close()
        self.data = list(zip(input_idxs, input_lens, seg_labels, coarse_labels, fine_labels))
        self.data_size = len(self.data)


class BatchGenerator(object):

    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.data_size = len(self.data)
        self.pointer = 0
        self.num_batches = 0
        self.create_batches()

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_batches(self):
        if self.data_size % self.batch_size == 0:
            self.num_batches = int(self.data_size / self.batch_size)
        else:
            self.num_batches = int(self.data_size / self.batch_size) + 1

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

    def next_batch(self):
        if self.pointer + self.batch_size > self.data_size:
            input_idx, input_len, seg_label, coarse_label, fine_label =\
                zip(*self.data[self.pointer: self.data_size])
            self.pointer = self.data_size
        else:
            input_idx, input_len, seg_label, coarse_label, fine_label = \
                zip(*self.data[self.pointer: self.pointer + self.batch_size])
            self.pointer += self.batch_size

        input_idx = np.array(input_idx)
        input_len = np.array(input_len)
        seg_label = np.array(seg_label)
        coarse_label = np.array(coarse_label)
        fine_label = np.array(fine_label)

        return input_idx, input_len, seg_label, coarse_label, fine_label


## no embeddings
class VocabularyLoader(object):
    def __init__(self):
        self.vocab_size = 0
        self.index_vocab_dict = []
        self.vocab_index_dict = dict()
        self.vocab_embedding = []

    def load_vocab(self, vocab_file, embedding_dim, encoding):
        self.vocab_size = 0
        self.index_vocab_dict = []
        self.vocab_index_dict.clear()
        self.vocab_embedding = []
        with codecs.open(vocab_file, 'r', encoding=encoding) as fin:
            index = 0
            # 0 embedding for not-found query term
            self.vocab_index_dict["NULL"] = index
            self.index_vocab_dict.append("NULL")
            self.vocab_embedding.append([0.0 for _ in range(embedding_dim)])
            index += 1
            for line in fin:
                term = line.strip()
                self.vocab_index_dict[term] = index
                self.index_vocab_dict.append(term)
                embedding = [0.0 for _ in range(embedding_dim)]
                self.vocab_embedding.append(embedding)
                index += 1
        self.vocab_size = len(self.vocab_embedding)
        self.vocab_embedding = np.array(self.vocab_embedding)


'''
## has embeddings
class VocabularyLoader(object):
    def __init__(self):
        self.vocab_size = 0
        self.index_vocab_dict = []
        self.vocab_index_dict = dict()
        self.vocab_embedding = []

    def load_vocab(self, vocab_file, embedding_dim, encoding):
        self.vocab_size = 0
        self.index_vocab_dict = []
        self.vocab_index_dict.clear()
        self.vocab_embedding = []
        with codecs.open(vocab_file, 'r', encoding=encoding) as fin:
            index = 0
            # 0 embedding for not-found query term
            self.vocab_index_dict["NULL"] = index
            self.index_vocab_dict.append("NULL")
            self.vocab_embedding.append([0.0 for _ in range(embedding_dim)])
            index += 1
            for line in fin:
                spt = line.strip().split()
                if len(spt) != embedding_dim + 1:
                    continue
                self.vocab_index_dict[spt[0]] = index
                self.index_vocab_dict.append(spt[0])
                embedding = [float(spt[i].strip()) for i in range(1, len(spt))]
                self.vocab_embedding.append(embedding)
                index += 1
        self.vocab_size = len(self.vocab_embedding)
        self.vocab_embedding = np.array(self.vocab_embedding)
'''

class LabelLoader(object):
    def __init__(self):
        self.label_size = 0
        self.index_label_dict = []
        self.label_index_dict = dict()

    def load_label(self, label_file, encoding):
        self.label_size = 0
        self.index_label_dict = []
        self.label_index_dict = dict()

        with codecs.open(label_file, 'r', encoding=encoding) as fin:
            index = 0
            for line in fin:
                line = line.strip()
                self.label_index_dict[line] = index
                self.index_label_dict.append(line)
                index += 1

        self.label_size = len(self.index_label_dict)
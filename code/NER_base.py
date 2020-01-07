import tensorflow as tf
import numpy as np
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class NER(object):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 word_dim,
                 lstm_dim,
                 max_seq_len,
                 seg_num_classes,
                 coarse_num_classes,
                 fine_num_classes,
                 l2_reg_lambda=0.0,
                 lr=0.001,
                 gradient_clip=5,
                 init_word_embedding=None,
                 layer_size=1,
                 is_multi_task=False):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.max_seq_len = max_seq_len
        ###############################
        self.seg_num_classes = seg_num_classes
        self.coarse_num_classes = coarse_num_classes
        self.fine_num_classes = fine_num_classes
        ###############################
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.layer_size = layer_size
        ###############################
        self.is_multi_task = is_multi_task

        if init_word_embedding is None:
            self.init_word_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_word_embedding = init_word_embedding

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None], name="x")
        self.y_seg = tf.placeholder(tf.int32, [None, None], name="y_seg")
        self.y_coarse = tf.placeholder(tf.int32, [None, None], name="y_coarse")
        self.y_fine = tf.placeholder(tf.int32, [None, None], name="y_fine")
        ####################

        self.seq_len = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self._build_graph()
        self.saver = tf.train.Saver()

    @abc.abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def train_step(self, sess, x_batch, y_seg_batch, y_coarse_batch, y_fine_batch, seq_len_batch, dropout_keep_prob):
        feed_dict = {
            self.x: x_batch,
            self.y_seg: y_seg_batch,
            self.y_coarse: y_coarse_batch,
            self.y_fine: y_fine_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [self.train_op, self.global_step, self.loss],
            feed_dict)

        return step, loss

    @abc.abstractmethod
    def decode(self, sess, x, y_seg, y_coarse, y_fine, seq_len):
        raise NotImplementedError

    def save(self, sess, dir_path):
        import os
        if not(os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        fp = dir_path + "/best_model"
        return self.saver.save(sess, fp)

    def load(self, sess, fp):
        self.saver.restore(sess, fp)


def eval_seq_crf_with_o(y_pred_, y_true_, tag_dict, method='precision'):
    """
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    names = set()
    for tag in tag_dict:
        if tag == 'O':
            names.add('O')
        else:
            names.add(tag[2:])

    correct = dict()
    act_cnt = dict()
    for name in names:
        correct[name] = 0
        act_cnt[name] = 0

    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag_num == 0:
                # tags "O"
                kind = 'O'
                act_cnt[kind] += 1
                if line_true[i] == line_pred[i]:
                    correct[kind] += 1
                i += 1
                continue
            else:
                kind = tag[2:]
                sign = tag[0]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[2:] == kind and next_tag[0] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    return overall


def eval_seq_crf_no_o(y_pred_, y_true_, tag_dict, method='precision'):
    """
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    names = set()
    for tag in tag_dict:
        if tag == 'O':
            continue
        else:
            names.add(tag[2:])

    correct = dict()
    act_cnt = dict()
    for name in names:
        correct[name] = 0
        act_cnt[name] = 0

    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag_num == 0:
                # tags "O"
                i += 1
                continue
            else:
                kind = tag[2:]
                sign = tag[0]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[2:] == kind and next_tag[0] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    return overall
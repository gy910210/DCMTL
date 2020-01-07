import tensorflow as tf
from NER_base import NER
import numpy as np


class NER_MTL(NER):
    def _build_graph(self):
        with tf.variable_scope("word_embedding"):
            #self.word_embedding = tf.Variable(self.init_word_embedding, dtype=tf.float32, name="word_embedding")
            self.word_embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)

        with tf.variable_scope("seg_label_embedding"):
            self.seg_label_embedding = tf.constant(np.eye(self.seg_num_classes), dtype=tf.float32)

        with tf.variable_scope("coarse_label_embedding"):
            self.coarse_label_embedding = tf.constant(np.eye(self.coarse_num_classes), dtype=tf.float32)

        with tf.variable_scope("parameters"):
            self.W_cascade_coarse = tf.get_variable(
                shape=[self.coarse_num_classes, self.lstm_dim * 2],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_cascade_coarse",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.W_cascade_seg = tf.get_variable(
                shape=[self.seg_num_classes, self.lstm_dim * 2],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_cascade_seg",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.W_residual = tf.get_variable(
                shape=[self.vocab_size, self.lstm_dim * 2],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_residual",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.W_seg = tf.get_variable(
                shape=[self.lstm_dim * 2, self.seg_num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_seg",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            self.b_seg = tf.Variable(tf.zeros([self.seg_num_classes], dtype=tf.float32, name="bias_seg"))

            self.W_coarse = tf.get_variable(
                shape=[self.lstm_dim * 2, self.coarse_num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_coarse",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            self.b_coarse = tf.Variable(tf.zeros([self.coarse_num_classes], dtype=tf.float32, name="bias_coarse"))

            self.W_fine = tf.get_variable(
                shape=[self.lstm_dim * 2, self.fine_num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights_fine",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            self.b_fine = tf.Variable(tf.zeros([self.fine_num_classes], dtype=tf.float32, name="bias_fine"))

        seq_len = tf.cast(self.seq_len, tf.int64)

        with tf.variable_scope("forward_seg"):
            word_input = tf.nn.embedding_lookup(self.word_embedding, self.x)  # [batch_size, seq_len, word_dim]

            fw_cell_seg = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
            bw_cell_seg = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

            (forward_output_seg, backward_output_seg), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell_seg,
                bw_cell_seg,
                word_input,
                time_major=False,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_forward_seg'
            )
            seg_output = tf.concat(axis=2, values=[forward_output_seg, backward_output_seg])
            seg_output = tf.nn.dropout(seg_output, keep_prob=self.dropout_keep_prob)

            size = tf.shape(seg_output)[0]
            seg_output_flat = tf.reshape(seg_output, [-1, 2 * self.lstm_dim])

            matricized_unary_scores_seg = tf.matmul(seg_output_flat, self.W_seg) + self.b_seg
            self.logits_seg = tf.reshape(matricized_unary_scores_seg, [size, -1, self.seg_num_classes])

        with tf.variable_scope("forward_coarse"):
            seg_label_input = tf.nn.embedding_lookup(self.seg_label_embedding, self.y_seg)  # [batch_size, seq_len, seg_num_classes]
            word_input = tf.nn.embedding_lookup(self.word_embedding, self.x)  # [batch_size, seq_len, word_dim]

            size = tf.shape(word_input)[0]

            word_input_flat = tf.reshape(word_input, [-1, self.vocab_size])
            word_input_flat = tf.matmul(word_input_flat, self.W_residual)
            word_input_flat = tf.reshape(word_input_flat, [size, -1, 2 * self.lstm_dim])

            seg_label_input_flat = tf.reshape(seg_label_input, [-1, self.seg_num_classes])
            seg_label_input_flat = tf.matmul(seg_label_input_flat, self.W_cascade_seg)
            seg_label_input_flat = tf.reshape(seg_label_input_flat, [size, -1, 2 * self.lstm_dim])

            coarse_input = word_input_flat \
                           + seg_label_input_flat \
                           + seg_output

            fw_cell_coarse = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
            bw_cell_coarse = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

            (forward_output_coarse, backward_output_coarse), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell_coarse,
                bw_cell_coarse,
                coarse_input,
                time_major=False,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_forward_coarse'
            )
            coarse_output = tf.concat(axis=2, values=[forward_output_coarse, backward_output_coarse])
            coarse_output = tf.nn.dropout(coarse_output, keep_prob=self.dropout_keep_prob)

            size = tf.shape(coarse_output)[0]
            coarse_output_flat = tf.reshape(coarse_output, [-1, 2 * self.lstm_dim])

            matricized_unary_scores_coarse = tf.matmul(coarse_output_flat, self.W_coarse) + self.b_coarse
            self.logits_coarse = tf.reshape(matricized_unary_scores_coarse, [size, -1, self.coarse_num_classes])

        with tf.variable_scope("forward_fine"):
            coarse_label_input = tf.nn.embedding_lookup(self.coarse_label_embedding, self.y_coarse)  # [batch_size, seq_len, coarse_num_classes]

            size = tf.shape(coarse_label_input)[0]

            coarse_label_input_flat = tf.reshape(coarse_label_input, [-1, self.coarse_num_classes])
            coarse_label_input_flat = tf.matmul(coarse_label_input_flat, self.W_cascade_coarse)
            coarse_label_input_flat = tf.reshape(coarse_label_input_flat, [size, -1, 2 * self.lstm_dim])

            fine_input = coarse_label_input_flat \
                         + coarse_output \
                         + word_input_flat \
                         + seg_output

            fw_cell_fine = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
            bw_cell_fine = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

            (forward_output_fine, backward_output_fine), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell_fine,
                bw_cell_fine,
                fine_input,
                time_major=False,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_forward_fine'
            )
            fine_output = tf.concat(axis=2, values=[forward_output_fine, backward_output_fine])
            fine_output = tf.nn.dropout(fine_output, keep_prob=self.dropout_keep_prob)

            size = tf.shape(fine_output)[0]
            fine_output_flat = tf.reshape(fine_output, [-1, 2 * self.lstm_dim])

            matricized_unary_scores_fine = tf.matmul(fine_output_flat, self.W_fine) + self.b_fine
            self.logits_fine = tf.reshape(matricized_unary_scores_fine, [size, -1, self.fine_num_classes])

        with tf.variable_scope("loss_seg"):
            log_likelihood_seg, self.transition_params_seg = tf.contrib.crf.crf_log_likelihood(
                self.logits_seg, self.y_seg, self.seq_len)

        with tf.variable_scope("loss_coarse"):
            log_likelihood_coarse, self.transition_params_coarse = tf.contrib.crf.crf_log_likelihood(
                self.logits_coarse, self.y_coarse, self.seq_len)

        with tf.variable_scope("loss_fine"):
            log_likelihood_fine, self.transition_params_fine = tf.contrib.crf.crf_log_likelihood(
                self.logits_fine, self.y_fine, self.seq_len)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.variable_scope("train_ops_seg"):
            self.loss_seg = tf.reduce_mean(-log_likelihood_seg)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_seg, tvars), self.gradient_clip)
            self.train_op_seg = self.optimizer.apply_gradients(zip(grads, tvars),
                                                               global_step=self.global_step)

        with tf.variable_scope("train_ops_coarse"):
            self.loss_coarse = tf.reduce_mean(-log_likelihood_coarse)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_coarse, tvars), self.gradient_clip)
            self.train_op_coarse = self.optimizer.apply_gradients(zip(grads, tvars),
                                                                  global_step=self.global_step)

        with tf.variable_scope("train_ops_fine"):
            self.loss_fine = tf.reduce_mean(-log_likelihood_fine)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_fine, tvars), self.gradient_clip)
            self.train_op_fine = self.optimizer.apply_gradients(zip(grads, tvars),
                                                                global_step=self.global_step)

        with tf.variable_scope("train_ops_all"):
            self.loss_all = tf.reduce_mean(-log_likelihood_fine) \
                            + tf.reduce_mean(-log_likelihood_coarse) \
                            + tf.reduce_mean(-log_likelihood_seg)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_all, tvars), self.gradient_clip)
            self.train_op_all = self.optimizer.apply_gradients(zip(grads, tvars),
                                                                global_step=self.global_step)


    def train_step(self, sess, x_batch, y_seg_batch, y_coarse_batch, y_fine_batch, seq_len_batch, dropout_keep_prob,
                   opt_type=None):
        feed_dict = {
            self.x: x_batch,
            self.y_seg: y_seg_batch,
            self.y_coarse: y_coarse_batch,
            self.y_fine: y_fine_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }

        if opt_type == "SEG":
            _, step, loss = sess.run(
                [self.train_op_seg, self.global_step, self.loss_seg],
                feed_dict)
        elif opt_type == "COARSE":
            _, step, loss = sess.run(
                [self.train_op_coarse, self.global_step, self.loss_coarse],
                feed_dict)
        elif opt_type == "FINE":
            _, step, loss = sess.run(
                [self.train_op_fine, self.global_step, self.loss_fine],
                feed_dict)
        elif opt_type == "ALL":
            _, step, loss = sess.run(
                [self.train_op_all, self.global_step, self.loss_all],
                feed_dict)
        else:
            raise Exception("opt_type parameter error!!!")

        return step, loss


    def decode(self, sess, x, y_seg, y_coarse, y_fine, seq_len):
        feed_dict_seg = {
            self.x: x,
            self.seq_len: seq_len,
            self.dropout_keep_prob: 1.0
        }
        logits_seg, transition_params_seg = sess.run(
            [self.logits_seg, self.transition_params_seg], feed_dict_seg)
        y_pred_seg = []
        for logits_, seq_len_ in zip(logits_seg, seq_len):
            logits_ = logits_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logits_, transition_params_seg)
            viterbi_sequence.extend([0 for _ in range(self.max_seq_len - seq_len_)])
            y_pred_seg.append(viterbi_sequence)

        feed_dict_coarse = {
            self.x: x,
            self.y_seg: y_pred_seg,
            self.seq_len: seq_len,
            self.dropout_keep_prob: 1.0
        }
        logits_coarse, transition_params_coarse = sess.run(
            [self.logits_coarse, self.transition_params_coarse], feed_dict_coarse)
        y_pred_coarse = []
        for logits_, seq_len_ in zip(logits_coarse, seq_len):
            logits_ = logits_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logits_, transition_params_coarse)
            viterbi_sequence.extend([0 for _ in range(self.max_seq_len - seq_len_)])
            y_pred_coarse.append(viterbi_sequence)

        feed_dict_fine = {
            self.x: x,
            self.y_seg: y_pred_seg,
            self.y_coarse: y_pred_coarse,
            self.seq_len: seq_len,
            self.dropout_keep_prob: 1.0
        }
        logits_fine, transition_params_fine = sess.run(
            [self.logits_fine, self.transition_params_fine], feed_dict_fine)
        y_pred_fine = []
        for logits_, seq_len_ in zip(logits_fine, seq_len):
            logits_ = logits_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logits_, transition_params_fine)
            y_pred_fine.append(viterbi_sequence)

        return y_pred_fine
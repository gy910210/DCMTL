import numpy as np
import tensorflow as tf

from config import config_train
from data_char import DataLoader, BatchGenerator, VocabularyLoader, LabelLoader
from NER_base import eval_seq_crf_with_o, eval_seq_crf_no_o
from NER_MTL import NER_MTL


args = config_train()

print("Parameters:")
for attr, value in sorted(args.items(), reverse=True):
    print("{}={}".format(attr.upper(), value))
print("")

vocab_loader = VocabularyLoader()
vocab_loader.load_vocab(args['vocab_file'], args["embedding_dim"], args['encoding'])
args['vocab_size'] = vocab_loader.vocab_size
print("Embedding shape: %s.", vocab_loader.vocab_embedding.shape)

seg_label_loader = LabelLoader()
seg_label_loader.load_label(args['seg_label_file'], args['encoding'])

coarse_label_loader = LabelLoader()
coarse_label_loader.load_label(args['coarse_label_file'], args['encoding'])

fine_label_loader = LabelLoader()
fine_label_loader.load_label(args['fine_label_file'], args['encoding'])

args['seg_num_classes'] = seg_label_loader.label_size
print("Number of seg labels: %d.", seg_label_loader.label_size)

args['coarse_num_classes'] = coarse_label_loader.label_size
print("Number of coarse labels: %d.", coarse_label_loader.label_size)

args['fine_num_classes'] = fine_label_loader.label_size
print("Number of fine labels: %d.", fine_label_loader.label_size)

data_loader_train = DataLoader(args["max_seq_len"],
                               vocab_loader.vocab_index_dict,
                               seg_label_loader.label_index_dict,
                               coarse_label_loader.label_index_dict,
                               fine_label_loader.label_index_dict)
data_loader_train.load(args["data_file_train"], args['encoding'])

data_loader_test = DataLoader(args["max_seq_len"],
                              vocab_loader.vocab_index_dict,
                              seg_label_loader.label_index_dict,
                              coarse_label_loader.label_index_dict,
                              fine_label_loader.label_index_dict)
data_loader_test.load(args["data_file_test"], args['encoding'])

print("Create train, test split...")
train_size = data_loader_train.data_size
test_size = data_loader_test.data_size

train_data = data_loader_train.data

query_idx_test, query_len_test, seg_label_test, coarse_label_test, fine_label_test = \
    zip(*data_loader_test.data)

print("train: test = %d: %d.", train_size, test_size)

batch_generator = BatchGenerator(train_data, args["batch_size"])

logger = open("logs/log.txt", "w")

print("Create models...")
# Training
# ==================================================
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        # build model
        model = NER_MTL(
            batch_size=args['batch_size'],
            vocab_size=args['vocab_size'],
            word_dim=args['embedding_dim'],
            lstm_dim=args['lstm_dim'],
            max_seq_len=args['max_seq_len'],
            seg_num_classes=args['seg_num_classes'],
            coarse_num_classes=args['coarse_num_classes'],
            fine_num_classes=args['fine_num_classes'],
            lr=args['learning_rate'],
            gradient_clip=args['max_grad_norm'],
            l2_reg_lambda=args['l2_reg_lambda'],
            init_word_embedding=vocab_loader.vocab_embedding,
            layer_size=args['layer_size'],
            is_multi_task=(args['is_multi_task'] == "True")
        )
        
        sess.run(tf.global_variables_initializer())

        best_f1_valid, best_precision_valid, best_recall_valid = 0.0, 0.0, 0.0
        best_path = None

        ############################
        label_test = fine_label_test
        index_label_dict = fine_label_loader.index_label_dict
        ############################

        for epoch in range(0, args["num_epochs"]):
            print("Epoch %d/%d...", epoch, args["num_epochs"])

            batch_generator.reset_batch_pointer()
            for batch in range(batch_generator.num_batches):
                x_batch, seq_len_batch, seg_label_batch, coarse_label_batch, fine_label_batch = \
                    batch_generator.next_batch()

                rand = np.random.choice([0, 1, 2, 3], size=1)

                if rand == 0:
                    opt_type = "SEG"
                elif rand == 1:
                    opt_type = "COARSE"
                elif rand == 2:
                    opt_type = "FINE"
                else:
                    opt_type = 'ALL'

                current_step, loss = model.train_step(sess, x_batch,
                                                      seg_label_batch, coarse_label_batch, fine_label_batch,
                                                      seq_len_batch,
                                                      args['dropout_keep_prob'],
                                                      opt_type=opt_type)

                print("Batch %d/%d ==> loss: %.4f",
                      batch+1, batch_generator.num_batches, loss)

                if current_step % args["eval_step"] == 0:
                    print("Eval on validation set...")
                    tag_list = model.decode(sess,
                                            np.array(list(query_idx_test)),
                                            np.array(list(seg_label_test)),
                                            np.array(list(coarse_label_test)),
                                            np.array(list(fine_label_test)),
                                            np.array(list(query_len_test)))

                    precision_valid = eval_seq_crf_no_o(
                        tag_list,
                        [x[:y] for x, y in zip(label_test, query_len_test)],
                        index_label_dict,
                        method='precision'
                    )
                    recall_valid = eval_seq_crf_no_o(
                        tag_list,
                        [x[:y] for x, y in zip(label_test, query_len_test)],
                        index_label_dict,
                        method='recall'
                    )

                    if precision_valid == 0 or recall_valid == 0:
                        f1_valid = 0.0
                    else:
                        f1_valid = 2 * precision_valid * recall_valid / (precision_valid + recall_valid)

                    print("Precision-valid: %.4f, Recall-valid: %.4f, F1-valid: %.4f",
                          precision_valid, recall_valid, f1_valid)

                    logger.write(str(current_step) + "\t" + str(precision_valid) +
                                 "\t" + str(recall_valid) + "\t" + str(f1_valid) + "\n")

                    # valid result improved, testing on test set
                    if f1_valid > best_f1_valid:
                        best_precision_valid = precision_valid
                        best_recall_valid = recall_valid
                        best_f1_valid = f1_valid
                        best_path = model.save(sess, "models")
                        print("Saved model checkpoint to {}\n".format(best_path))

        print('-------------Show the results:--------------')
        print("Best Precision-valid: %.4f, Best Recall-valid: %.4f, Best F1-valid: %.4f",
              best_precision_valid, best_recall_valid, best_f1_valid)

        logger.write("\nBest Precision, Recall and F1: \n")
        logger.write(str(best_precision_valid) + "\t" + str(best_recall_valid) + "\t" + str(best_f1_valid) + "\n")

logger.close()
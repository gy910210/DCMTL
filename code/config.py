import argparse

def config_train():
    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file_train', type=str,
                        help='training data file.')
    parser.add_argument('--data_file_test', type=str,
                        help='test data file.')
    parser.add_argument('--vocab_file', type=str,
                        help='vocab embedding file.')

    parser.add_argument('--seg_label_file', type=str,
                        help='seg label index file.')
    parser.add_argument('--coarse_label_file', type=str,
                        help='coarse label index file.')
    parser.add_argument('--fine_label_file', type=str,
                        help='fine label index file.')

    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    # Data format
    parser.add_argument('--max_seq_len', type=int,
                        help='max sequence length')

    parser.add_argument('--seg_num_classes', type=int,
                        help='number of seg labels')
    parser.add_argument('--coarse_num_classes', type=int,
                        help='number of coarse labels')
    parser.add_argument('--fine_num_classes', type=int,
                        help='number of fine labels')

    parser.add_argument('--vocab_size', type=int,
                        help='size of vocab')
    parser.add_argument('--embedding_dim', type=int,
                        help='dimension of embedding')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='dropout keep rate, default to 1.0 (no dropout).')
    parser.add_argument('--eval_step', type=int, default=10,
                        help='every steps to evaluation.')

    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='initial learning rate')
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0,
                        help='l2 reg lambda.')

    # Parameters for model.
    parser.add_argument('--lstm_dim', type=int,
                        help='lstm dimension.')
    parser.add_argument('--layer_size', type=int,
                        help='layer size.')

    parser.add_argument('--is_multi_task', type=str, default='False',
                        help = 'is multi_task.')

    args = parser.parse_args()

    return vars(args)
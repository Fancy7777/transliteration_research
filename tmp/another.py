#import dependencies
import tensorflow as tf
import numpy as np
import helpers
import time


def prepare_chinese_id_dict():
    # prepare chinese_id_dict
    chinese_id_list = []
    with open('./chinese_vocabulary.txt') as file:
        for line in file:
            chinese_id_list.append(line.split('\n')[0])
    chinese_id_list = ['PAD', 'EOS'] + chinese_id_list
    chinese_id_dict = {}
    for i in range(len(chinese_id_list)):
        chinese_id_dict[chinese_id_list[i]] = i
    # finish preparation for chinese_id_dict
    return chinese_id_dict

def prepare_english_phoneme_dict():
    english_phoneme_dict = {}
    for i in range(len(vocab_inputs)):
        english_phoneme_dict[vocab_inputs[i]] = i
    return english_phoneme_dict

def prepare_vocab_input():
    vocab_inputs = []
    with open('./english_phoneme_vocabulary_output.txt') as file:
        for line in file:
            vocab_inputs.append(line.split('\n')[0])
    vocab_inputs.remove('_PAD')
    vocab_inputs.remove('_GO')
    vocab_inputs.remove('_EOS')
    vocab_inputs.remove('_UNK')
    vocab_inputs = ['PAD', 'EOS'] + (vocab_inputs)
    return vocab_inputs

def prepare_vocab_predict():
    vocab_predict = list(chinese_id_dict.keys())
    return vocab_predict


def batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major


def feeding(inputs, labels,encoder_inputs,decoder_inputs,decoder_targets,english_phoneme_dict):
    inputs_int = [];
    predict_int = []
    for i in range(len(inputs)):
        single_input = []
        single_predict = []
        for x in range(len(labels[i])):
            try:
                single_input += [english_phoneme_dict[inputs[i][x]]]
            except:
                single_input += [0]
        for x in range(len(labels[i])):
            single_predict += [chinese_id_dict[labels[i][x]]]
        inputs_int.append(single_input);
        predict_int.append(single_predict)

    enc_input, _ = helpers.batch(inputs_int)
    dec_target, _ = helpers.batch([(sequence) + [1] for sequence in predict_int])
    dec_input, _ = helpers.batch([[1] + (sequence) for sequence in inputs_int])

    return {encoder_inputs: enc_input, decoder_inputs: dec_input, decoder_targets: dec_target}

def graph():
    #reset the whole thing

    #tf.reset_default_graph()
    with tf.device('/gpu:6'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
    #sess = tf.InteractiveSession()

    #add placeholders
        encoder_inputs = tf.placeholder(shape = (None, None), dtype = tf.int32)
        decoder_targets = tf.placeholder(shape = (None, None), dtype = tf.int32)
        decoder_inputs = tf.placeholder(shape = (None, None), dtype = tf.int32)


        encoder_embeddings = tf.Variable(tf.random_uniform([len(vocab_inputs), greatestvalue_predict]
                                                   , -1.0, 1.0), dtype = tf.float32)

        decoder_embeddings = tf.Variable(tf.random_uniform([len(vocab_predict), greatestvalue_predict]
                                                   , -1.0, 1.0), dtype = tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)


    #define a hidden unit here. Try the hyper parameter
        hidden = 280
    #################################encoder part############################################
    # RNN size of greatestvalue_inputs
        #encoder_cell = tf.contrib.rnn.LSTMCell(hidden)

    # 2 layers of RNN
    #encoder_rnn_cells =tf.contrib.rnn.MultiRNNCell([encoder_cell] * 2)
        # add dropout layer here
        #dropout_lstm_encoder = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=0.5)

        # try this
        cells = []
        for _ in range(2):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden), input_keep_prob=0.5)
            cells.append(cell)
        multicell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

        _, encoder_final_state = tf.nn.dynamic_rnn(multicell, encoder_inputs_embedded,
                                           dtype = tf.float32, time_major = True)

    #################################end of encoder part#########################################


    #################################decoder part############################################
    # RNN size of greatestvalue_predict
        #decoder_cell = tf.contrib.rnn.LSTMCell(hidden)

        # 2 layers of RNN
        # decoder_rnn_cells = tf.contrib.rnn.MultiRNNCell([decoder_cell] * 2)
        # add dropout layer here
        #dropout_lstm_decoder = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=0.5)

        # try this
        decoder_cells = []
        # bidirectional
        for _ in range(2):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden), input_keep_prob=0.5)
            decoder_cells.append(cell)
        decoder_multicell = tf.contrib.rnn.MultiRNNCell(decoder_cells, state_is_tuple=False)

        # declare a scope for our decoder, later tensorflow will confuse
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_multicell, decoder_inputs_embedded,
                                                             initial_state=encoder_final_state,
                                                             dtype=tf.float32, time_major=True, scope='decoder')

    #################################end of decoder part############################################

        decoder_logits = tf.contrib.layers.linear(decoder_outputs, len(vocab_predict))

        decoder_prediction = tf.argmax(decoder_logits, 2)

    # this might very costly if you have very large vocab

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(decoder_targets, depth=len(vocab_predict), dtype=tf.float32),
            logits=decoder_logits)

        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        sess.run(tf.global_variables_initializer())

    batch_size = 2500
    epoch = 3000
    LOSS = []

    for q in range(epoch):
        total_loss = 0
        lasttime = time.time()
        for w in range(0, len(english_phoeneme) - batch_size, batch_size):
            _, losses = sess.run([optimizer, loss],
                                 feeding(english_phoeneme[w: w + batch_size], chinese[w: w + batch_size],
                                 encoder_inputs,decoder_inputs,decoder_targets,english_phoeneme_dict))

            total_loss += losses

        total_loss = total_loss / ((len(english_phoeneme) - batch_size) / (batch_size * 1.0))
        LOSS.append(total_loss)

        if (q + 1) % 10 == 0:
            print('epoch: ' + str(q + 1) + ', total loss: ' + str(total_loss) + ', s/epoch: ' + str(
                time.time() - lasttime))

    with open('BEST','w') as f:
        for ele in LOSS:
            f.write(str(ele)+'\t')
    f.close()

if __name__ == '__main__':


    vocab_inputs = prepare_vocab_input()

    chinese_id_dict = prepare_chinese_id_dict()
    vocab_predict = prepare_vocab_predict()

    english_phoeneme_dict = prepare_english_phoneme_dict()

    # get in the data
    chinese = []
    english_phoeneme = []
    with open('dataset.txt') as f:
        for line in f:
            temp = line.split('\t')
            chinese.append(temp[1])
            english_phoeneme.append(temp[2])
    english_phoeneme = [phoeneme.split('\n')[0].split() for phoeneme in english_phoeneme]
    chinese = [list(word) for word in chinese]


    # the dimension of english phonemes
    greatestvalue_predict = 42

    #run the session
    graph()

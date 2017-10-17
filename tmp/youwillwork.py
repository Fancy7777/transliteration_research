#import dependencies
import tensorflow as tf
import numpy as np
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

    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_lengths

def feeding(inputs, labels,encoder_inputs,decoder_inputs,decoder_targets,english_phoneme_dict):
    encoder_max_length = 39
    decoder_max_length = 75

    inputs_int = [];
    predict_int = []
    for i in range(len(inputs)):
        single_input = []
        single_predict = []
        for x in range(encoder_max_length):
            try:
                single_input += [english_phoneme_dict[inputs[i][x]]]
            except:
                single_input += [0]
        for x in range(decoder_max_length):
            try:
                single_predict += [chinese_id_dict[labels[i][x]]]
            except:
                single_predict += [0]
        inputs_int.append(single_input);
        predict_int.append(single_predict)

    enc_input, _ = batch(inputs_int)
    dec_target, _ = batch([(sequence) + [1] for sequence in predict_int])
    dec_input, _ = batch([[1] + (sequence) for sequence in predict_int])
    return {encoder_inputs: enc_input, decoder_inputs: dec_input, decoder_targets: dec_target}

def label_to_chinese(label):
    chinese = ''
    for i in range(len(label)):
        if label[i][0] == 0 or label[i][0] == 1:
            continue
        chinese += vocab_predict[label[i][0]] + ' '
    return chinese

def sent_len(sentence):
    '''
    Args:
        sentence: [None, max_sentence_length]
    Returns:
        length: [None], the actual length of each sentence in the batch
    '''
    used = tf.sign(tf.abs(sentence))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(x=length, dtype=tf.int32)
    return length


def graph():
    #reset the whole thing

    #tf.reset_default_graph()
    with tf.device('/gpu:5'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))

        embedding_size = 256

        # placeholders pass data in
        encoder_inputs = tf.placeholder(shape=(None, encoder_max_length), dtype=tf.int32)
        decoder_targets = tf.placeholder(shape=(None, decoder_max_length + 1), dtype=tf.int32)
        decoder_inputs = tf.placeholder(shape=(None, decoder_max_length + 1), dtype=tf.int32)

        # variables
        # size should be [max_time,batch_size] try batch_size here
        encoder_embeddings = tf.Variable(tf.random_uniform([len(vocab_inputs), embedding_size]
                                                           , -1.0, 1.0), dtype=tf.float32)
        decoder_embeddings = tf.Variable(tf.random_uniform([len(vocab_predict), embedding_size]
                                                           , -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)
        # initial encoder cell to be LSTM
        # initial encoder cell to be LSTM
        encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                                 dtype=tf.float32, time_major=False)
        # initial decoder to be also LSTMcell
        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded,
                                                                 initial_state=encoder_final_state, dtype=tf.float32,
                                                                 time_major=False, scope='decoder')

        decoder_logits = tf.contrib.layers.linear(decoder_outputs, len(vocab_predict))
        # [None, seq_len, vocab_size]

        # 在第二个维度上取argmax
        decoder_prediction = tf.argmax(decoder_logits, 2)

        cross_entropy_mask = tf.sequence_mask(
            lengths=sent_len(decoder_targets),
            maxlen=decoder_max_length + 1,
            dtype=tf.float32,
        )
        # [None, max_sentence_length]

        loss = tf.contrib.seq2seq.sequence_loss(
            logits=decoder_logits,  # [None, max_sentence_length, tag_set_size]
            targets=decoder_targets,  # [None, max_sentence_length]
            weights=cross_entropy_mask,
            average_across_timesteps=True,
            average_across_batch=True,
            name="seq2seq_sequence_loss",
        )


        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        sess.run(tf.global_variables_initializer())

        LOSS = []

        import time
        import random

        for q in range(epoch):
            total_loss = 0
            lasttime = time.time()

            for w in range(0, len(english_phoeneme) - batch_size, batch_size):
                # shuffle the batch
                combine = list(zip(english_phoeneme, chinese))
                random.shuffle(combine)
                batch_english_phoneme, batch_chinese = zip(*combine)

                _, losses = sess.run([optimizer, loss],
                                     feeding(batch_english_phoneme[w: w + batch_size],
                                             batch_chinese[w: w + batch_size],encoder_inputs,decoder_inputs,decoder_targets,english_phoneme_dict))

                total_loss += losses

            total_loss = total_loss / ((len(english_phoeneme) - batch_size) / (batch_size * 1.0))
            LOSS.append(total_loss)

            if (q + 1) % 10 == 0:
                print('epoch: ' + str(q + 1) + ', total loss: ' + str(total_loss) + ', s/epoch: ' + str(
                    time.time() - lasttime))
                for i in range(10):
                    rand = np.random.randint(len(english_phoeneme))
                    test = feeding(english_phoeneme[rand: rand + 1], chinese[rand: rand + 1], encoder_inputs,
                                   decoder_inputs, decoder_targets, english_phoneme_dict)
                    predict = sess.run(decoder_prediction, test)
                    print('input: ' + str(english_phoeneme[rand: rand + 1]))
                    print('supposed label: ' + str(chinese[rand: rand + 1]))
                    print('predict label:' + str(label_to_chinese(predict)) + '\n')
                print('#######################next 10 epoch#############################')



if __name__ == '__main__':


    vocab_inputs = prepare_vocab_input()

    chinese_id_dict = prepare_chinese_id_dict()
    vocab_predict = prepare_vocab_predict()

    english_phoneme_dict = prepare_english_phoneme_dict()

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

    for ele in chinese:
        if ' ' in ele:
            ele.remove(' ')

    for ele in chinese:
        if ' ' in ele:
            ele.remove(' ')

    batch_size = 256
    encoder_hidden_units = 100
    decoder_hidden_units = 100
    encoder_max_length = 39
    decoder_max_length = 75
    epoch = 100

    #run the session
    graph()
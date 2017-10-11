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

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major


def feed(text,english_phoneme_dict ,English=True):
    text_int = []
    if English:
        text_int_decode = [1]
    strings = text
    for i in range(greatestvalue_predict):
        try:
            if English:
                text_int.append(english_phoneme_dict[strings[i]])
                text_int_decode.append(english_phoneme_dict[strings[i]])
            else:
                text_int.append(chinese_id_dict[strings[i]])
        except:
            text_int.append(0)
            if English:
                text_int_decode.append(0)

    text_int[greatestvalue_predict - 1] = 1

    if English:
        del text_int_decode[len(text_int_decode) - 1]
        return text_int, text_int_decode
    else:
        return text_int

def label_to_chinese(label):
    chinese = ''
    for i in range(len(label)):
        if label[i][0] == 0 or label[i][0] == 1:
            continue
        chinese += vocab_predict[label[i][0]] + ' '
    return chinese

def graph(english_phoeneme_dict):
    #reset the whole thing

    #tf.reset_default_graph()
    with tf.device('/gpu:1'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
    #sess = tf.InteractiveSession()

    greatestvalue_predict = 75
    encoder_inputs = tf.placeholder(shape=[greatestvalue_predict], dtype=tf.int32)
    decoder_inputs = tf.placeholder(shape=[greatestvalue_predict], dtype=tf.int32)
    decoder_targets = tf.placeholder(shape=[greatestvalue_predict], dtype=tf.int32)

    # Build RNN cell
    hidden = 200
    size_layers = 1
    cell = tf.contrib.rnn.LSTMCell(hidden)

    outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
        encoder_inputs=[encoder_inputs],
        decoder_inputs=[decoder_inputs],
        cell=cell,
        num_encoder_symbols=len(vocab_inputs),
        num_decoder_symbols=len(vocab_inputs),
        embedding_size=size_layers)

    decoder_logits = tf.contrib.layers.linear(outputs, greatestvalue_predict)

    decoder_prediction = tf.argmax(decoder_logits, 2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=greatestvalue_predict, dtype=tf.float32),
        logits=decoder_logits)

    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess.run(tf.global_variables_initializer())

    epoch = 50
    LOSS = []

    import time

    for q in range(epoch):
        total_loss = 0
        lasttime = time.time()

        for w in range(len(english_phoeneme)):
            input_seq_encode, input_seq_decode = feed(english_phoeneme[w],english_phoeneme_dict, English=True)
            output_seq = feed(chinese[w],english_phoeneme_dict, English=False)
            _, losses = sess.run([optimizer, loss],
                                 feed_dict={encoder_inputs: input_seq_encode, decoder_inputs: input_seq_decode,
                                            decoder_targets: output_seq})
            total_loss += losses

        total_loss = total_loss / (len(english_phoeneme) * 1.0)
        LOSS.append(total_loss)
        print('epoch: ' + str(q + 1) + ', total loss: ' + str(total_loss) + ', s/epoch: ' + str(time.time() - lasttime))

        for i in range(10):
            rand = np.random.randint(len(english_phoeneme))
            input_seq_encode, input_seq_decode = feed(english_phoeneme[rand], english_phoeneme_dict, English=True)
            output_seq = feed(chinese[rand], english_phoeneme_dict, English=False)

            predict = sess.run(decoder_prediction, feed_dict={encoder_inputs: input_seq_encode, decoder_inputs: input_seq_decode,
                                            decoder_targets: output_seq})
            print('input: ' + str(english_phoeneme[rand: rand + 1]))
            print('supposed label: ' + str(chinese[rand: rand + 1]))
            print('predict label:' + str(label_to_chinese(predict)) + '\n')
        print('#######################next 10 epoch#############################')



    with open('attention','w') as f:
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
    for ele in chinese:
        if ' ' in ele:
            ele.remove(' ')


    # the dimension of english phonemes
    greatestvalue_predict = 75

    #run the session
    graph(english_phoeneme_dict)

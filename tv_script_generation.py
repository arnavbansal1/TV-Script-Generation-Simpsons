#Arnav Bansal
import os
from os.path import join
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq

def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], 'input')
    targets = tf.placeholder(tf.int32, [None, None])
    learning_rate = tf.placeholder(tf.float32)
    return (inputs, targets, learning_rate)

def get_init_cell(batch_size, rnn_size):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, 'initial_state')
    return (cell, initial_state)

def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return (outputs, final_state)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    inputs = get_embed(input_data, vocab_size, embed_dim)
    rnn = build_rnn(cell, inputs)
    outputs = rnn[0]
    final_state = rnn[1]
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    return (logits, final_state)

def get_batches(int_text, batch_size, seq_length):
    num_batches = len(int_text)//(batch_size * seq_length)
    batches = np.zeros([num_batches, 2, batch_size, seq_length], dtype=int)
    data = int_text[:num_batches * batch_size * seq_length]
    
    for i in range(0, len(data), seq_length):
        batches[(i%(num_batches * seq_length))//seq_length, 0, i//(num_batches * seq_length)] = data[i:i+seq_length]
    
    for i in range(1, len(data), seq_length):
        batches[((i-1)%(num_batches * seq_length))//seq_length, 1, (i-1)//(num_batches * seq_length)] = data[i:i+seq_length] if i+seq_length <= len(data) else data[i:i+seq_length-1] + data[0:1]
    
    return batches

num_epochs = 100
batch_size = 64
rnn_size = 256
embed_dim = 300
seq_length = 50
learning_rate = 0.01
show_every_n_batches = 100

save_dir = './save'
data_dir = './data/simpsons/moes_tavern_lines.txt'

input_file = os.path.join(data_dir)
with open(input_file, "r") as f:
    text = f.read()
text = text[81:]

token_dict = {'.':'||Period||', ',':'||Comma||', '"':'||Quotation_Mark||', ';':'||Semicolon||', '!':'||Exclamation_Mark||', '?':'||Question_Mark||', '(':'||Left_Parentheses||', ')':'||Right_Parentheses||', '--':'||Dash||', '\n':'||Return||'}

for key, token in token_dict.items():
    text = text.replace(key, ' {} '.format(token))

text = text.lower()
text = text.split()
vocab = set(text)
vocab_to_int = {word : i for i, word in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
int_text = [vocab_to_int[word] for word in text]
pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

int_text, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    probs = tf.nn.softmax(logits, name='probs')

    cost = seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))

    optimizer = tf.train.AdamOptimizer(lr)

    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {input_text: x, targets: y, initial_state: state, lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(epoch_i, batch_i, len(batches), train_loss))

    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
    
pickle.dump((seq_length, save_dir), open('params.p', 'wb'))

_, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, load_dir = pickle.load(open('params.p', mode='rb'))

def get_tensors(loaded_graph):
    inputs = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return (inputs, initial_state, final_state, probs)

def pick_word(probabilities, int_to_vocab):
    return int_to_vocab[np.argmax(probabilities)]

gen_length = 200
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run([probs, final_state], {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    tv_script = ' '.join(gen_sentences)
    
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import matplotlib.pyplot as plt
import io

def train_test_set(training_size):
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    return training_sentences, testing_sentences, training_labels, testing_labels


def tokenize_to_padded(sentences, vocab_size, oov_tok, max_length, padding_type, trunc_type):

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index

    sentence_to_sequences = tokenizer.texts_to_sequences(sentences)
    padded_sentence = pad_sequences(sentence_to_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded_sentence, word_index


def padded_to_arary(training_padded, training_labels, testing_padded, testing_labels):
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    return training_padded, training_labels, testing_padded, testing_labels


import matplotlib.pyplot as plt

def evaluating_model(history, string):

    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()



def reverse_word_index(word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return reverse_word_index



def decode_sentence_from_padding(training_padded_or_testing_padded):
    return ' '.join([reverse_word_index.get(i, '?') for i in training_padded_or_testing_padded])



def input_shape(model):
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(f'(vocab_size, embedding_dim): {weights.shape}') # shape: (vocab_size, embedding_dim)



def downloading_embedded_text(vocab_size, reverse_word_index):
    import io

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')


    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    try:
        from google.colab import files
    
    except ImportError:
        pass
    else:
        files.download('vecs.tsv')
        files.download('meta.tsv')


def model_save(model):
    model.save("test.h5")


def sequence_text(corpus):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences



def input_sequences_toarray(input_sequences, padding_type):

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding= padding_type))


def input_sequences_to_xs_ys(input_sequences):

    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


def predicting_sequence_text(seed_text, next_words_num):
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
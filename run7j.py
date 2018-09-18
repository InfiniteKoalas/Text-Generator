import faiss
import random
import numpy as np
import spacy
import tensorflow as tf
import model7j

BATCH_SIZE = 64
WORDVEC_SIZE = 300
BUFFER_SIZE = 10000

tf.enable_eager_execution()
nlp = spacy.load('en_core_web_lg')
faiss_index = faiss.IndexFlatL2(WORDVEC_SIZE)
faiss_index.add(nlp.vocab.vectors.data)
row2key = {v: k for k, v in nlp.vocab.vectors.key2row.iteritems()}


def is_ascii(x):
    return all(ord(c) < 128 for c in x)


def most_similar(vectors):
    vectors = np.asarray(vectors)
    D, I = faiss_index.search(vectors, 10)
    result = []
    for i in I:
        for w in i:
            if not w:
                continue
            t = nlp.vocab[row2key[w]]
            if not is_ascii(t.text):
                continue
            result.append(t)
            break
    return result


model = model7j.Model()
model.restore()

num_generate = 100
input_eval = [nlp.vocab[u'The'].vector]
input_eval = tf.expand_dims(input_eval, 0)
output = []
temperature = 0.0
hidden = [tf.zeros((1, model.units))]
for i in xrange(num_generate):
    predictions, hidden = model(input_eval, hidden)
    predictions = np.reshape(predictions.numpy(), [-1])
    tokens = most_similar([predictions])
    token = tokens[0]
    output.append(str(token.text).lower())
    v = token.vector
    v = 2 * (temperature * random.random() + (1 - temperature)) * v
    input_eval = tf.reshape(v, (1, 1, -1))

print ' '.join(output)

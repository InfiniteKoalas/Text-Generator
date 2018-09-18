import time
import spacy
import tensorflow as tf
import model7j

BATCH_SIZE = 64
SEQUENCE_LENGTH = 100
SENTENCE_LIMIT = 0
WORDVEC_SIZE = 300
BUFFER_SIZE = 10000
EPOCHS = 20


def get_sentences(text):
    start = 0
    end = 0
    sentences = []
    while end >= 0:
        end = text.find('.', start)
        short = text[start:end]
        start = end + 1
        words = nlp(unicode(short))
        words = [w for w in words if w.is_alpha or w.is_punct]
        sentences.append(words)
        if SENTENCE_LIMIT > 0 and len(sentences) >= SENTENCE_LIMIT:
            break
    return sentences


tf.enable_eager_execution()
nlp = spacy.load('en_core_web_lg')

path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/'
        'data/shakespeare.txt')

print 'starting get_sentences'
raw_text = open(path_to_file).read()
sentences = get_sentences(raw_text)
print 'get_sentences done', len(sentences)

text = []
for s in sentences:
    text.extend(s)

input_text = []
target_text = []

print 'starting generate_input'
for f in range(0, len(text)-SEQUENCE_LENGTH, SEQUENCE_LENGTH):
    inps = text[f:f+SEQUENCE_LENGTH]
    targ = text[f+1:f+1+SEQUENCE_LENGTH]

    input_text.append([i.vector for i in inps])
    target_text.append([t.vector for t in targ])
print 'generate_input done', len(input_text)

dataset = tf.data.Dataset.from_tensor_slices(
        (input_text, target_text)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

model = model7j.Model()


def loss_function(real, preds):
    return tf.losses.mean_squared_error(
            labels=tf.reshape(real, [-1]),
            predictions=tf.reshape(preds, [-1]))


for epoch in range(EPOCHS):
    start = time.time()

    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions, hidden = model(inp, hidden)

            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)

            grads = tape.gradient(loss, model.variables)
            model.optimizer.apply_gradients(zip(grads, model.variables))

        print ('Epoch {} Batch {} Loss {:.4f}'.format(
            epoch+1, batch, loss))

    if (epoch + 1) % 5 == 0:
        model.save()

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

import time
import tensorflow as tf
import unidecode
import model6j

EPOCHS = 20
SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000


tf.enable_eager_execution()

path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/'
        'data/shakespeare.txt')

text = unidecode.unidecode(open(path_to_file).read())
unique = sorted(set(text))
char2idx = {u: i for i, u in enumerate(unique)}
idx2char = {i: u for i, u in enumerate(unique)}
vocab_size = len(unique)

model = model6j.Model(vocab_size)
saved_data = model.get_saved_data()
saved_data.char2idx = char2idx
saved_data.idx2char = idx2char

input_text = []
target_text = []

for f in range(0, len(text)-SEQUENCE_LENGTH, SEQUENCE_LENGTH):
    inps = text[f:f+SEQUENCE_LENGTH]
    targ = text[f+1:f+1+SEQUENCE_LENGTH]
    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])

dataset = tf.data.Dataset.from_tensor_slices(
        (input_text, target_text)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


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
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, loss))
    if (epoch + 1) % 5 == 0:
        model.save()
    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

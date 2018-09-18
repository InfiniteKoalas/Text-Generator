import tensorflow as tf
import model6j

tf.enable_eager_execution()

model = model6j.restore()
saved_data = model.get_saved_data()
char2idx = saved_data.char2idx
idx2char = saved_data.idx2char

num_generate = 1000
start_string = 'Q'
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)
text_generated = ''
temperature = 1.0

# hidden state shape == (batch_size, number of rnn units); batch size == 1
hidden = [tf.zeros((1, model.units))]
for i in range(num_generate):
    predictions, hidden = model(input_eval, hidden)
    predictions = predictions / temperature
    predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated += idx2char[predicted_id]
print (start_string + text_generated)

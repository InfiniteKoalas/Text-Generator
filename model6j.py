import cPickle
import os
import tensorflow as tf

GRU_UNITS = 1024
EMBEDDING_DIM = 256


class _SavedData(object):
    pass


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM,
                 units=GRU_UNITS,
                 checkpoint_dir='model6j_training_checkpoints'):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.optimizer = tf.train.AdamOptimizer()
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                                  model=self)

    def get_saved_data(self):
        if not hasattr(self, 'saved_data'):
            self.saved_data = _SavedData()
        return self.saved_data

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        save_data = self.get_saved_data()
        save_data.vocab_size = self.vocab_size
        save_data.embedding_dim = self.embedding_dim
        save_data.units = self.units
        cPickle.dump(
                save_data,
                open(os.path.join(self.checkpoint_dir, "saved_data"), "w"))

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(
            self.checkpoint_dir))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, states = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, states


def restore(checkpoint_dir='model6j_training_checkpoints'):
    saved_data = cPickle.load(open(os.path.join(checkpoint_dir,
                                                "saved_data")))
    model = Model(saved_data.vocab_size, saved_data.embedding_dim,
                  saved_data.units, checkpoint_dir)
    model.restore()
    model.saved_data = saved_data
    return model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_train = (x_train > 0.5).astype('float32')
x_train = x_train.reshape(-1, 784)

batch_size = 64
train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(10000)
    .batch(batch_size)
)

class RBM(tf.keras.Model):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        initializer = tf.initializers.RandomNormal(
            mean=0.0, stddev=0.01
        )

        self.W = tf.Variable(
            initializer([n_visible, n_hidden]),
            name='weights'
        )
        self.h_bias = tf.Variable(
            tf.zeros([n_hidden]), name='hidden_bias'
        )
        self.v_bias = tf.Variable(
            tf.zeros([n_visible]), name='visible_bias'
        )

    def sample_prob(self, probs):
        return tf.nn.relu(
            tf.sign(probs - tf.random.uniform(tf.shape(probs)))
        )

    def sample_h(self, v):
        prob_h = tf.nn.sigmoid(
            tf.matmul(v, self.W) + self.h_bias
        )
        return prob_h, self.sample_prob(prob_h)

    def sample_v(self, h):
        prob_v = tf.nn.sigmoid(
            tf.matmul(h, tf.transpose(self.W)) + self.v_bias
        )
        return prob_v, self.sample_prob(prob_v)

    def contrastive_divergence(self, v, lr=0.01):
        # Positive phase
        prob_h, h0 = self.sample_h(v)

        # Negative phase
        prob_v, v1 = self.sample_v(h0)
        prob_h1, _ = self.sample_h(v1)

        positive_grad = tf.matmul(tf.transpose(v), prob_h)
        negative_grad = tf.matmul(tf.transpose(v1), prob_h1)

        batch_size = tf.cast(tf.shape(v)[0], tf.float32)

        self.W.assign_add(
            lr * (positive_grad - negative_grad) / batch_size
        )
        self.v_bias.assign_add(
            lr * tf.reduce_mean(v - v1, axis=0)
        )
        self.h_bias.assign_add(
            lr * tf.reduce_mean(prob_h - prob_h1, axis=0)
        )

        loss = tf.reduce_mean(tf.square(v - v1))
        return loss

n_visible = 784
n_hidden = 128

rbm = RBM(n_visible, n_hidden)

n_epochs = 5
lr = 0.05
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0

    for batch in train_dataset:
        loss = rbm.contrastive_divergence(batch, lr)
        epoch_loss += loss.numpy()

    avg_loss = epoch_loss / len(list(train_dataset))
    losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{n_epochs}, "
        f"Reconstruction Loss: {avg_loss:.4f}"
    )

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('RBM Training Loss')
plt.grid(True)
plt.show()

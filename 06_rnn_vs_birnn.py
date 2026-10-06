import tensorflow as tf
import numpy as np

seq_length = 10
num_samples = 1000

X = []
y = []

for i in range(num_samples):
    start = np.random.randint(0, 100)
    seq = np.arange(start, start + seq_length)
    X.append(seq[:-1])
    y.append(seq[1:])

X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.float32)[..., np.newaxis]

# Simple RNN
rnn_model = tf.keras.Sequential()
rnn_model.add(
    tf.keras.layers.SimpleRNN(
        32, return_sequences=True,
        input_shape=(seq_length-1, 1)
    )
)
rnn_model.add(tf.keras.layers.Dense(1))
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
rnn_model.fit(X, y, epochs=50, verbose=0)

# Bidirectional RNN
birnn_model = tf.keras.Sequential()
birnn_model.add(
    tf.keras.layers.Bidirectional(
        tf.keras.layers.SimpleRNN(32, return_sequences=True),
        input_shape=(seq_length-1, 1)
    )
)
birnn_model.add(tf.keras.layers.Dense(1))
birnn_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
birnn_model.fit(X, y, epochs=50, verbose=0)

rnn_loss = rnn_model.evaluate(X, y, verbose=0)
birnn_loss = birnn_model.evaluate(X, y, verbose=0)

print(f"Final Loss (RNN): {rnn_loss:.4f}")
print(f"Final Loss (BiRNN): {birnn_loss:.4f}")

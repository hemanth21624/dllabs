import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# SGD
model_sgd = Sequential()
model_sgd.add(Input(shape=(2,)))
model_sgd.add(Dense(4, activation='sigmoid'))
model_sgd.add(Dense(1, activation='sigmoid'))
model_sgd.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.1))
history_sgd = model_sgd.fit(X, y, epochs=5000, verbose=0)

# Momentum
model_momentum = Sequential()
model_momentum.add(Input(shape=(2,)))
model_momentum.add(Dense(4, activation='sigmoid'))
model_momentum.add(Dense(1, activation='sigmoid'))
model_momentum.compile(
    loss='mean_squared_error',
    optimizer=SGD(learning_rate=0.1, momentum=0.9)
)
history_momentum = model_momentum.fit(X, y, epochs=5000, verbose=0)

# Adam
model_adam = Sequential()
model_adam.add(Input(shape=(2,)))
model_adam.add(Dense(4, activation='sigmoid'))
model_adam.add(Dense(1, activation='sigmoid'))
model_adam.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.1))
history_adam = model_adam.fit(X, y, epochs=5000, verbose=0)

plt.plot(history_sgd.history['loss'], label='SGD')
plt.plot(history_momentum.history['loss'], label='Momentum')
plt.plot(history_adam.history['loss'], label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimization Algorithms Comparison')
plt.show()

print(f"Final Loss (SGD): {history_sgd.history['loss'][-1]:.6f}")
print(f"Final Loss (Momentum): {history_momentum.history['loss'][-1]:.6f}")
print(f"Final Loss (Adam): {history_adam.history['loss'][-1]:.6f}")

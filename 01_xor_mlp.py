import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {accuracy*100:.2f}%")

predictions = model.predict(X, verbose=0)
predictions = (predictions > 0.5).astype(int)

print("\nPredictions")
for i, prediction in enumerate(predictions):
    print(f"Input: {X[i]} - Predicted Output: {prediction} - True Output: {y[i]}")

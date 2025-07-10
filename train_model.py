import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint

X = np.load("input.npy")
y = np.load("output.npy")

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1]))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

checkpoint = ModelCheckpoint("music_model.h5", monitor="loss", save_best_only=True)
model.fit(X, y, epochs=100, batch_size=64, callbacks=[checkpoint])

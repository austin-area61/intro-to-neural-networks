import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#inputs: temperature, isSunny(1 for it being sunny, 0 for not sunny)
x = np.array([
  [30, 1],
  [10, 0],
  [20, 1],
  [15, 0],
  [25, 1],
  [5, 0]
])

y = np.array([1,0,1,0,1,0])

model = Sequential([
  Dense(1, input_shape =(2,), activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x,y, epochs = 500, verbose = 0)
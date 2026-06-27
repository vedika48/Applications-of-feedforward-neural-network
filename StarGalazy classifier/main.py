import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("dataset/star_classification.csv")
df.drop(
    ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID'],
    axis=1,
    inplace=True
)
x = df.drop('class', axis=1)
y = df['class']

encoder = LabelEncoder() # converts categorical data into numeric values stating from 0 to n-1 classes
y = encoder.fit_transform(y)

scalar = StandardScaler()
x = scalar.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs") # tensorboard --logdir=./logs
history = model.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)
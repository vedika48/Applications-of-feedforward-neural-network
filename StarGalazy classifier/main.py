import pandas as pd
import matplotlib.pyplot as plt
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


# Network structure
layers = [13, 64, 32, 16, 3]  # input → hidden → output
layer_names = [
    "Input Layer\n(13 features)",
    "Hidden Layer 1\n(64)",
    "Hidden Layer 2\n(32)",
    "Hidden Layer 3\n(16)",
    "Output Layer\n(3 classes)"
]

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

v_spacing = 1
h_spacing = 2

positions = []

for i, layer_size in enumerate(layers):
    layer_x = i * h_spacing
    layer_positions = []

    # center the layer vertically
    y_offset = -(layer_size - 1) * v_spacing / 2

    for j in range(layer_size):
        y = j * v_spacing + y_offset
        layer_positions.append((layer_x, y))

        # color coding
        if i == 0:
            color = "#4A90E2"  # input blue
        elif i == len(layers) - 1:
            color = "#F39C12"  # output orange
        else:
            color = "#1ABC9C"  # hidden teal

        circle = plt.Circle((layer_x, y), 0.15, color=color, ec="black", zorder=3)
        ax.add_patch(circle)

    positions.append(layer_positions)

    # layer label
    ax.text(layer_x, y_offset + (layer_size * v_spacing) / 2 + 1,
            layer_names[i],
            ha='center',
            fontsize=10,
            fontweight='bold')

for i in range(len(layers) - 1):
    for (x1, y1) in positions[i]:
        for (x2, y2) in positions[i + 1]:
            ax.plot([x1, x2], [y1, y2],
                    color="gray",
                    alpha=0.3,
                    linewidth=0.5)

ax.text(layers[-1], max([max([y for x, y in layer]) for layer in positions]) + 2,
        "Deep Feedforward Neural Network (MLP)",
        ha="center",
        fontsize=14,
        fontweight="bold")

plt.show()
# General libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Tensorflow
import tensorflow as tf
from pyts.image import GramianAngularField
from scipy.signal import find_peaks
from tensorflow import keras
from keras.layers import *

# Scripts
import utils

pd.set_option("display.precision", 10)
pd.set_option("display.max_columns", 500)


def create_model():
    kernel_size = (20, 3)
    filter_size = 128
    dropout = 0.3
    input_layer = keras.layers.Input(shape=(window_size, window_size, 3))
    conv1 = keras.layers.Conv2D(
        filters=filter_size,
        kernel_size=kernel_size,
        activation="relu",
    )(input_layer)
    conv1 = keras.layers.MaxPool2D()(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(dropout)(conv1)

    conv2 = keras.layers.Conv2D(
        filters=filter_size, kernel_size=kernel_size, activation="relu"
    )(conv1)
    conv2 = keras.layers.MaxPool2D()(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(dropout)(conv2)

    conv3 = keras.layers.Conv2D(
        filters=filter_size, kernel_size=kernel_size, activation="relu"
    )(conv2)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.MaxPool2D()(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(dropout)(conv3)

    gap = keras.layers.GlobalMaxPooling2D()(conv3)

    output_layer = keras.layers.Dense(len(appliances), activation="sigmoid")(
        gap
    )

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


# Constants
DATA_PATH = "data/6_months"
MODEL_PATH = "2D-CNN/" + DATA_PATH.split("/")[-1]

appliances = [
    "Stove",
    "Coffee_machine",
    "Microwave",
    "Dishwasher",
    "Washing_machine",
]
window_size = 180
step = 1

train_df_mapping = pd.read_csv("image_csvs/train.csv", index_col=0)
val_df_mapping = pd.read_csv("image_csvs/val.csv", index_col=0)
test_df_mapping = pd.read_csv("image_csvs/test.csv", index_col=0)

for appliance in appliances:
    train_df_mapping = train_df_mapping.astype({appliance: np.float32})
    val_df_mapping = val_df_mapping.astype({appliance: np.float32})
    test_df_mapping = test_df_mapping.astype({appliance: np.float32})

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_ds = train_datagen.flow_from_dataframe(
    train_df_mapping,
    directory="GAF/train",
    x_col="X",
    y_col=appliances,
    batch_size=16,
    seed=42,
    target_size=(window_size, window_size),
    class_mode="raw",
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
val_ds = val_datagen.flow_from_dataframe(
    val_df_mapping,
    directory="GAF/val",
    x_col="X",
    y_col=appliances,
    batch_size=16,
    seed=42,
    target_size=(window_size, window_size),
    class_mode="raw",
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = test_datagen.flow_from_dataframe(
    test_df_mapping,
    directory="GAF/test",
    x_col="X",
    y_col=appliances,
    batch_size=16,
    seed=42,
    target_size=(window_size, window_size),
    class_mode="raw",
)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

EPOCHS = 30
BATCH_SIZE = 64

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, verbose=2, restore_best_weights=True
    ),
]
LR = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
f"Setting learning rate to {LR}"
loss_fn = tf.keras.losses.BinaryCrossentropy()
model = create_model()
model.compile(
    optimizer=optimizer, loss=loss_fn, metrics=["AUC", "binary_accuracy"]
)

history = model.fit(
    train_ds,
    callbacks=callbacks,
    validation_data=val_ds,
    epochs=EPOCHS,
)

THRESHOLD = 0.5
y_pred = model.predict(test_ds)
y_pred[y_pred >= THRESHOLD] = 1
y_pred[y_pred < THRESHOLD] = 0

test_loss, test_auc, test_acc = model.evaluate(test_ds)

print("Test AUC", test_auc)
print("Test Acc", test_acc)
print("Test loss", test_loss)

MODEL_NAME = "128,128,128 CNN"

model.save(
    f"{MODEL_PATH}/{MODEL_NAME}/LR={LR},Epochs={EPOCHS},BATCH={BATCH_SIZE}"
)

metric = "auc"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig(
    f"{MODEL_PATH}/{MODEL_NAME}/LR={LR},Epochs={EPOCHS},BATCH={BATCH_SIZE}"
)
plt.show()
plt.close()

from sklearn.metrics import classification_report, multilabel_confusion_matrix

mcm = multilabel_confusion_matrix(test_ds.labels, y_pred)

print(
    classification_report(
        test_ds.labels, y_pred, target_names=appliances, zero_division=False
    )
)
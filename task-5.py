# Only if needed uncomment the next line
# !pip install numpy keras tf-explain matplotlib

import numpy as np
import keras
from keras import layers

# from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
import matplotlib.pyplot as plt


BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 4
INPUT_SHAPE = (128, 128, 1)  # the images from the dataset are this size

# import zipfile
# with zipfile.ZipFile("sample_data/retinal-oct-sample.zip", "r") as zip_ref:
#   zip_ref.extractall("processed/retinal-oct-sample")
#   zip_ref.close()

import tensorflow as tf


# oct images of the eye are greyscale, converting them will not loose data
def preprocess_image(image, label):
    # resizing because of unecessary data, unecessarily too large, but smaller than this
    # too much data is lost - see paper about it
    image = tf.image.resize(
        image, (128, 128), method="bilinear"
    )  # method uses linear interpolation and considers the 2x2 closest neighbors
    image = tf.image.rgb_to_grayscale(
        image
    )  # needed the 3 channels for transfer leraning later
    return image, label


def resnet_preprocess(image, label):
    image = tf.image.resize(
        image, (224, 224), method="bilinear"
    )  # method uses linear interpolation and considers the 2x2 closest neighbors
    return image, label


from keras import utils

SEED = 127


test = utils.image_dataset_from_directory(
    "/kaggle/input/retinal-oct-sample-zip/test", seed=SEED, batch_size=BATCH_SIZE
)
resnet_test = test.map(resnet_preprocess)  # type: ignore
test = test.map(preprocess_image)  # type: ignore

train = utils.image_dataset_from_directory(
    "/kaggle/input/retinal-oct-sample-zip/train_sample2",
    seed=SEED,
    batch_size=BATCH_SIZE,
)
resnet_train = train.map(resnet_preprocess)  # type: ignore
train = train.map(preprocess_image)  # type: ignore


val = utils.image_dataset_from_directory(
    "/kaggle/input/retinal-oct-sample-zip/val", seed=SEED, batch_size=BATCH_SIZE
)
resnet_val = val.map(resnet_preprocess)  # type: ignore
val = val.map(preprocess_image)  # type: ignore


def separate_imglabel(dataset):
    x = []
    y = []
    for img_batch, label_batch in dataset:
        x.append(img_batch.numpy())
        y.append(label_batch.numpy())

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


x_test, y_test = separate_imglabel(test)
x_train, y_train = separate_imglabel(train)
x_val, y_val = separate_imglabel(val)

x_test_resnet, y_test = separate_imglabel(resnet_test)
x_train_resnet, y_train = separate_imglabel(resnet_train)
x_val_resnet, y_val = separate_imglabel(resnet_val)

del test, train, val

x_val_resnet.shape


print(x_train.shape)
print(x_train_resnet.shape)


# Scale images to the [0, 1] range
x_train = (
    x_train.astype("float32") / 255
)  # max value for a pixel is 255, this is to normalize it
x_test = x_test.astype("float32") / 255
x_val = x_val.astype("float32") / 255
x_train_resnet = x_train_resnet.astype("float32") / 255
x_val_resnet = x_val_resnet.astype("float32") / 255
x_test_resnet = x_test_resnet.astype("float32") / 255


#  0 - 255 values for the pixels

# Make sure images have shape (256, 256, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# x_val = np.expand_dims(x_val, -1)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("x_val shape:", x_val.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print(x_val.shape[0], "validation samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
print("y_train classes:", y_train.shape[0])
print("y_test classes:", y_test.shape[0])
print("y_val classes:", y_val.shape[0])

model = keras.Sequential(
    [
        keras.Input(shape=INPUT_SHAPE),  # (128, 128, 3) # input layer (placeholder)
        layers.Conv2D(32, kernel_size=(4, 4), activation="relu"),  # convolution 1
        # After conv, activation or pooling
        # if activation, add pooling later
        # layers.BatchNormalization(), # makes the epochs slower
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(3, 3)),  # pooling 1
        layers.Conv2D(64, kernel_size=(4, 4), activation="relu"),  # convolution 2
        # layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(3, 3)),  # pooling 2
        # removing the second layer, drops to 0.95 instead of 0.99
        layers.Flatten(),  # makes the shape have only one dimension
        # dense layer needs the input to be flat (is 1d, before it is 2d)
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),  # dropout
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])  # type: ignore


x_train.shape

# final evalutation, uses the actual test set
model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=30, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)  # type: ignore
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %% [code]
img_index = 42
image = x_test[img_index]

pred = model.predict(np.expand_dims(image, axis=0))[0]
for digit in range(10):
    print("Probability for digit {}: {}".format(digit, pred[digit]))
print("\nThe winner is {}".format(np.argmax(pred)))
print("The correct class is {}\n".format(np.argmax(y_test[img_index])))

plt.imshow(image.squeeze(), cmap="gray")
plt.show()

# %% [code]
# INPUT_SHAPE

from keras import applications

resnet = applications.ResNet50(include_top=False, weights="imagenet")
# resnet.trainable = False

model = keras.Sequential()
model.add(resnet)
model.add(layers.Flatten())
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(256, activation="relu"))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.4))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(4, activation="softmax"))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy", "AUC"])  # type: ignore

model.summary()

print(x_train_resnet.shape)
print(x_train_resnet.shape)
print(y_train.shape)
print(x_val_resnet.shape)

# print(model.input.shape)
history = model.fit(
    x_train_resnet,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=15,
    validation_data=(x_val_resnet, y_val),
)
score = model.evaluate(x_test_resnet, y_test, verbose=1)  # type: ignore
print(f"Test Loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
print(f"Test AUC: {score[2]}")

# %% [code]

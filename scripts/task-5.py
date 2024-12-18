import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Constants
TRAIN_DIR = "data/retinal-oct-sample-zip/train_sample2"
VAL_DIR = "data/retinal-oct-sample-zip/val"
TEST_DIR = "data/retinal-oct-sample-zip/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 36
SEED = 127
NUM_CLASSES = 4

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)


# Function to separate images and labels
def separate_imglabel(dataset):
    x = []
    y = []
    for img_batch, label_batch in dataset:
        x.append(img_batch.numpy())
        y.append(label_batch.numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


# Separate images and labels
x_train, y_train = separate_imglabel(train_ds)
x_val, y_val = separate_imglabel(val_ds)
x_test, y_test = separate_imglabel(test_ds)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_val = x_val.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Compute class weights using class indices
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)

# Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Print shapes for verification
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# Define the model
import keras_tuner as kt
from keras import layers


def build_cnn(hp):
    # Define the input layer
    inputs = layers.Input(shape=(128, 128, 1))

    # Initialize the first layer
    x = inputs

    # Add convolutional layers based on the hyperparameter choice
    n_layers = hp.Choice("n_layers", values=[1, 2, 3])
    for i in range(n_layers):
        x = layers.Conv2D(
            32 * (i + 1), kernel_size=(3, 3), activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

    # Flatten the output and add the final dense layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Define the optimizer
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,  # type: ignore
        loss="categorical_crossentropy",
        metrics=["accuracy", "AUC"],
    )

    model.summary()
    return model


tuner = kt.RandomSearch(
    build_cnn,
    objective="val_accuracy",
    max_trials=10,
    seed=SEED,
    directory="tuner-results",
    project_name="task4",  # need to change this to "task5"
)

tuner.search(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    class_weight=class_weights_dict,
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)  # type: ignore

# train the model with class weights and separate validation data
history_cnn = model.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    class_weight=class_weights_dict,
)

trials_results = []
for i, trial in enumerate(tuner.oracle.get_best_trials(num_trials=3)):
    trial_results = {
        "Trial Number": i + 1,
        "hyperparameters": trial.hyperparameters.values,
        "Validation Accuracy": trial.score,
    }
    trials_results.append(trial_results)

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test AUC:", score[2])

import pandas

results_df = pandas.DataFrame({"Metric": ["Loss", "Accuracy", "AUC"], "Value": score})

with open("5_cnn_results.txt", "w") as f:
    f.write(results_df.to_latex(index=False))


all_trials_results_df = pandas.DataFrame(trials_results)
hyperparameters_trials = pandas.json_normalize(all_trials_results_df["hyperparameters"])  # type: ignore
all_trials_results_df = all_trials_results_df.drop(columns=["hyperparameters"]).join(
    hyperparameters_trials
)
all_trials_results_df = all_trials_results_df.sort_values(by=["Validation Accuracy"])
all_trials_results_df.to_csv("5_hyperparametr_tuning_results.csv")

with open("5_latex_table.txt", "w") as f:
    f.write(all_trials_results_df.to_latex(index=False))

import matplotlib.pyplot as plt

# Assuming `history_cnn` is the History object returned by model.fit
# Extract the accuracy and loss values for training and validation
train_accuracy = history_cnn.history["AUC"]
val_accuracy = history_cnn.history["val_AUC"]
train_loss = history_cnn.history["loss"]
val_loss = history_cnn.history["val_loss"]

# Get the number of epochs
epochs = range(1, len(train_accuracy) + 1)

# Create a figure with two subplots side by side
plt.figure(figsize=(14, 5))

# Plot training accuracy and loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, "bo-", label="AUC")
plt.plot(epochs, train_loss, "ro-", label="Loss")
plt.title("Training")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()

# Plot validation accuracy and loss
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracy, "bo-", label="AUC")
plt.plot(epochs, val_loss, "ro-", label="Loss")
plt.title("Validation")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("5_epochs_results.png", dpi=300)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Make predictions on the test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # get the actual prediction

# Convert true labels to class indices
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)  # type: ignore
plt.savefig("5_confusion_matrix.png")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Assuming NUM_CLASSES is the number of classes
# Binarize the output
y_test_binarized = label_binarize(y_test, classes=range(NUM_CLASSES))

# Make predictions on the test set
y_pred_probs = model.predict(x_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])  # type: ignore
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
plt.savefig("5_roc_auc_curve.png", dpi=300)


from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

explainer = OcclusionSensitivity()
image = x_test[0]
print(type(image))

class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
data = ([image], None)
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(9, 3))

for i, name in enumerate(class_names):
    grid_cnn = explainer.explain(data, model, i, patch_size=8)
    axes[i].imshow(grid_cnn, cmap="viridis")
    axes[i].axis("off")
    axes[i].set_title(f"CNN - class: {name}")

plt.savefig("5_occlusion_sensitivity_cnn.png", dpi=300)


# --------------------- Define the VGG16 Transfer Learning Model ---------------------
import keras

IMG_SIZE_RESNET = (224, 224)
train_resnet = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE_RESNET,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

val_resnet = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE_RESNET,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

test_resnet = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE_RESNET,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

x_train_resnet, y_train = separate_imglabel(train_resnet)
x_val_resnet, y_val = separate_imglabel(val_resnet)
x_test_resnet, y_test = separate_imglabel(test_resnet)

# Scale images to the [0, 1] range
x_train_resnet = x_train_resnet.astype("float32") / 255
x_val_resnet = x_val_resnet.astype("float32") / 255
x_test_resnet = x_test_resnet.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Print shapes for verification
print("x_train_resnet shape:", x_train_resnet.shape)
print("y_train shape:", y_train.shape)
print("x_val_resnet shape:", x_val_resnet.shape)
print("y_val shape:", y_val.shape)
print("x_test_resnet shape:", x_test_resnet.shape)
print("y_test shape:", y_test.shape)

base_model = keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

base_model.summary()

from keras import models

transfer_model = models.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4, activation="softmax"),
    ]
)

transfer_model.summary()

from keras import optimizers

optimizer = optimizers.Adam(learning_rate=1e-6)

# Compile the model with categorical crossentropy
transfer_model.compile(
    optimizer=optimizer,  # type: ignore
    loss="categorical_crossentropy",  # Use categorical crossentropy for one-hot encoded labels
    metrics=["accuracy", "AUC"],
)
transfer_model.summary()

# Now you can train your model
history_transfer = transfer_model.fit(
    x=x_train_resnet,
    y=y_train,
    epochs=36,
    validation_data=(x_val_resnet, y_val),
    batch_size=BATCH_SIZE,
    verbose="auto",
)


# Evaluate the Custom CNN model on the test set
score_transfer = transfer_model.evaluate(x_test_resnet, y_test, verbose="auto")
print("Test loss:", score_transfer[0])
print("Test accuracy:", score_transfer[1])
print("Test AUC:", score_transfer[2])


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Make predictions on the test set
y_pred_probs = transfer_model.predict(x_test_resnet)
y_pred = np.argmax(y_pred_probs, axis=1)  # get the actual prediction

# Convert true labels to class indices
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)  # type: ignore
plt.savefig("5_confusion_matrix_transfer.png")


# Extract the accuracy and loss values for training and validation
train_accuracy = history_transfer.history["AUC"]
val_accuracy = history_transfer.history["val_AUC"]
train_loss = history_transfer.history["loss"]
val_loss = history_transfer.history["val_loss"]

# Get the number of epochs
epochs = range(1, len(train_accuracy) + 1)

# Create a figure with two subplots side by side
plt.figure(figsize=(14, 5))

# Plot training accuracy and loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, "bo-", label="AUC")
plt.plot(epochs, train_loss, "ro-", label="Loss")
plt.title("Training")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()

# Plot validation accuracy and loss
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracy, "bo-", label="AUC")
plt.plot(epochs, val_loss, "ro-", label="Loss")
plt.title("Validation")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("5_epochs_results_transfer.png", dpi=300)


explainer = OcclusionSensitivity()
image = x_test_resnet[0]

class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
data = ([image], None)
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(9, 3))

for i, name in enumerate(class_names):
    grid_cnn = explainer.explain(data, transfer_model, i, patch_size=8)
    axes[i].imshow(grid_cnn, cmap="viridis")
    axes[i].axis("off")
    axes[i].set_title(f"CNN - class: {name}")

plt.savefig("5_occlusion_sensitivity_transfer.png", dpi=300)


results_df = pandas.DataFrame(
    {"Metric": ["Loss", "Accuracy", "AUC"], "Value": score_transfer}
)

with open("5_transfer_results.txt", "w") as f:
    f.write(results_df.to_latex(index=False))

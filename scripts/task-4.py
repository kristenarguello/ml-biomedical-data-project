# %%
import keras
import io
import lzma
import tarfile

import numpy as np
import pandas
from matplotlib import pyplot as plt

import keras_tuner as kt
from sklearn import preprocessing

from sklearn.model_selection import KFold, train_test_split
import time
from keras_hub import layers

with lzma.open("data/cath.tar.xz") as f:
    cath = f.read()

with io.BytesIO(cath) as tar_buffer:
    with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
        tar.extractall(path="processed")


# open and read sq60
def clean_up_sq60_line(line: str):
    aux = line.replace("\n", "")
    aux_2 = aux.split("|")

    return aux_2[-1].split("/")[0]
    # return aux_2[-1]


sq60_lines = []
with open("processed/proteins/seqs_S60.fa", "r") as f:
    sq60_lines = [clean_up_sq60_line(l) for l in f.readlines()]

    domain_names = sq60_lines[0::2]
    sequences = sq60_lines[1::2]
sq60_df = pandas.DataFrame(
    data={"domain_name": domain_names, "sequence": sequences},
)
sq60_df = sq60_df.replace("", np.nan)


# get superfamily names
def clean_up_superfamily_name(line: str):
    aux = line.replace("\n", "")
    return aux.split("\t")[-1]


def get_cath_id(line: str):
    aux = line.replace("\n", "").split("\t")
    return aux[0]


with open("processed/proteins/superfamily_names.txt", "r") as f:
    lines = f.readlines()[1:]
    superfamily_names = [clean_up_superfamily_name(l) for l in lines]
    cath_ids = [get_cath_id(l) for l in lines]

superfamily_df = pandas.DataFrame(
    data={"cath_id": cath_ids, "superfamily_name": superfamily_names}
)
superfamily_df = superfamily_df.replace("", np.nan)


# domain classification
def create_cath_id(line: str):
    aux = line.split()
    cath_id = ""
    for i in range(4):
        cath_id += aux[i + 1] + "."
    return cath_id[:-1]


def get_domain_name(line: str):
    return line.split()[0]


with open("processed/proteins/domain_classification.txt", "r") as f:
    lines = [l for l in f.readlines() if not l.startswith("#")]
    cath_ids = [create_cath_id(l) for l in lines]
    domain_names = [get_domain_name(l) for l in lines]
    h = [l.split()[4] for l in lines]

domain_classification_df = pandas.DataFrame(
    data={"cath_id": cath_ids, "domain_name": domain_names, "h": h}
)

# merge dataframes
aux = pandas.merge(
    left=domain_classification_df, right=superfamily_df, on="cath_id", how="left"
)

single_df = pandas.merge(left=sq60_df, right=aux, on="domain_name", how="left")
single_df = single_df.dropna(subset=["superfamily_name"])

# exclude rows that have a superfamily h with less than 1000 occurrences
h_counts = single_df["h"].value_counts()
h_less_than_1000 = h_counts[h_counts < 1000]
top_5_h = h_less_than_1000.nlargest(5).index
single_df = single_df[single_df["h"].isin(top_5_h)]

# have only sequence and superfamily name on the single df
single_df = single_df[["sequence", "superfamily_name"]]
print(single_df.shape)
print(single_df.isna().sum())

# %%
single_df.sequence.str.len().describe()
# max len is 580
# min is 21

# %%


# 1) encode the superfamily names and the sequences

# make the superfamily names into number categories
lb = preprocessing.LabelBinarizer()
lb.fit(single_df.superfamily_name)
y = lb.transform(single_df.superfamily_name)

# vectorize the sequences
# lower just to make sure (vai que tem algo fora)
vectorizer = keras.layers.TextVectorization(split="character", standardize="lower")
vectorizer.adapt(single_df.sequence)
x = vectorizer(single_df.sequence)

# sequences have too different lengths, need to pad them to be uniform
# shorten some and truncate others
MAX_LENGTH = 300  # more than 75% is less than this (285)
x = keras.preprocessing.sequence.pad_sequences(x, MAX_LENGTH, truncating="post")

# %%
# transformer


TOKENS = len(vectorizer.get_vocabulary()) + 1
CLASSES = lb.classes_.shape[0]
# %%


def build_transformer(hp):
    model = keras.models.Sequential()

    emb_dim = hp.Choice("embed_dim", values=[16, 32])
    model.add(
        layers.TokenAndPositionEmbedding(TOKENS, MAX_LENGTH, emb_dim, mask_zero=True)
    )

    n_encoders = hp.Choice("n_encoders_layers", values=[2, 4])
    for i in range(n_encoders):
        # add more layers here
        num_heads = hp.Choice(f"num_heads_for_{i}_layer", values=[2, 4])
        model.add(
            layers.TransformerEncoder(
                emb_dim * 4,
                num_heads,
                # dropout=0.2,  # type: ignore # made it constant because of computational power
                activation="relu",
            )
        )
    model.add(keras.layers.GlobalAveragePooling1D())  # flatten the result

    n_dense = hp.Choice("n_dense_layers", values=[1, 2])
    for i in range(n_dense):
        hp_units = hp.Choice(f"units_for_{i}_layer", values=[32, 64])

        model.add(keras.layers.Dense(hp_units, activation="relu"))  # relu is simpler
        model.add(keras.layers.Dropout(0.2))

    model.add(
        keras.layers.Dense(CLASSES, activation="softmax")
    )  # needs to be softmax for multiple classes

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3])
    opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(
        optimizer=opt,  # type: ignore
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall", "AUC"],  # type: ignore
    )
    return model


# %%
# define tuner and stop early criteria
SEED = 34

tuner = kt.RandomSearch(
    build_transformer,
    objective="val_accuracy",
    max_trials=10,
    seed=SEED,
    directory="tuner-results",
    project_name="task4",
)

stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)


# define the number of splits for cross-validation
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# separate a test set from the training data
x_tune, x_test, y_tune, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED
)

# list to store the results of each fold
all_trials_results = []
fold_results = []

# ensure y is a numpy array
y = np.array(y)

fold = 1
for train_index, val_index in kf.split(x_tune, y_tune):
    print(f"Fold {fold}")
    start = time.time()

    x_train, x_val = x_tune[train_index], x_tune[val_index]
    y_train, y_val = y_tune[train_index], y_tune[val_index]

    # Search for the best hyperparameters
    tuner.search(
        x_train,
        y_train,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[stop_early],
    )

    # Get the best hyperparameters and build the model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)  # type: ignore

    # Train the model with the best hyperparameters
    # use early stopping
    history = model.fit(
        x_train, y_train, epochs=50, validation_data=(x_val, y_val), batch_size=32
    )

    evaluate = model.evaluate(x_test, y_test, verbose=1)

    f = {
        "Fold": fold,
        "Hyperparameters": best_hps.values,
        "Train Accuracy": history.history["accuracy"],
        "Validation Accuracy": history.history["val_accuracy"],
        "Train AUC": history.history["AUC"],
        "Validation AUC": history.history["val_AUC"],
        "Train Loss": history.history["loss"],
        "Validation Loss": history.history["val_loss"],
        "Elapsed Time": time.time() - start,
        "Evaluation": evaluate,
    }
    fold_results.append(f)

    for i, trial in enumerate(tuner.oracle.get_best_trials(num_trials=5)):
        trial_results = {
            "Fold": fold,
            "Trial Number": i + 1,
            "Trial ID": trial.trial_id,
            "hyperparameters": trial.hyperparameters.values,
            "Validation Accuracy": trial.score,
            "Elapsed Time": time.time() - start,
        }
        all_trials_results.append(trial_results)

    print(f"Elapsed time for fold {fold}: {time.time() - start}")
    fold += 1


# Calculate the average metrics across all folds
all_train_accuracies = [f["Train Accuracy"] for f in fold_results]
all_train_auc = [f["Train AUC"] for f in fold_results]
all_train_losses = [f["Train Loss"] for f in fold_results]
all_val_accuracies = [f["Validation Accuracy"] for f in fold_results]
all_val_losses = [f["Validation Loss"] for f in fold_results]
all_val_auc = [f["Validation AUC"] for f in fold_results]


# %%
average_train_accuracy = np.mean(all_train_accuracies, axis=0)
average_train_auc = np.mean(all_train_auc, axis=0)
average_train_loss = np.mean(all_train_losses, axis=0)
average_val_accuracy = np.mean(all_val_accuracies, axis=0)
average_val_loss = np.mean(all_val_losses, axis=0)
average_val_auc = np.mean(all_val_auc, axis=0)
# %%

best_epoch = np.argmin(average_val_loss)


# rerun with the best epoch amount

# list to store the results of each fold
best_epoch_all_trials_results = []
best_epoch_fold_results = []

# ensure y is a numpy array
y = np.array(y)

fold = 1
for train_index, val_index in kf.split(x_tune, y_tune):
    print(f"Fold {fold}")
    start = time.time()

    x_train, x_val = x_tune[train_index], x_tune[val_index]
    y_train, y_val = y_tune[train_index], y_tune[val_index]

    # Search for the best hyperparameters
    tuner.search(
        x_train,
        y_train,
        epochs=best_epoch,
        validation_data=(x_val, y_val),
        callbacks=[stop_early],
    )

    # Get the best hyperparameters and build the model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)  # type: ignore

    # Train the model with the best hyperparameters
    # use early stopping
    history = model.fit(
        x_train,
        y_train,
        epochs=best_epoch,
        validation_data=(x_val, y_val),
        batch_size=32,
    )

    evaluate = model.evaluate(x_test, y_test, verbose=1)

    f = {
        "Fold": fold,
        "Hyperparameters": best_hps.values,
        "Train Accuracy": history.history["accuracy"],
        "Validation Accuracy": history.history["val_accuracy"],
        "Train AUC": history.history["AUC"],
        "Validation AUC": history.history["val_AUC"],
        "Train Loss": history.history["loss"],
        "Validation Loss": history.history["val_loss"],
        "Elapsed Time": time.time() - start,
        "Evaluation": evaluate,
    }
    best_epoch_fold_results.append(f)

    for i, trial in enumerate(tuner.oracle.get_best_trials(num_trials=5)):
        trial_results = {
            "Fold": fold,
            "Trial Number": i + 1,
            "Trial ID": trial.trial_id,
            "hyperparameters": trial.hyperparameters.values,
            "Validation Accuracy": trial.score,
            "Elapsed Time": time.time() - start,
        }
        best_epoch_all_trials_results.append(trial_results)

    print(f"Elapsed time for fold {fold}: {time.time() - start}")
    fold += 1

best_epoch_train_accuracies = np.mean(
    [f["Train Accuracy"] for f in best_epoch_fold_results], axis=0
)
best_epoch_train_auc = np.mean(
    [f["Train AUC"] for f in best_epoch_fold_results], axis=0
)
best_epoch_train_losses = np.mean(
    [f["Train Loss"] for f in best_epoch_fold_results], axis=0
)
best_epoch_val_accuracies = np.mean(
    [f["Validation Accuracy"] for f in best_epoch_fold_results], axis=0
)
best_epoch_val_loss = np.mean(
    [f["Validation Loss"] for f in best_epoch_fold_results], axis=0
)
best_epoch_auc = np.mean([f["Validation AUC"] for f in best_epoch_fold_results], axis=0)

evaluation_losses = [f["Evaluation"][0] for f in best_epoch_fold_results]
evaluation_accuracies = [f["Evaluation"][1] for f in best_epoch_fold_results]
evaluation_AUC = [f["Evaluation"][4] for f in best_epoch_fold_results]

evaluation_df = pandas.DataFrame(
    {
        "Fold": [f"Fold {i+1}" for i in range(len(evaluation_losses))] + ["CV"],
        "Loss": evaluation_losses + [np.mean(evaluation_losses)],
        "Accuracy": evaluation_accuracies + [np.mean(evaluation_accuracies)],
        "AUC": evaluation_AUC + [np.mean(evaluation_AUC)],
    }
)
evaluation_df.set_index("Fold")
evaluation_df.to_csv("results/4_evaluation.csv", index=False)


columns = [
    "Train Accuracy",
    "Train AUC",
    "Train Loss",
    "Validation Accuracy",
    "Validation AUC",
    "Validation Loss",
]

average_values = {
    "Fold": "CV",
    "Train Accuracy": np.mean(average_train_accuracy),
    "Train AUC": np.mean(average_train_auc),
    "Train Loss": np.mean(average_train_loss),
    "Validation Accuracy": np.mean(average_val_accuracy),
    "Validation AUC": np.mean(average_val_auc),
    "Validation Loss": np.mean(average_val_loss),
}
fold_results.append(average_values)


fold_results_df = pandas.DataFrame(fold_results)
hyperparameters_fold = pandas.json_normalize(fold_results_df["Hyperparameters"])  # type: ignore
fold_results_df = fold_results_df.drop(columns=["Hyperparameters", "Evaluation"]).join(
    hyperparameters_fold
)
for col in columns:
    for i in range(4):
        mean_value = np.mean(fold_results_df[col][i])
        fold_results_df.loc[i, col] = mean_value


all_trials_results_df = pandas.DataFrame(all_trials_results)
hyperparameters_trials = pandas.json_normalize(all_trials_results_df["hyperparameters"])  # type: ignore
all_trials_results_df = all_trials_results_df.drop(columns=["hyperparameters"]).join(
    hyperparameters_trials
)
all_trials_results_df = all_trials_results_df.sort_values(
    by=["Fold", "Validation Accuracy"], ascending=[True, False]
)

all_trials_results_df.to_csv("results/4_all_trials_results.csv", index=False)
fold_results_df.to_csv("results/4_fold_results.csv", index=False)

all_trials_results_df = all_trials_results_df.drop(columns=["Trial ID"])
dfs = [all_trials_results_df, fold_results_df, evaluation_df]

# make latex tables
with open("results/4_latex_tables.txt", "w") as f:
    latex = ""
    for df in dfs:
        latex += df.to_latex(index=False)
        latex += "\n\n"
    f.write(latex)


# Plotting
epochs = range(1, len(average_train_accuracy) + 1)
all_values = (
    list(average_train_auc)
    + list(average_val_auc)
    + list(average_train_loss)
    + list(average_val_loss)
)
y_min = min(all_values)
y_max = max(all_values)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training accuracy and loss
ax1.plot(epochs, average_train_auc, "bo-", label="Training AUC")
ax1.plot(epochs, average_train_loss, "ro-", label="Training Loss")
ax1.set_title("Training")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Value")
ax1.set_ylim(y_min, y_max)
ax1.legend()

# Plot validation accuracy and loss
ax2.plot(epochs, average_val_auc, "bo-", label="Validation AUC")
ax2.plot(epochs, average_val_loss, "ro-", label="Validation Loss")
ax2.set_title("Validation")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Value")
ax2.set_ylim(y_min, y_max)
ax2.legend()

# Show the plots
plt.tight_layout()
# plt.show()
plt.savefig("results/4_nn_performance_50_epochs.png", bbox_inches="tight")

epochs = range(1, best_epoch + 1)
all_values = (
    list(best_epoch_train_auc)
    + list(best_epoch_train_losses)
    + list(best_epoch_auc)
    + list(best_epoch_val_loss)
)
y_min = min(all_values)
y_max = max(all_values)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training accuracy and loss
ax1.plot(epochs, best_epoch_train_auc, "bo-", label="Training AUC")
ax1.plot(epochs, best_epoch_train_losses, "ro-", label="Training Loss")
ax1.set_title("Training")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Value")
ax1.set_ylim(y_min, y_max)
ax1.legend()

# Plot validation accuracy and loss
ax2.plot(epochs, best_epoch_auc, "bo-", label="Validation AUC")
ax2.plot(epochs, best_epoch_val_loss, "ro-", label="Validation Loss")
ax2.set_title("Validation")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Value")
ax2.set_ylim(y_min, y_max)
ax2.legend()

# Show the plots
plt.tight_layout()
# plt.show()
plt.savefig("results/4_nn_performance_best_epoch.png", bbox_inches="tight")

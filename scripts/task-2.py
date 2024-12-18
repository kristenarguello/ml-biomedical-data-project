# %%
import io
import lzma
import os
import tarfile

from matplotlib import pyplot as plt
import numpy as np
import pandas

## DATA PREPROCESSING
with lzma.open("data/visits.tar.xz") as f:
    visits = f.read()

with io.BytesIO(visits) as tar_buffer:
    with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
        tar.extractall(path="processed")


visits_csvs = []
folder_path = "processed/visits"
# iterate throguh the visits file and get the data
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        with open(f"{folder_path}/{filename}", "r") as f:
            visits_csvs.append(pandas.read_csv(f))

# because you will only need the dementia rating from the follow-up visits,
# you can delete other attributes there before joining the dataframes on patient ID
for i in range(1, len(visits_csvs)):
    visits_csvs[i] = visits_csvs[i][["ID", "CDR"]]

# merge all into one df
data = visits_csvs[0]
for i in range(1, len(visits_csvs)):
    data = data.merge(visits_csvs[i], on="ID", how="left", suffixes=("", f"_{i+1}"))
del visits_csvs

# set ID as index
data.set_index("ID", inplace=True)

# drop MRI_ID, visit, delay and hand, not needed for prediction = all the same for every patient
data = data.drop(columns=["MRI_ID", "visit", "delay"])

# fill the blanks for SES with the mode
data["SES"] = data["SES"].fillna(data["SES"].mode()[0])


# there are typos in CDR and CDR 2 columns (checked with value counts)
possible_values = "none, very mild, mild, moderate, severe".split(", ")
# TYPOSSSSSS
for column in data.columns:
    if "CDR" in column:
        for value in data[column]:
            data[column] = data[column].replace("very midl", "very mild")
            data[column] = data[column].replace("vry mild", "very mild")
            data[column] = data[column].replace("very miId", "very mild")
            data[column] = data[column].replace("midl", "mild")

# change the data types of the categorical columns and other values to float
# nominal categories
nominal_cat = "sex hand".split()
for column in nominal_cat:
    data[column] = data[column].astype("category").cat.codes.astype(int)

# ordinal categories
cdr_map = {}
for i, category in enumerate(possible_values):
    cdr_map[category] = i

cdr_ordinal = "CDR CDR_2 CDR_3 CDR_4 CDR_5".split()
for column in cdr_ordinal:
    data[column] = data[column].map(cdr_map).fillna(-1).astype(int)

ses_map = {}
for i in data["SES"].unique():
    ses_map[i] = int(i)
data["SES"] = data["SES"].map(ses_map)


data["ASF"] = data["ASF"].str.replace(",", ".")
data["ASF"] = data["ASF"].astype(float)

# %%
# create new class: got worse or not
initial_values = data["CDR"]
other_visits = data[["CDR_2", "CDR_3", "CDR_4", "CDR_5"]]
patients_ids = data.index

# for each patient, check if the dementia rating got worse at any given point
for i in patients_ids:
    initial_value = initial_values[i]
    other_values = other_visits.loc[i]
    if initial_value < other_values.max():
        data.loc[i, "worsening"] = 1  # 1 if got worse
    else:
        data.loc[i, "worsening"] = 0
    del other_values, initial_value
# if the patient got better at a given moment, still got worse, an improvement doesnt indicate anything it could be a number of other reasons, but the patient still got worse at some point

data["worsening"] = data["worsening"].astype(int)

del initial_values, other_visits, patients_ids


def learning_performance_mean_CI(cv_results, penalty_str: str, c: float | str):
    # roc auc
    roc_auc_scores = cv_results["mean_test_roc_auc"]
    mean_roc_auc = np.mean(roc_auc_scores)
    std_error_roc_auc = np.std(roc_auc_scores, ddof=1) / np.sqrt(len(roc_auc_scores))
    ci_lower_roc_auc = mean_roc_auc - 1.96 * std_error_roc_auc
    ci_upper_roc_auc = mean_roc_auc + 1.96 * std_error_roc_auc

    # recall
    f1_scores = cv_results["mean_test_f1"]
    mean_f1 = np.mean(f1_scores)
    std_error = np.std(f1_scores, ddof=1) / np.sqrt(len(f1_scores))
    ci_lower = mean_f1 - 1.96 * std_error
    ci_upper = mean_f1 + 1.96 * std_error

    # 1.96 is the z value for 95% confidence interval
    # used because of the normalizatino made by the standard scaler
    # sampe size bigger than 30

    return {
        "Penalty": penalty_str.upper(),
        "Mean ROC-AUC": mean_roc_auc,
        "ROC-AUC CI Lower": ci_lower_roc_auc,
        "ROC-AUC CI Upper": ci_upper_roc_auc,
        "Mean F1": mean_f1,
        "F1 CI Lower": ci_lower,
        "F1 CI Upper": ci_upper,
        "C": c,
    }


# %%
## LOGISTIC REGRESSION
# logistic regression without regularisation (mean + 95% CI)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

SEED = 42

# predict if the patient got worse after the first visit
# the other CDR values need to go to prevent target leakage (access to data that will not be present when predictin real world problems)
# the model needs to have access to data it would have access to in the real world: data from the first visit only (if the patient got worse the model will not have access to in a real world situation)
x = data.drop(columns=["CDR_2", "CDR_3", "CDR_4", "CDR_5", "worsening"])
y = data["worsening"]
# 1 = 119
# 0 = 31

# getting high values are difficult
# too imbalanced

# %%
# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED
)
# small dataset needs to have a small test size


# %%
# pipeline
pipe = Pipeline(
    [
        (
            "scaler",
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            # robust scaler was the worst one
        ),
        ("lr", LogisticRegression(random_state=SEED)),
    ]
)

# linear regression = used to predict a continuous value
# logistic regression = used to predict a binary value

# regularization = adding a penalty to the loss function to prevent overfitting CRITICAL

# solvers:
# - liblinear = small datasets
# Algorithm to use in the optimization problem.
# newton cholesky and liblinear only handles binomial prediction
# newton-cholesky’ is a good choice for n_samples >> n_features,

# newton cholesky is the best choice: especially with one-hot encoded categorical features with rare categories.
# Be aware that the memory usage of this solver has a quadratic dependency
# on n_features because it explicitly computes the Hessian matrix.


# grid search for lr
param_grid = [
    {
        "lr__penalty": [None],  # no regularization
        "lr__solver": [
            "lbfgs",
            "newton-cg",
            # "newton-cholesky",
            # "sag",  # sag and saga need scaled data - sag not used because small datasets, not efficient
            # "saga",
        ],  # solvers that support none as penalty
        "lr__class_weight": ["balanced", None],  # for imbalanced data
        "lr__fit_intercept": [True, False],
        # C is not here because with no regularization, C is not used, will not change the outcome
    },
]
# lr class weight as none makes a model that predicts always 1, since the dataset is really imbalanced

cv = model_selection.StratifiedKFold(
    15, shuffle=True, random_state=SEED
)  # make it separetly to get shuffle as true
# Avoiding Overfitting: Shuffling helps prevent any potential overfitting that might occur if there were any hidden patterns in the order of the data (e.g., if IDs were assigned in a way that correlates with the target variable).


#  GridSearchCV is the process of performing hyperparameter tuning in order to determine the optimal values for a given model.
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,  # higher cv = more accurate but slower, got balanced as the better param (needs to be true)
    n_jobs=-1,
    scoring=["roc_auc", "f1"],  # best metrics for imbalanced data
    refit="roc_auc",
    error_score="raise",
    return_train_score=True,
)
grid.fit(x_train, y_train)
pred = grid.predict(x_test)
grid.best_score_

# %%
# use different regularisation penalties
# l2
param_grid_l2 = {
    "lr__penalty": ["l2"],
    "lr__C": np.arange(0.1, 1, 0.1),  # regularization strength
    "lr__solver": [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        # not including sag and saga because of the small dataset
    ],
    "lr__class_weight": ["balanced", None],  # for imbalanced data
    "lr__fit_intercept": [True, False],
}
grid_l2 = GridSearchCV(
    pipe,
    param_grid_l2,
    cv=cv,
    n_jobs=-1,
    scoring=["roc_auc", "f1"],
    refit="roc_auc",
    error_score="raise",
    return_train_score=True,
)
grid_l2.fit(x_train, y_train)
pred_l2 = grid_l2.predict(x_test)
grid_l2.best_score_

# %%
# l1
param_grid_l1 = {
    "lr__penalty": ["l1"],
    "lr__C": np.arange(0.1, 1, 0.1),  # regularization strength
    "lr__solver": ["liblinear"],
    "lr__fit_intercept": [True, False],
    "lr__intercept_scaling": np.arange(0.1, 5, 0.25),
    "lr__class_weight": ["balanced", None],
}
grid_l1 = GridSearchCV(
    pipe,
    param_grid_l1,
    cv=cv,
    n_jobs=-1,
    scoring=["roc_auc", "f1"],
    refit="roc_auc",
    error_score="raise",
    return_train_score=True,
)
grid_l1.fit(x_train, y_train)
pred_l1 = grid_l1.predict(x_test)
grid_l1.best_score_

# %%
results = [
    learning_performance_mean_CI(grid.cv_results_, "no", "N/A"),
    learning_performance_mean_CI(
        grid_l1.cv_results_, "l1", grid_l1.best_params_["lr__C"]
    ),
    learning_performance_mean_CI(
        grid_l2.cv_results_, "l2", grid_l2.best_params_["lr__C"]
    ),
]

# create a df and use to make a latex table
df_results = pandas.DataFrame(results)
df_results = df_results.round(3)

# iter throgu rows
for i, row in df_results.iterrows():
    interval = f"[{row['ROC-AUC CI Lower']} , {row['ROC-AUC CI Upper']}]"
    df_results.at[i, "ROC-AUC CI"] = interval

    interval = f"[{row['F1 CI Lower']} , {row['F1 CI Upper']}]"
    df_results.at[i, "F1 CI"] = interval


df_results = df_results.drop(
    columns=["ROC-AUC CI Lower", "ROC-AUC CI Upper", "F1 CI Lower", "F1 CI Upper", "C"]
)
# reorder columns to be mean and CI of each metric together
df_results = df_results[["Penalty", "Mean ROC-AUC", "ROC-AUC CI", "Mean F1", "F1 CI"]]

df_results.to_csv("results/2_logistic_regression_results.csv", index=False)
with open("results/2_logistic_regression_results_latex_table.txt", "w") as f:
    f.write(df_results.to_latex(index=False))


# In such cases, evaluation metrics like ROC-AUC curve are a good indicator of
# classifier performance. It is a measure of how good model is at distinguishing
#  between various class. Higher the ROC-AUC score, better the model is at
# predicting 0s as 0s and 1s as 1s. Just to remind, ROC is a probability curve
# and AUC represents degree or measure of separability. Apart from this metric,
# we will also check on recall score, false-positive (FP) and false-negative (FN)
# score as we build our classifier.
# https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b

# %%
# plot of the disease progression over time - CDR values over time = for each visit (_2, _3, _4, _5)
import seaborn as sns

df = data[["CDR", "CDR_2", "CDR_3", "CDR_4", "CDR_5"]]
# consider patients that didnt go to the visits
df.replace(-1, np.nan, inplace=True)
df = df.ffill(axis=1)


# %%


grouped = df.groupby("CDR")

# Determine the number of unique initial CDR values
num_groups = len(grouped)

# Create subplots
fig, axes = plt.subplots(
    nrows=1,
    ncols=num_groups,
    figsize=(5 * num_groups, 6),
    constrained_layout=True,
    sharey=True,
)

cdr_categories = ["None", "Very Mild", "Mild", "Moderate", "Severe"]

# Plotting heatmaps for each group
for ax, (cdr_value, group) in zip(axes, grouped):
    # Calculate proportions for each CDR category across visits
    proportions = group.apply(lambda x: x.value_counts()).fillna(0)
    for category in range(5):
        if category not in proportions.index:
            proportions.loc[category] = 0

    # Plot heatmap
    sns.heatmap(
        proportions,
        annot=True,
        cmap="coolwarm",
        cbar=(ax == axes[-1]),
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )
    ax.set_title(f"Initial CDR: {cdr_categories[int(cdr_value)]}")
    # ax.set_xlabel("Visit")
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticklabels(
        ["Visit 1", "Visit 2", "Visit 3", "Visit 4", "Visit 5"], rotation=0
    )
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(["None", "Very Mild", "Mild", "Moderate", "Severe"], rotation=0)

# plt.suptitle("Disease Progression Trends by Initial CDR Level", fontsize=16)

plt.savefig("results/2_disease_progression_over_time.png", bbox_inches="tight", dpi=300)

# %%
# make a df with the value count distribution
df_value_counts = data["worsening"].value_counts().reset_index()
df_value_counts.columns = ["Worsening", "Proportion"]
df_value_counts["Worsening"] = df_value_counts["Worsening"].replace({0: "No", 1: "Yes"})
df_value_counts.to_csv("results/2_worsening_value_counts.csv", index=False)

# to latex
with open("results/2_worsening_value_counts_latex_table.txt", "w") as f:
    f.write(df_value_counts.to_latex(index=False))

# %%

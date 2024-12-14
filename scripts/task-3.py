# %%
import io
import lzma
import tarfile
import time

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.svm import SVC

# %%
start_ever = time.time()

with lzma.open("data/appointments.tar.xz") as f:
    appointments = f.read()

with io.BytesIO(appointments) as tar_buffer:
    with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
        tar.extractall(path="processed")

# read appointments
with open("processed/appointments/appointments.txt", "r") as f:
    appointments_lines = f.readlines()


cleaned_data = [line.strip().split() for line in appointments_lines]
data = pandas.DataFrame(cleaned_data[1:], columns=cleaned_data[0])

# read participants
with open("processed/appointments/participants.txt", "r") as f:
    participants_lines = f.readlines()

cleaned_data = [line.strip().split() for line in participants_lines]
participants = pandas.DataFrame(cleaned_data[1:], columns=cleaned_data[0])

# merge appointments and participants using left join to not loose any data and to have more rows instead of more columns
data = data.merge(participants, on="participant", how="left")

# drop rows with count < 5
data = data[data["count"].astype(int) >= 5]
data = data.reset_index(drop=True)


# changing data types to numbers and categories
numbers = ["participant advance day month age count".split()]
for column in numbers:
    data[column] = data[column].astype(int)

categories = [
    "sms_received weekday status sex hipertension diabetes alcoholism".split()
]
for category in categories:
    data[category] = data[category].astype("category")


data = data.drop(columns=["participant"])
# if in a real world case scenario, youlll prob recieve data about a patient that has never been there, so the number of the patient shouldng matter for the prediction
# saw that it mattered too much on the feature importance chart

# %%
from sklearn import feature_selection, impute, model_selection, pipeline, preprocessing
from sklearn.compose import ColumnTransformer

SEED = 127

# advanced pipeline
y = data["status"].cat.codes
x = data.drop("status", axis=1)


# imbalanced data!
# 0 (fullfilled) = 14942
# 1 (no-show) = 3896
# trying to predict no show, so it is a 1

# transform categorical and numerical data as part of the pipeline
categorical = x.columns[x.dtypes == "category"]
numeric = x.columns.difference(categorical)

# try out different imputers
imputer_num = impute.KNNImputer(weights="distance")
imputer_cat = impute.SimpleImputer(strategy="most_frequent")

scaler = preprocessing.MinMaxScaler()  # need to try out with different scalers
encoder = preprocessing.OneHotEncoder(
    handle_unknown="infrequent_if_exist", min_frequency=0.01
)
# setting this for sat that appears twice in the whole dataset
# if a category appears less than 1% of the time, it will be considered unknown - SAT appears twice only
# the transformer was not working when encountering on the test and not on the train
# one hot encoder makes more sense because of the non-ordinal nature of the categories

transformer = ColumnTransformer(
    transformers=[
        ("PN", pipeline.make_pipeline(imputer_num, scaler), numeric),
        ("PC", pipeline.make_pipeline(imputer_cat, encoder), categorical),
    ]
)
transformer.fit_transform(x)

# selector = feature_selection.SelectPercentile(feature_selection.mutual_info_classif)
selector = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k=10)
selector.fit_transform(transformer.fit_transform(x), y)
# Sometimes removing some less important features in the training set,
# that is, selecting the top ’n’ best features will improve the accuracy
# of the model, this is called feature selection. Pipelines can be used
# for feature selection and thus help in improving the accuracies by
# eliminating the unnecessary or least important features.


outer_cv = model_selection.StratifiedKFold(6, shuffle=True, random_state=SEED)
inner_cv = model_selection.StratifiedKFold(2, shuffle=True, random_state=SEED + 1)

print("preprocessing ready and cv variables done")
# %% SEE WHAT COMINATION OF HYPERPARAMETERS AND WHICH ALGORITHM WORKS BEST
# need to try out different algorithms: save in a dict to later comparison

all_results = {}
pipes: dict[str, pipeline.Pipeline] = {}
models = {
    "svc": SVC(random_state=SEED),
    "rf": RandomForestClassifier(random_state=SEED),
    "gb": GradientBoostingClassifier(random_state=SEED),
}
print("Starting hyperparameter tuning")

# chose this ones because they are the most efficient to classification
param_grid = {
    "svc": [
        {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 0.5, 0.7, 1],
            "class_weight": ["balanced"],
            "gamma": ["scale", "auto"],
        },
    ],
    "rf": [
        {
            "n_estimators": [300, 400],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 3],
            "bootstrap": [True],
            "criterion": ["gini"],
            "class_weight": ["balanced"],
        }
    ],
    "gb": {
        "learning_rate": [0.1, 0.5],
        "n_estimators": [300, 400],
        "max_depth": [10, None],
    },
}

scoring = [
    "roc_auc",  # model's ability to distinguish between the classes across all thresholds. higher AUC indicates better performance in distinguishing between those who will and will not miss appointments.
    "f1",  # harmonic mean of precision and recall. It is useful when you want a balance between precision and recall, especially in cases of imbalanced datasets.
    # precision = TP / (TP + FP) - how many of the predicted positive are actually positive
    # recall = TP / (TP + FN) - how many of the actual positive are predicted positive
]  # roc auc still the best for imbalanced data

start = time.time()

for key, value in models.items():
    print("Starting model:", key)

    grid = model_selection.GridSearchCV(
        value,
        param_grid=param_grid[key],
        scoring=scoring,
        refit="f1",
        cv=inner_cv,
        error_score="raise",
        n_jobs=-1,
    )

    # just need to tune the hyperparameters from the model, not other stuff from the pipeline
    pipe = pipeline.Pipeline(
        [
            ("transformer", transformer),
            ("selector", selector),
            ("classifier", grid),
        ]
    )

    pipes[key] = pipe

    # get cross validation results (including estimators, train scores and dataset indices)
    cross_result = model_selection.cross_validate(
        pipe,
        x,
        y,
        return_indices=True,  # type: ignore
        scoring=scoring,
        cv=outer_cv,
        # n_jobs=-1,
        return_estimator=True,
        return_train_score=True,
        verbose=3,
    )
    all_results[key] = cross_result

    end = time.time()
    print(f"TIME ELAPSED: {time.strftime("%H:%M:%S", time.gmtime(end - start))}")
    print("Finishing model:", key)
print("Hyperparameter tuning ready")

best_algorithm = max(
    all_results,
    key=lambda key: all_results[key]["test_f1"].mean(),
)
print("Best algorithm:", best_algorithm)
for key, value in all_results.items():
    print(f"{key}: {value["test_f1"].mean()} as f1 at cross validation")


# %%
# PLOT THE PRECISION RECALL CURVE
# plot of the precision-recall curve (for test folds)
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay


print("Starting precision recall curve")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # Adjust the figsize as needed
axes = axes.flatten()

y_true_pooled = []
y_pred_pooled = []
# iterar pelo dict de resposta, pegar o estimator e fazer precision
estimators = all_results[best_algorithm]["estimator"]
indices = all_results[best_algorithm]["indices"]
for i, estimator in enumerate(estimators):
    x_test = x.iloc[indices["test"][i]]
    y_test = y.iloc[indices["test"][i]]

    aux_pipe = pipeline.Pipeline(
        [
            ("transformer", transformer),
            ("selector", selector),
            ("classifier", estimator["classifier"]),
        ]
    )

    y_true_fold = y.iloc[indices["test"][i]]
    y_pred_fold = aux_pipe.predict(x_test)

    y_true_pooled.extend(y_true_fold)
    y_pred_pooled.extend(y_pred_fold)

    # get true labels and predicted probabilities for the current fold
    # Calculate precision-recall curve
    curve = PrecisionRecallDisplay.from_estimator(
        estimator,
        x_test,
        y_test,
        name=f"Fold {i+1}",
        ax=axes[i],
    )

plt.tight_layout()
plt.savefig("results/3_precision_recall_curve_folds.png", bbox_inches="tight", dpi=300)
print("Precision recall curve ready")

disp = ConfusionMatrixDisplay.from_predictions(
    y_true_pooled, y_pred_pooled, display_labels=["Fullfilled", "No-show"]
)
# disp.plot(cmap="Blues")
plt.savefig("results/3_combined_confusion_matrices.png", bbox_inches="tight", dpi=300)

# reasons for the rapid fall and then slow one:
# Class Imbalance: If the dataset is imbalanced with a large number of negative samples compared to positive ones, the model might initially classify many negative samples as positive, leading to high precision at low recall. As recall increases, more false positives are included, causing precision to drop.


# PLOT THE FEATURES IMPACT ON THE MODEL OUTPUT (BEST MODEL / ALL DATA / NO-SHOW) - maybe do the mean value of the feature importances?
# dont need to use cross validation here
# making a model on the all of the data
from sklearn import inspection

print("Starting feature importances")

best_pipe = pipes[best_algorithm]
validation_scores = all_results[best_algorithm]["test_f1"]
best_index = np.argmax(validation_scores)
best_pipe.fit(x, y)  # all data
print("Best model trained")

pi = inspection.permutation_importance(
    best_pipe,
    x,
    y,
    scoring="f1",
    random_state=SEED,
)
fig, p = plt.subplots()
order = pi.importances_mean.argsort()  # type: ignore
p.barh(x.columns[order], pi.importances_mean[order], xerr=pi.importances_std[order])  # type: ignore
plt.savefig(
    "results/3_feature_importances_permutation.png", bbox_inches="tight", dpi=300
)
print("Feature importances ready")

# %%
# import shap

# start = time.time()
# explainer = shap.Explainer(estimator_shap)
# explanation = explainer(transformer.fit_transform(x))
# # %%
# shap_values = explanation.values[:, 1]  # just class 1
# # %%
# expected = explanation.base_values[1]
# explanation = shap.Explanation(
#     shap_values, expected, data=x.values, feature_names=x.columns
# )
# # %%
# shap.plots.bar(explanation)
# plt.savefig("results/3_feature_importances_shap.png", bbox_inches="tight", dpi=300)
# end = time.time()
# print(f"TIME ELAPSED FOR SHAP EXPLANATION: {end - start}")

# %%


## EXTENSIONS
# %% STACK THE MODELS
# transform the dict into a list of tuples
print("Starting stacking model")
stacking_estimators = []
for key, value in all_results.items():
    if key == best_algorithm:
        continue
    best_estimator = value["estimator"][0]["classifier"].best_estimator_
    stacking_estimators.append((key, best_estimator))


stack = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=None,
)  # use logistic regression as the final one
stacking_pipe = pipeline.Pipeline(
    [
        ("transformer", transformer),
        ("selector", selector),
        ("classifier", stack),
    ]
)
print("Stacking model created")

# stacking_score = model_selection.cross_validate(
#     stacking_pipe, x, y, cv=outer_cv, scoring=scoring, verbose=5
# )

# %% plot learning curves for stacking model and the best raw  (performance for increasing size of the training set)
from sklearn.model_selection import learning_curve

leraning_curve_time = time.time()

print("Starting learning curves")


def plot_learning_curve(estimator, title, X, y, cv, scoring, ax, n_jobs=-1):
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(  # type: ignore
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5),
        verbose=3,
        n_jobs=n_jobs,
    )
    print("Learning curve ready for ", title)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.grid()

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    ax.legend(loc="best")


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

algorihtm_name_mapping = {
    "svc": "Support Vector Classifier",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

best_pipe_estimator = best_pipe.named_steps["classifier"].best_estimator_
# created a pipeline with the already hyperparemetr tuned model
new_pipe = pipeline.Pipeline(
    [
        ("transformer", transformer),
        ("selector", selector),
        ("classifier", best_pipe_estimator),
    ]
)


# Plot the first learning curve
plot_learning_curve(
    new_pipe,
    f"Algorithm: {algorihtm_name_mapping[best_algorithm]}",
    x,
    y,
    cv=outer_cv,
    scoring="roc_auc",
    ax=axes[0],
)
print("First learning curve ready")

# Plot the second learning curve
plot_learning_curve(
    stacking_pipe,
    "Algorithm: Stacking Classifier",
    x,
    y,
    cv=outer_cv,
    scoring="roc_auc",
    ax=axes[1],
)
learning_curve_time = time.time() - leraning_curve_time
print(
    f"TIME ELAPSED FOR LEARNING CURVES: {time.strftime('%H:%M:%S', time.gmtime(learning_curve_time))}"
)

# Save the figure
plt.savefig("results/3_learning_curves.png", bbox_inches="tight", dpi=300)
print("Learning curves ready")

# %% ERROR MATRIX FOR POOLED PREDICITONS
# confusion matrix for combined predictions from all test folds
# from sklearn.metrics import ConfusionMatrixDisplay


# # need to pool it, this is not pooling
# print("Starting confusion matrix")

# # Initialize lists to collect true and predicted labels
# y_true_pooled = []
# y_pred_pooled = []

# # Iterate over the estimators from the best algorithm's cross-validation results
# for i, estimator in enumerate(all_results[best_algorithm]["estimator"]):
#     # Get the test indices for the current fold
#     test_indices = indices["test"][i]

#     # Get the true labels for the current fold
#     y_true_fold = y.iloc[test_indices]

#     # Get the predicted labels for the current fold
#     y_pred_fold = estimator["classifier"].predict(x.iloc[test_indices])

#     # Append the true and predicted labels to the pooled lists
#     y_true_pooled.extend(y_true_fold)
#     y_pred_pooled.extend(y_pred_fold)

# # Plot the confusion matrix
# disp = ConfusionMatrixDisplay.from_predictions(
#     y_true_pooled, y_pred_pooled, display_labels=["Fullfilled", "No-show"]
# )
# # disp.plot(cmap="Blues")
# plt.savefig("results/3_combined_confusion_matrices.png", bbox_inches="tight", dpi=300)

# print("Confusion matrix ready")


# %%
# add stacking cross validate results
# create a df with those results
results_df = pandas.DataFrame(
    {
        "Algorithm": list(all_results.keys()),
        "ROC AUC": [value["test_roc_auc"].mean() for value in all_results.values()],
        # "Recall": [value["test_recall"].mean() for value in all_results.values()]
        # + [stacking_score["test_recall"].mean()],
        "F1": [value["test_f1"].mean() for value in all_results.values()],
        # "Precision": [value["test_precision"].mean() for value in all_results.values()]
        # + [stacking_score["test_precision"].mean()],
    }
)
results_df.to_csv("results/3_algorithms_performances.csv", index=False)
with open("results/3_algorithms_performances_latex_table.txt", "w") as f:
    f.write(results_df.to_latex(index=False))

end_ever = time.time()
# format time in hours, minutes and seconds
print(
    "EVERYTHING READY - time elapsed: ",
    time.strftime("%H:%M:%S", time.gmtime(end_ever - start_ever)),
)

# %%

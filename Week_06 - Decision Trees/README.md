# Goals for week 06

1. Practice using decision trees.
2. Implement a classification decision tree.

## Data Science

Learning how to model data effectively.

### Task 01

**Description:**

A healthcare research team wants to improve early detection of breast cancer. They reach out to your team for help in understanding the factors that influence the diagnosis. The dataset they provide you is available in `sklearn`: [`datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

Perform exploratory data analysis to see how the features can be used to predict an individual's diagnosis.

**Acceptance criteria:**

1. An Excel file showing the data audit is produced.

### Task 02

**Description:**

Build a classification model that predicts whether a tumor is malignant or benign using the breast cancer dataset. Create the model report file. Conduct experiments:

- with all classification models we've learned so far;
- using all and some of the features;
- that show the impact of feature selection on model performance;
- with ensembling different models into one using the [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

**Acceptance criteria:**

1. An Excel file showing the model report is produced.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. Model ensembling is performed via the class [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

### Task 03

**Description:**

A car sales company wants to improve the quality of the cars they have on sale by offering only the ones with the highest miles per gallon. The dataset they provide you is available in our folder `DATA`: `auto.csv`.

Perform exploratory data analysis to see how the features can be used to predict a car's miles per gallon (`mpg`).

**Acceptance criteria:**

1. An Excel file showing the data audit is produced.

### Task 04

**Description:**

Build a regression model that predicts a car's miles per gallon using the dataset `auto.csv`. Conduct experiments:

- with all regression models we've learned so far;
- using all and some of the features;
- that show the impact of feature selection on model performance;
- with ensembling different models into one using the [`VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html).

**Acceptance criteria:**

1. An Excel file showing the model report is produced.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. Model ensembling is performed via the class [`VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html).

### Task 05

**Description:**

We'll now move on to work with the [Indian Liver Patient Dataset](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) from the UCI Machine learning repository. It can be found in our `DATA` folder under the name `indian_liver_patient_dataset.csv`. When you load it, you'll notice that there are no column labels. Add them using the information in the UCI Machine learning repository.

Name the target feature `has_liver_disease`. Encode it with `0` for healthy and `1` for a patient with liver disease. A healthy patient is encoded with the value `2` in the column `'Selector'`.

Perform exploratory data analysis to see how the features can be used to predict whether a person has liver disease.

**Acceptance criteria:**

1. An Excel file showing the data audit is produced.

### Task 06

**Description:**

Build a classification model that predicts whether a person has liver disease. Create the model report file. Conduct experiments:

- with all classification models we've learned so far;
- using all and some of the features;
- that show the impact of feature selection on model performance;
- with ensembling different models into one using the [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

**Acceptance criteria:**

1. An Excel file showing the model report is produced.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. Model ensembling is performed via the class [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

## Engineering

Building systems and implementing models.

### Task 01

**Description:**

Let's implement the most popular entropy functions: `gini_index` and `entropy`. Add them to the module `stats`. They should accept `numpy` arrays and return Python `float`s.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `gini_index` is added to `ml_lib.stats`.
2. A function `entropy` is added to `ml_lib.stats`.
3. The functions accept `numpy` arrays.
4. The functions return Python `float`s.

### Task 02

**Description:**

Implement a decision tree for classification. Name the class `DecisionTreeClassifier` and put it in a new module `tree`. It should be configurable via the following parameters:

- `min_samples_leaf`;
- `min_samples_split`;
- `max_depth`;
- `criterion`.

Often when implementing non-linear data structures, it is helpful to have a class for each element in the tree. This is the case with our task as well - it is encouraged that you implement a `struct`-like class `Node` that will hold metainformation for each node.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A class `DecisionTreeClassifier` is added in the module `ml_lib.tree`.

### Task 03

**Description:**

Use your model to recreate the best decision tree in `Task 06`.

**Acceptance criteria:**

1. It is shown that the implementation can be used to recreate the results obtained via `sklearn`'s `DecisionTreeClassifier`.

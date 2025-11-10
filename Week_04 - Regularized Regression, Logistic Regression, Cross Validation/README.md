# Goals for week 04

1. Practice cross validation.
2. Practice regularized regression.
3. Practice classification metrics.
4. Practice creating and interpreting the ROC curve.
5. Practice performing hyperparameter optimization/tuning.
6. Practice building and using a regularized regression models.
7. Practice building and using a logistic regression model.

## Data Science

Learning how to model data effectively.

> **Note:** In all of this week's tasks, utilize only numerical features.

### Task 01

**Description:**

Last week you created a linear regression model on the dataset `advertising_and_sales_clean.csv`. Let's continue that modelling part with a comparison between the performance of your model and regularized alternatives.

Create a new file `model_report.xlsx` with experiments involving Ridge regression and Lasso regression. Train your models using cross validation and perform hyperparameter tuning for the parameters you deem important and want to explore. For each of your experiments, add one column before your column `Comments`. In it paste a horizontal bar chart sorted by the `x`-axis showing the mean train and test scores. The `y`-axis should have as labels the parameters that were used for the model.

> **Note:** The base model in this task should be the best linear regression model you got last week.

Analyze the coefficients of the best Lasso model you obtain - which is the most important numeric feature to predict `sales`? Paste the answer in your column `Comments`.

**Acceptance criteria:**

1. Models are trained via cross validation.
2. Hyperparameter tuning is performed for each model.
3. Horizontal bar charts are present for each experiment in which hyperparameter tuning was performed. They are formatted as per the description.
4. An answer to the question in the description is present in the column `Comments`.

### Task 02

**Description:**

Analyze the below question. Create a text file with the answer to this question (the letter of the correct line) and an explanation **in Bulgarian** about what metric(s) can be used for the other situations.

Where would *precision* be best suited as the primary metric?

- A. A model predicting the presence of cancer as the positive class.
- B. A classifier predicting the positive class of a computer program containing malware.
- C. A model predicting if a customer is a high-value lead for a sales team with limited capacity.

**Acceptance criteria:**

1. A text file (`task02.txt`) is created.
2. The letter of the correct line is present there.
3. Explanations for the other two cases are given in which the correct metric(s) to use for them is (are) written.

### Task 03

**Description:**

A medical company wants to help people who are is risk of developing diabetes. They contact your team as they need support in identifying patients who are likely to develop it so that they can reach out to them before it sets in. They provide you a dataset - `diabetes_clean.csv` (present in the `DATA` folder in the GitHub repository). A target value of `0` indicates that the individual is not likely to develop diabetes, while a value of `1` indicates that there's a high chance to get the disease.

Perform exploratory data analysis.

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced, similar to the one referenced in the test case.

### Task 04

**Description:**

Create a classification model that predicts the risk of developing diabetes. Create the `model_report` file as described in `notes.md`. Create all models via the class `GridSearchCV`. Use the metrics you get to decide on smaller or larger spaces.

Analyze at least two metrics and add columns showing them **per class** as well as **globally** (and their percentage increase). For each model, paste the confusion matrix in a cell next to your `Comments`. Interpret what you see in the column `Comments`.

For the best model, attach the ROC curve and the full classification report.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. The metrics are analyzed globally and per class (and their percentage increase).

### Task 05

**Description:**

`GridSearchCV` can be computationally expensive, especially if you are searching over a large hyperparameter space. Let's see whether we can obtain a better model if we increase our search space drastically!

Use [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), which tests a fixed number of hyperparameter settings from specified distributions. Increase all of your parameter spaces and attempt to find a better performing model than the one in the previous task.

Analyze at least two metrics and add columns analyzing them **per class** as well as **globally** (and their percentage increase). For each model, paste the confusion matrix in a cell next to your `Comments`. Interpret what you see in the column `Comments`.

> **Note:** The baseline model is now the best model from `Task 04`.

For the best model, attach the ROC curve and the full classification report.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. The metrics are analyzed globally and per class (and their percentage increase).

### Task 06

**Description:**

Create a regression model that predicts blood glucose levels on the diabetes dataset (column `glucose`). For each of your experiments, add one column before your column `Comments`. In it paste a horizontal bar chart sorted by the `x`-axis showing the mean train and test scores. The `y`-axis should have as labels the parameters that were used for the model.

Provide a table analyzing the importance of each feature.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two diagrams that show the model visually are attached in the model report.
3. The values of at least one metric and one loss functions are reported.
4. A table analyzing the importance of each feature is present.
5. Horizontal bar charts are present for each experiment in which hyperparameter tuning was performed. They are formatted as per the description.

## Engineering

Building systems and implementing models.

### Task 01

**Description:**

Let's implement some classification metrics. Add the following functions to the module `metrics`: `accuracy_score`, `recall_score`, `precision_score`, and `f1_score`.

Each of them should accept:

- `y_true`: ground truth values as a Python list;
- `y_pred`: predicted values as a Python list.

And return the corresponding metrics.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `accuracy_score` is added to `ml_lib.metrics`.
2. A function `recall_score` is added to `ml_lib.metrics`.
3. A function `precision_score` is added to `ml_lib.metrics`.
4. A function `f1_score` is added to `ml_lib.metrics`.

### Task 02

**Description:**

In order to implement our logistic regression in a way that allows it to work with multiple classes we'll need to define the logic for the functions `sigmoid` and `softmax`. Add them to the module `stats`. Each of them should be able to work with matrices. `sigmoid` will transform each value independent into a probability. `softmax` will transform values into probabilities so that **each row** in a matrix **sums to $1$**.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `sigmoid` is added to `ml_lib.stats`.
2. A function `softmax` is added to `ml_lib.stats`.

### Task 03

**Description:**

The loss function for logistic regression is the binary cross-entropy function. Let's implement an equivalent that is generalized to work with multiple classes. Add a function `log_loss` to the module `metrics`. It should return the cross-entropy between ground truth labels and predicted ones.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `log_loss` is added to `ml_lib.metrics`.
2. The function `log_loss` is generalized to calculate the cross-entropy when the number of unique labels is greater than $2$.

### Task 04

**Description:**

We should be ready now to define our logistic regression model. Add a class `LogisticRegression` to the module `linear_model`. It should support the following functionality:

- coefficients and intercept of the fitted model are available as object fields.
- a method `fit` that supports:
  - L2 regularization (uses it by default);
  - parameter `C` controlling the regularization strength (smaller values should mean stronger regularization);
  - multiclass classification using the one-vs-rest strategy;
  - gradient descent with parameters `max_iter` and `lr` controlling the step size and number of steps;
  - a parameter `random_state` that is used to initialize the model parameters using a Normal distribution. Play around with the values for the mean and deviation to choose the optimal ones.
- a method `predict_proba` that returns probabilities of a sample belonging to each class of the fitted data. The returned values should add up to `1` along the `x`-axis.
  - when the model has not been fitted and this method is called, raise an appropriate error.
- a method `predict` that returns class index predictions;
- a method `score` that returns the accuracy of the fitted model.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A class `LogisticRegression` is added to the module `linear_model` with the API specification described above.

### Task 05

**Description:**

Let's implement Ridge regression - it modifies the linear regression we implemented earlier by adding `L2` regularization. It should follow the API of the linear regression we implemented earlier and accept and utilize a parameter `alpha` controlling the regularization strength (higher values should indicate stronger regularization).

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A class `Ridge` is added to the module `linear_model` with the API specification described above.

### Task 06

**Description:**

Let's implement Lasso regression - it modifies the linear regression we implemented earlier by adding `L1` regularization. It should follow the API of the linear regression we implemented earlier and accept and utilize a parameter `alpha` controlling the regularization strength (higher values should indicate stronger regularization).

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A class `Lasso` is added to the module `linear_model` with the API specification described above.

### Task 07

**Description:**

Use the classes you just implemented to recreate the best models in tasks 5 and 6 from the data science section.

**Acceptance criteria:**

1. The best models from tasks 5 and 6 are created, trained and evaluated using `ml_lib`.

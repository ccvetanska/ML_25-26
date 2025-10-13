# Goals for week 02

1. Introduce ourselves to the world of machine learning.
2. Solve classification problems.
3. Implement one of the most famous classification algorithms.

## Guidelines

1. Split in teams of at least $2$ people.
2. Go over `notes.md` for the current week.
3. Start solving the tasks on $1$ computer.
4. I'll be stepping in sporadically to track the progress, ensure you're not stuck and help out when needed.
5. Iterate. Whichever team is done with all tasks is free to leave the class.

## Table of Contents

- [Goals for week 02](#goals-for-week-02)
  - [Guidelines](#guidelines)
  - [Table of Contents](#table-of-contents)
  - [Data Science](#data-science)
    - [Task 01](#task-01)
    - [Task 02](#task-02)
    - [Task 03](#task-03)
    - [Task 04](#task-04)
  - [Engineering](#engineering)
    - [Task 01](#task-01-1)
    - [Task 02](#task-02-1)
    - [Task 03](#task-03-1)
    - [Task 04](#task-04-1)
    - [Task 05](#task-05)
    - [Task 06](#task-06)

## Data Science

Learning how to model data effectively.

### Task 01

**Description:**

The `.head()` of a dataset, `churn_df`, is shown below. You can expect the rest of the data to contain similar values.

|   | account_length | total_day_charge | total_eve_charge | total_night_charge | total_intl_charge | customer_service_calls | churn |
|---|----------------|------------------|------------------|--------------------|-------------------|------------------------|-------|
| 0 | 101            | 45.85            | 17.65            | 9.64               | 1.22              | 3                      | 1     |
| 1 | 73             | 22.3             | 9.05             | 9.98               | 2.75              | 2                      | 0     |
| 2 | 86             | 24.62            | 17.53            | 11.49              | 3.13              | 4                      | 0     |
| 3 | 59             | 34.73            | 21.02            | 9.66               | 3.24              | 1                      | 0     |
| 4 | 129            | 27.42            | 18.75            | 10.11              | 2.59              | 1                      | 0     |

Answer the following questions:

1. What is classification?
2. What is binary classification?
3. Which column could be the target variable for binary classification?
4. What term is used to describe the other columns?

**Acceptance criteria:**

1. A Python module (i.e. a `.py` file) is created with the questions put there in a multiline comment.
2. The answers are written **in Bulgarian**.

### Task 02

**Description:**

A telecom company contacts your business. They provide a churn dataset - `telecom_churn_clean.csv` (present in the `DATA` folder in the GitHub repository) alongside a `data dictionary` - a document with column semantics (column definitions):

| Column Name                | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| account_length             | The number of months the customer has been with the telecom company.        |
| area_code                  | The area code of the customer's phone number, indicating their geographic region. |
| international_plan         | Binary indicator (0 = No, 1 = Yes) of whether the customer has an international calling plan. |
| voice_mail_plan            | Binary indicator (0 = No, 1 = Yes) of whether the customer has a voicemail plan. |
| number_vmail_messages      | The number of voicemail messages the customer has.                          |
| total_day_minutes          | Total minutes of calls made during the day (typically daytime hours).       |
| total_day_calls            | Total number of calls made during the day.                                 |
| total_day_charge           | Total charges incurred for daytime calls, in dollars.                       |
| total_eve_minutes          | Total minutes of calls made during the evening (typically evening hours).   |
| total_eve_calls            | Total number of calls made during the evening.                             |
| total_eve_charge           | Total charges incurred for evening calls, in dollars.                       |
| total_night_minutes        | Total minutes of calls made during the night (typically nighttime hours).   |
| total_night_calls          | Total number of calls made during the night.                               |
| total_night_charge         | Total charges incurred for nighttime calls, in dollars.                     |
| total_intl_minutes         | Total minutes of international calls made by the customer.                  |
| total_intl_calls           | Total number of international calls made by the customer.                   |
| total_intl_charge          | Total charges incurred for international calls, in dollars.                 |
| customer_service_calls     | The number of calls the customer made to customer service.                  |
| churn                      | Binary indicator (0 = No, 1 = Yes) of whether the customer discontinued service (churned). |

The telecom wants to be able to predict whether a customer will leave the company (i.e. 'churn'). Your business' internal consultants (i.e. domain experts) tell you that account length should indicate customer loyalty while frequent customer service calls may signal dissatisfaction. Both of these can be good predictors of churning so you decide to use only them to train your model.

Build your first classification model - `KNeighborsClassifier`, and use it to make predictions. Set the number of neighbors to `6`.

**Acceptance criteria:**

1. A Python module is created.
2. A model of type `KNeighborsClassifier` is trained.
3. The test case passes.

**Test case:**

Prediction on `X_new`, as it's defined below, produces `[0 1 0]` (a warning about feature names might come up - ignore it for now).

```python
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
```

### Task 03

**Description:**

Ok, perfect! Let's now move on to **exploratory data analysis**. Remember: **if your data is garbage, then your model will be garbage as well**.

The goal of this step is to get a sense of the data - visualize it and eye-ball any patterns. The desired outcomes are:

1. If you notice any features that combined (ex. summed or multiplied together) can produce an even better / more informative feature, create that feature. This is called **feature engineering**.
2. If you notice features that are irrelevant for the task given (`churn` prediction), discard them at this step and don't bother modelling them.
3. If you notice any features that can be transformed (ex. via exponentiation, logarithmization, standardization, normalization, encoding), do it at this step.

We'll do this in four stages:

1. Univariate analysis.
2. Dealing with missing values and outliers - removing / converting to numbers / capping, etc.
3. Encoding non-numerical features into numerical ones.
4. Multivariate analysis.

Univariate analysis involves looking at all the values each feature has and producing descriptive statistics via [the `pandas` method `describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#pandas-dataframe-describe).

- For the numerical features, this would mean creating descriptive statistics (`mean`, `mode`, `min`, `max`, `number and percentage of missing values`, etc) and creating a histogram. Usually, when the number of unique values is low, i.e. below 10, we can create a bar chart instead.
- For non-numerical features, this would mean calculating the `mode`, `number and percentage of unique values`, `number and percentage of missing values`, etc, and creating a bar chart.

We'll talk about steps `2` (dealing with missing values and outliers) and `3` (encoding) in future lessons.

Multivariate analysis is the creating of a chart via which we check the correlation between the features and the target variable. The idea would be to pick only the features that are correlated with our target.

- For regression tasks, this means creating a heatmap and picking the features that have strong negative or positive correlation with the target.
- For classification tasks, this means creating a scatter plot, the points of which are colored with the class they belong to. If there are clear separation lines, these variables are good candidates for the model.

This is also a good point to do correlation analysis between the features themselves. If two features are correlated, then one of them is redundant and should not be used (it does not add new information).

- For regression tasks, use the heatmap - do not pick correlated features above a threshold, ex `0.8`.
- For classification tasks, use the scatter plot - do not pick features that form a distinct shape.

Let's do all of this by creating an Excel file - `data_audit`. In it place, the outputs of `df.describe().T`. This will give you most of the information you need. Add another right-most column `Comments` in which you explain what you interpret in the current row and what transformations need to be applied to it (if any). This will result in a sheet with two tables - one for the numeric features and another for the object (or string) features (with the current dataset only the first table makes sense).

Then, create multiple sheets for each feature - in each of the sheets, paste a series of the sorted unique values and their count of each feature alongside either a histogram or a bar chart. Use these sheets to expand the contents in the columns `Comments` in the first sheet. This is the univariate analysis.

The multivariate analysis should produce one table - paste it in the main sheet and next to it merge the cells and write your `Comments`.

You can create these sheets manually or automate the process using the package [`openpyxl`](https://cheatography.com/dima/cheat-sheets/openpyxl-cheatsheet/).

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced, similar to the one referenced in the test case.
2. Any sections, marked with `TODO` in the sample solution data audit, are filled-in.

**Test case:**

Download the sample solution data audit (`data_audit.xlsx`, part of this week's folder) to get a sense of what you have to produce.

### Task 04

**Description:**

Awesome! Let's now move on to data modelling!

Search the parameter space of the algorithm `KNeighborsClassifier` to see what accuracies you obtain by tweaking its hyperpameters (it's not just `n_neighbors`). Recall, this is known as **`hyperparameter optimization/tuning`**. Create the `model_report` file as described in `notes.md` and a diagram showing the relationship between the number of neighbors and the accuracy of the best model (keeping the other hyperparameters fixed/constant).

Note that this is the point at which you can also start experimenting with the data values - if you think you can engineer new ones (composing several existing or transforming the existing ones by scaling, normalizing, etc) that would help the model, do it here and report the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created (try to follow **all** guidelines).
2. The best `KNN` model for the task is identified.
3. A diagram showing the relationship between the number of neighbors and the accuracy of the resulting model is present in the `model_report` file.
4. A better model than the baseline model is created.

## Engineering

Building systems and implementing models.

Today we'll start developing our oun machine learning framework - `ml_lib`! The first algorithm in it will be the K-Nearest Neighbors algorithm. Here's how we'll organize it:

1. Assuming you've cloned the GitHub repo, create a folder, named `ml_lib` in the root directory:

Currently, you should have this file structure:

```console
assets\
DATA\
Week_00 - Hello, Python\
...
Week_01 - Numpy, Pandas, Matplotlib, Seaborn\
...
```

After creating the folder, you'll have this:

```console
assets\
DATA\
Week_00 - Hello, Python\
...
Week_01 - Numpy, Pandas, Matplotlib, Seaborn\
...
ml_lib\           # <---------- Notice this - you should create this folder
```

Inside `ml_lib` we'll start creating modules (`.py` scripts) and putting those modules in packages (collections of modules). **Always place an empty file `__init__.py` in every package** - this will tell the Python interpreter that this is a package, not just a directory! This means that in `ml_lib` you'll also place an empty file `__init__.py`.

2. Place an empty file `__init__.py` inside `ml_lib`.
3. Each of the following tasks will tell you whether you should put your code in `ml_lib` or in a separate folder `engineering`, just like you did in week 1. When you're done with the tasks, submit the `ml_lib` as well.

### Task 01

**Description:**

Let's implement train-test splitting. Create a module `model_selection` in the package `ml_lib` and define a function `train_test_split` inside it. It should support the following parameters:

- `X`: a pandas dataframe with features;
- `y`: a pandas series with labels;
- `test_size`: (optional) the proportion of the dataset to include in the test split. Default value: `0.25`;
- `train_size`: (optional) the proportion of the dataset to include in the train split. Default value: the complement of the `test_size`;
- `shuffle`: (optional) whether to shuffle the data. Default value: `True`;
- `random_state`: (optional) a seed for the shuffling. Default value: `None`, meaning that results will not be reproducible;
- `stratify`: (optional) a pandas series with labels. Default value: `None`, meaning that the split will not be stratified.

Adding unit tests is encouraged, though not strictly necessary.

**Acceptance criteria:**

1. A function `train_test_split` is added to `ml_lib`.

### Task 02

**Description:**

Let's implement the metric accuracy. Create a module `metrics` in the package `ml_lib` and define a function `accuracy_score` inside it. It should support the following parameters:

- `y_true`: the ground truth labels;
- `y_pred`: the predicted labels;
- `normalize`: (optional) whether to return the fraction of correctly classified samples. If `False`, returns the number instead of the fraction. Default value: `True`.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `accuracy_score` is added to `ml_lib`.

### Task 03

**Description:**

Add another function to the module `metrics` - `euclidean_distance`. It should accept two `n`-dimensional points and return the Euclidean distance between them.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `euclidean_distance` is added to `ml_lib`.

### Task 04

**Description:**

Add another function to the module `metrics` - `manhattan_distance`. It should accept two `n`-dimensional points and return the Manhattan distance between them.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `manhattan_distance` is added to `ml_lib`.

### Task 05

**Description:**

Create a module `neighbors` in the package `ml_lib` and define a class `KNeighborsClassifier` inside it. It should have:

- a constructor, supporting the parameters `n_neighbors` and `metric` (either `euclidean` or `manhattan`);
- a method `fit`;
- a method `predict`;
- a method `score` that returns the accuracy of the model on the passed data.

Adding unit tests is encouraged, though not strictly necessary.

**Acceptance criteria:**

1. A class `KNeighborsClassifier` is implemented.

### Task 06

**Description:**

Let's compare your library to `sklearn`. Use `ml_lib` to resolve `Task 04` in the previous section (via a script or notebook `task_06.py` in the folder `engineering`). Skip the hyperparameter tuning stage, but do include the creation of the graph between number of neighbors and accuracy score.

**Acceptance criteria:**

1. `ml_lib` is used to resolve `Task 04`.
2. A diagram showing the relationship between the number of neighbors and the accuracy of the resulting model is produced.

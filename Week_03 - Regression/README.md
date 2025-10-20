# Goals for week 03

1. Practice simple linear regression.
2. Practice multiple linear regression.
3. Implement linear regression using the ordinary least squares strategy.

## Data Science

Learning how to model data effectively.

### Task 01

**Description:**

A sales company markets its products on different media - TV commercials, radio and Social Media platforms. They contact your team as they need support in finding out which type of media makes them the most sales (and consequently profit). They provide a sales dataset - `advertising_and_sales_clean.csv` (present in the `DATA` folder in the GitHub repository) which contains information on advertising campaign expenditure across different media types, and the number of dollars generated in sales for the respective campaign.

Perform exploratory data analysis.

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced, similar to the one referenced in the test case.

**Test case:**

Download the sample solution data audit (`Week_02 - Hello, Machine Learning\data_audit.xlsx`) to get a sense of what you have to produce.

### Task 02

**Description:**

Research the definition of $R_{adj}^2$:

- What is the problem with always using $R^2$?
- How does using $R_{adj}^2$ help solve this problem?
- How could we calculate $R_{adj}^2$ in Python?

Create a text file and answer the questions above **in Bulgarian**. Add links to the sources you used.

**Acceptance criteria:**

1. A text file is created.
2. Answers to the questions are given **in Bulgarian**.
3. Sources are added in the end of the file.

### Task 03

**Description:**

Create a linear regression model that predicts the values for `sales`. Play around with the size of the train and test sets and the features you use. Create the `model_report` file as described in `notes.md` and at least two diagrams showing the best obtained model. The diagrams can be of any type as long as they provide value and are **combined with a text-based analysis** written by you. Some ideas include: plot the model (if the number of dimensions allows it), plot the residuals of the model and similar.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two diagrams that show the model visually are attached in the model report.
3. The values of at least one metric and one loss functions are reported.

## Engineering

Building systems and implementing models.

### Task 01

**Description:**

Let's implement the metric $R^2$. Add a function called `r2_score` to the module `metrics` in the package `ml_lib`. It should accept:

- `y_true`: ground truth values **as a Python list**;
- `y_pred`: predicted values **as a Python list**.

And return the coefficient of determination.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `r2_score` is added to `ml_lib.metrics`.

### Task 02

**Description:**

Let's now implement a loss function - for linear models there are plenty to choose from, but we'll go with the `root_mean_squared_error`. To be consistent with `sklearn`, we'll add it to the module `metrics`. It should accept:

- `y_true`: ground truth values **as a Python list**;
- `y_pred`: predicted values **as a Python list**.

And return the root mean squared error regression loss.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `root_mean_squared_error` is added to `ml_lib.metrics`.

### Task 03

**Description:**

Awesome! Now we're ready to implement the next machine learning algorithm: linear regression! Create a module called `linear_model` and in it define the class `LinearRegression`.

It should have as minimum functionality a method:

- `fit`, that takes training data and training labels and fits a line through the data. Calculate the coefficients using the formula we showed in class. To invert a matrix, use the function [np.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html).
  - If the matrix is singular (i.e. cannot be inverted, raise a `ValueError` with the text `'Matrix is not invertible. Please remove collinear features.'`).
  - The method should return the fitted object and the fields `coef_` and `intercept_` should be accessible and populated in that returned object with the coefficients and the intercept, respectively.
- `predict`, that takes a matrix with data and returns the model's predictions.
- `score`, that takes a matrix with data alongside the labels for this data and returns the coefficient of determination.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A class `LinearRegression` is added in `ml_lib.linear_model`.

### Task 04

**Description:**

Let's compare your library to `sklearn`. Use `ml_lib` to resolve `Task 03` from the previous section.

**Acceptance criteria:**

1. `ml_lib` is used to resolve `Task 03`.

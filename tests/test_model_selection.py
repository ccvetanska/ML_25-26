import pandas as pd
import numpy as np
from ml_lib.model_selection import train_test_split

def test_basic_split():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_stratified_split():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0]*50 + [1]*50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    assert y_train.value_counts()[0] == 40
    assert y_train.value_counts()[1] == 40
    assert y_test.value_counts()[0] == 10
    assert y_test.value_counts()[1] == 10

def test_shuffle_false():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series(range(10))

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    assert list(X_test['a']) == [0, 1, 2]
    assert list(X_train['a']) == [3, 4, 5, 6, 7, 8, 9]
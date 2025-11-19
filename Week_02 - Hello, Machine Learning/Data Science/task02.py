from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('DATA/telecom_churn_clean.csv')

    model = KNeighborsClassifier(6)
    Y = df[['account_length', 'customer_service_calls']]
    x = df['churn']
    model.fit(Y, x)

    X_new = np.array([[30.0, 17.5], [107.0, 24.1], [213.0, 10.9]])

    predictions = model.predict(X_new)

    print(predictions)


if __name__ == '__main__':
    main()

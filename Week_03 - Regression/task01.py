import pandas as pd

def main():
    data = pd.read_csv('D:/Fmi/ML_25-26/DATA/advertising_and_sales_clean.csv')
    print(data.head())
    df = pd.DataFrame()
    count = []
    for col in data:
        count.append(data[col].count())
    
    df['COUNT'] = count
    print(df)
    
if(__name__ == '__main__'):
    main()
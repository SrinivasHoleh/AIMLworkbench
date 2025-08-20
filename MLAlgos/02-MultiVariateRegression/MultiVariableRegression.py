import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

def main():
    df = pd.read_csv(os.path.join("../../Data","Car_Price_Prediction.csv"))
    r, c = df.shape
    print(df.head())
    print(df.info())
    print(f"No of rows x Cols = {r} x {c}")
    
    train, test = model_selection.train_test_split(df, test_size=0.2, random_state=2025)

    reg = linear_model.LinearRegression()
    reg.fit(X = train[["Mileage", "Year", "Engine Size", ]], y = train["Price"])

    reg.predict(X=test[["Mileage", "Year", "Engine Size"  ]])

    plt.scatter(x=test["Mileage"], y=test["Price"])
    plt.show()


if __name__ == "__main__":
    main()
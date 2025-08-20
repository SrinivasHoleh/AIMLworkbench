import os   
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

def main():
  
    df = pd.read_csv(os.path.join("../../Data", "Car_Price_Prediction.csv"))
    print(df.info())
    print(df.head())
    #siz = df.size - gives total cells
    rows, colms = df.shape

    print(f"Total rows x columns: {rows} x {colms}")

    train, test = model_selection.train_test_split(df, test_size= 0.2, random_state=2025)

    print(train.head())
    print(test.head())
    r,c = train.shape
    print(f"Train data rows = {r}")
    r,c = test.shape
    print(f"Test data rows = {r}")

    #Regress "Hours Studied" over "Performance Index"
    #Scatter plot
    plt.subplot(2,2,1)
    plt.scatter(x=train["Mileage"], y = train["Price"])

    plt.subplot(2,2,2)
    plt.boxplot(x=train["Mileage"])

    plt.subplot(2,2,3)
    plt.boxplot(x=train["Price"])

    #model
    reg_model = linear_model.LinearRegression()
    # reg_model.fit(x=train["Mileage"], y=train["Price"])
    reg_model.fit(X = df[["Mileage"]], y=df["Price"])

    test["Predicted Price"] = reg_model.predict(X=test[["Mileage"]])

    #t = pd.DataFrame(test)
    #t.to_csv("Predicted.csv", index = False)

    plt.subplot(2,2,4)
    plt.scatter(x=test["Mileage"], y=test["Predicted Price"])
    plt.show()

if __name__ == "__main__":
    main()
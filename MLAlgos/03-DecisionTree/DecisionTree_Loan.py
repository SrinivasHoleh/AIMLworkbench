#Working!
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

def main():
    df = pd.read_csv(os.path.join("../../Data","loan.csv"))
    # print(df.head())
    # print(df.info())

    #Drop columns that are not relevant
    df = df.drop(['ApplicationDate', 'EmploymentStatus', 'EducationLevel','MonthlyIncome', 'RiskScore', 'TotalDebtToIncomeRatio', 'BaseInterestRate', 'InterestRate'], axis='columns')
    df = df.drop(['MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries'], axis='columns')
    df = df.drop(['BankruptcyHistory', 'PaymentHistory', 'SavingsAccountBalance', 'CheckingAccountBalance'], axis='columns')
    df = df.drop(['UtilityBillsPaymentHistory', 'JobTenure', 'MonthlyLoanPayment'], axis = 'columns')
    # print(df.info())

    le_MaritalStatus = LabelEncoder()
    le_NumberOfDependents = LabelEncoder()
    le_HomeOwnershipStatus = LabelEncoder()
    le_LoanPurpose = LabelEncoder()


    df['MaritalStatus_n'] = le_MaritalStatus.fit_transform(df['MaritalStatus'])
    df['NumberOfDependents_n'] = le_NumberOfDependents.fit_transform(df['NumberOfDependents'])
    df['HomeOwnershipStatus_n'] = le_HomeOwnershipStatus.fit_transform(df['HomeOwnershipStatus'])
    df['LoanPurpose_n'] = le_LoanPurpose.fit_transform(df['LoanPurpose'])

    df = df.drop(['MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus', 'LoanPurpose'], axis='columns')
    # print(df.info())
    # print(df.shape)
    X = df.drop(['LoanApproved'], axis='columns')
    y = df['LoanApproved']

    X_train, X_Test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=2025)
    print(X_train.shape)
    print(X_Test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(X.info())

    # decision_tree_model = tree.DecisionTreeClassifier()
    # decision_tree_model.fit(X=X_train, y=y_train)
    # print(decision_tree_model.score(X_Test, y_test))

    # plt.figure(figsize=(15, 10)) # Adjust figure size for better readability
    # plot_tree(decision_tree_model,
    #           feature_names=df.columns, # Display actual feature names
    #         #   class_names=df.target_names,   # Display actual class names
    #           filled=True,                     # Fill nodes with colors based on class
    #           rounded=True,                    # Round node corners
    #           fontsize=10)                     # Adjust font size for readability
    # plt.title("Decision Tree Visualization")
    # plt.show()


    decision_tree_model = tree.DecisionTreeClassifier(max_depth=4)
    decision_tree_model.fit(X=X_train, y=y_train)
    print(decision_tree_model.score(X_Test, y_test))

    plt.figure(figsize=(15, 10)) # Adjust figure size for better readability
    plot_tree(decision_tree_model,
              feature_names=df.columns, # Display actual feature names
            #   class_names=df.target_names,   # Display actual class names
              filled=True,                     # Fill nodes with colors based on class
              rounded=True,                    # Round node corners
              fontsize=10)                     # Adjust font size for readability
    plt.title("Decision Tree Visualization")
    plt.show()

    y_predicted = decision_tree_model.predict(X_Test)
    cm = confusion_matrix(y_test, y_predicted)
    classifictionReport =  classification_report(y_test, y_predicted)
    print(classifictionReport)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


if __name__ == "__main__":
    main()  
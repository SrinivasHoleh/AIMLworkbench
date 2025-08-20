#Working!

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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


    Rf = RandomForestClassifier(n_estimators=60, max_depth=4)
    Rf.fit(X = X_train, y=y_train)
    print(Rf.score(X=X_Test, y= y_test))
    
    # fig, axes = plt.subplots(nrows = 1 ,ncols = 10,figsize = (10,2), dpi=900)
    # for index in range(1,10):
    #     tree.plot_tree(Rf.estimators_[index],
    #                feature_names = df.columns, 
    #                filled = True,
    #                ax = axes[index]);
    #     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)

    # plt.title("Decision Tree Visualization")
    # plt.show()

    # y_predicted = Rf.predict(X_Test)
    # cm = confusion_matrix(y_test, y_predicted)
    # classifictionReport =  classification_report(y_test, y_predicted)
    # print(classifictionReport)
    # # plt.figure(figsize=(10,7))
    # sn.heatmap(cm,annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # plt.show()


if __name__ == "__main__":
    main()  
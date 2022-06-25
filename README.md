# Credit_Risk_Analysis


## Project Background

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. <br>
Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. <br>


### Project Purpose

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company,<br>
we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. <Br>Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. <br>
Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. <br>
Finally, we will evaluate the performance of these models and give recommendations on whether they should be used to predict credit risk.


## Objectives
1. Use Resampling Models to Predict Credit Risk
2. Use the SMOTEENN Algorithm to Predict Credit Risk
3. Use Ensemble Classifiers to Predict Credit Risk

## Resources
- Data Sources: LoanStats_2019Q1.csv, credit_risk_resampling.ipynb, 
- Software & Frameworks: Python (3.7).
- Libraries & Packages: Jupyter Notebook, NumPy (1.21.5), Scipy (1.7.3), Scikit-learn (1.0.2), imbalanced-learn library.
- Online Tools: Credit_Risk_Analysis GitHub repository.


## Methods & Code

1. We will use the imbalanced-learn and scikit-learn libraries and evaluate three resampling machine learning models to determine which is better at predicting credit card risk. 
- First, we will use the oversampling RandomOverSampler and SMOTE algorithms,
-- We started with the credit card loans dataset that contained 144 columns and we filtered it down to 86 columns of interest: <br>
![The Filtered Loans Stats Dataset.]('./Images/loansstats_filtered_df.png')
-- We transformed the focused dataset by dropping NaNs, converting interest rate to float type, and filtered for not-yet issued loans. 
-- Additionally we encoded the loan status -which is the target variable, to low risk and high risk applications then converted them to numerical values, so that low risk: 1, and high risk: 0.
-- Finally, we transformed the string values of 9 columns into numerical ones using the get_dummies() method, which increased the total number of columns to 95.
![The String Columns in need of Conversion to Binary Values.]('./Images/loansstats_stringCols.png')
![The Final Loans DataFrame.]('./Images/loansstats_binaryCols_df.png')
![The Final Loans DataFrame Binary Columns.]('./Images/loansstats_binaryCols.png')

-- Next, we preprocessed the final dataset and defined the target variable to be the loan status, while all other variables (95) are features. 
![The Features DataFrame.]('./Images/loansstats_X_df.png')

-- We first resampled the training data with the Naive Random Oversampling model, then we instantiated and trained a logistic regression model on the resampled data to make predictions on the test data.
- Next, we resample the data using the Synthetic Minority Oversampling Technique or SMOTE to balance the loans dataset, and train the logistic regression model on the resampled data.



## Results

1. The loans dataset contain 115,675 rows of data in 144 columns. <br>
Each row in the dataset represents an application for a credit card loan and information about the applicant including: <br>
loan amount, interest rate, home ownership, annual income, demographics, payments, hardship and settlement info among many other details.<br>
![The Loans Stats Original DataFrame.]('./Images/loansstats_original_df.png')

-- The summary statistics on the loans features are as follows: 
![Summary Statistics on Loans Features.]('./Images/loansstats_X_stats.png')

-- Out of 68,817 records in the loans dataset,68,470 applications were low risk, and only 347 were high risk.
-- The Random Oversampling model redistributed the data as follows: {'low_risk': 51366, 'high_risk': 51366}
-- The logistic regression model using the resample data gave such predictions: 
![The Oversampling Predictions DataFrame.]('./Images/oversampling_predictions.png')
-- The accuracy of the logistic regression model used to predict the risk of credit card loan applications based on Random Oversampling technique was only 66%. 
-- The classification report would show us that while precision is very high (100%) for the majority class of low risk applications, 
-- precision in predicting high risk applications is extremely low (1%) which indicates high number of false positives meaning an unreliable positive classification.
-- However, the recall (sensitivity) is 60% and 70% for low-risk and high-risk applications, respectively. 
-- It seems that the logistic regression model using the Random Oversampling technique was better in screening for high risk loan applications.
![Results of Logistic Regression on Randomly Oversampled Credit Card Loan Applications.]('./Images/oversampling_report.png')

- The Synthetic Minority Oversampling Technique or SMOTE balanced the loans dataset in the same way as the previous model
    {'low_risk': 51366, 'high_risk': 51366}.
-- The confusion matrix results of the logistic regression model in this instance were identical the Random Oversampling technique. 
-- The accuracy of the logistic regression model used to predict the risk of credit card loan applications based on Random Oversampling technique was also 66%. 
-- The classification report for SMOTE is as follows:
![Results of Logistic Regression on SMOTE Oversampled Credit Card Loan Applications.]('./Images/SMOTE_report.png')




## Recommendations & Limitations

    
    
    
    
---

# CREDIT-CARD-FRAUD-DETECTION-MODEL
CREDIT CARD FRAUD DETECTION MODEL
Author : Okeke Felix Emeka
Email : okeke243@gmail.com	linkedin : felix-emeka

**INTRODUCTION**
Credit card fraud is a significant challenge in the financial industry, impacting both financial institutions and cardholders. As technology advances, fraudsters find new ways to exploit vulnerabilities, making it crucial for financial institutions to employ sophisticated fraud detection models. The dataset provided is a valuable resource for developing a credit card fraud detection model, incorporating various features related to the credit card usage patterns and payment history of individuals.

**DATASET OVERVIEW
Dataset Information**
The dataset used for this project is UCI_Credit_Card dataset originates from the UCI Machine Learning repository This dataset has been collected from free or free for research sources at the Internet. The collection is composed of just one text file, where each line has the correct class followed by the raw message. This dataset is comma-separated values (CSV) file. 

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

**Content**
There are 25 variables with 30,000 entries:
The dataset consists of several attributes that provide insights into the financial behavior and credit card transactions of individuals. Each row represents a unique entry with the following features:
•**	ID:** Unique identifier for each entry.
•**	LIMIT_BAL**: Credit limit for the individual.
•	**SEX**: Gender of the cardholder (1 for male, 2 for female).
•	**EDUCATION**: Level of education (1 for graduate school, 2 for university, 3 for high school, 4 for others).
•	**MARRIAGE:** Marital status (1 for married, 2 for single, 3 for others).
•	**AGE:** Age of the cardholder.
•	PAY_0 to PAY_6: Payment status for the past six months.
•	**BILL_AMT1 to BILL_AMT6:** Bill amount for the past six months.
•	**PAY_AMT1 to PAY_AMT6**: Payment amount for the past six months.
•	**default.payment.next.month:** Binary label indicating whether the individual will default on payment next month (1 for yes, 0 for no).

**INSPIRATION**
Some ideas for exploration:
1.	How does the probability of default payment vary by categories of different demographic variables?
2.	Which variables are the strongest predictors of default payment?

**OBJECTIVES**
The primary goal of this project is to develop a robust credit card fraud detection model that can accurately identify transactions at risk of default. By leveraging machine learning techniques, the model aims to analyze historical transaction patterns and payment behaviors to predict the likelihood of future defaults. The dataset will serve as the training and testing ground for building and evaluating the model's performance.

**IMPORTANCE OF FRAUD DETECTION**
Credit card fraud detection is vital for financial institutions to mitigate financial losses and protect the interests of both the institution and its customers. Early detection of fraudulent activities not only helps in preventing unauthorized transactions but also contributes to maintaining trust and confidence in the financial system.

**APPROACH**
The approach to building the fraud detection model involves preprocessing the dataset, selecting relevant features, and employing machine learning algorithms for training and evaluation. Techniques such as data normalization, feature engineering, and model optimization will be applied to enhance the model's performance. The success of the model will be assessed based on metrics such as accuracy, precision, recall, confusion matrix, and F1 score.
By developing an effective credit card fraud detection model, we aim to contribute to the ongoing efforts in enhancing financial security and protecting individuals and institutions from the adverse effects of credit card fraud.

**BENCH MARK MODEL**
**Overview**
To establish a baseline for credit card fraud detection, three commonly used machine learning algorithms will be employed: Logistic Regression, Random Forest, and XGBoost. Each model will be implemented and evaluated using the provided dataset.

**Benchmark Model Details**
The benchmark model will be implemented using the following steps:
**1. Data Preprocessing**
**•	Handle missing values: **Check and handle any missing values in the dataset.
**•	Feature scaling: **Normalize numerical features to ensure consistent scales.
**2. Feature Selection**
•	Evaluate the importance of features and select the most relevant ones for training the model.
**3. Model Training**
•	Implement logistic regression using standard libraries (e.g., scikit-learn) for binary classification.
•	Implement Random Forest classification using standard libraries (e.g. scikit-learn) for binary classification.
•	Implement Xgboost Classifier using standard libraries 
•	Implement cross validation for each model.
•	Train the model on the preprocessed dataset.
**4. Model Evaluation**
•	Assess the model's performance using standard evaluation metrics:
•	Accuracy: The proportion of correctly classified instances.
•	Precision: The ability of the model to avoid false positives.
•	Recall: The ability of the model to identify all relevant instances.
•	F1 Score: The harmonic mean of precision and recall.
**5. Results**
•	Summarize the performance metrics and provide insights into the benchmark model's effectiveness.
•	Use these results as a baseline for comparing more sophisticated models.


**CONCLUSION**
The benchmark models (Logistic Regression, Random Forest, and XGBoost) with cross-validation provide a solid foundation for evaluating more advanced credit card fraud detection models. The cross-validation results offer a more robust estimate of the models' performance, considering multiple splits of the dataset. Subsequent models should aim to surpass these benchmarks, demonstrating their effectiveness in improving fraud detection capabilities.


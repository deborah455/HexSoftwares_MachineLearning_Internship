\# Telecom Customer Churn Prediction System



A machine learning system that predicts whether a telecommunications customer is likely to churn based on their account, service, contract, and billing information.



\## Project Overview



Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to leave allows businesses to take proactive retention measures.



This project develops an end-to-end machine learning pipeline for customer churn prediction using the IBM Telco Customer Churn dataset.



The system includes:



\- Data cleaning and preprocessing

\- Exploratory Data Analysis (EDA)

\- Customer churn analysis

\- Feature preprocessing and encoding

\- Machine learning model training

\- Model comparison

\- Churn probability prediction

\- Flask-based prediction interface



> \*\*Business Context:\*\* The project represents a fictional telecommunications customer-retention use case. The underlying dataset contains customer information such as tenure, contract type, internet service, payment method, monthly charges, and churn status.



\---



\## Project Objectives



The main objectives were to:



1\. Understand customer churn patterns.

2\. Identify factors associated with customer churn.

3\. Build classification models to predict churn.

4\. Compare Logistic Regression and Random Forest.

5\. Evaluate models using appropriate classification metrics.

6\. Develop a web application for interactive churn prediction.



\---



\## Dataset



The project uses the \*\*IBM Telco Customer Churn\*\* dataset.



The original dataset contains \*\*7,043 customer records and 21 columns\*\*.



After cleaning and removing records with missing `TotalCharges` values, the final dataset contained:



\- \*\*7,032 customers\*\*

\- \*\*20 usable features including the target\*\*

\- \*\*5,163 customers who did not churn\*\*

\- \*\*1,869 customers who churned\*\*



\### Churn Distribution



| Customer Status | Customers | Percentage |

|---|---:|---:|

| No Churn | 5,163 | 73.42% |

| Churn | 1,869 | 26.58% |



\---



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Flask

\- Joblib

\- HTML

\- CSS



\---



\## Project Structure



```text

Project\_02\_Customer\_Churn\_Prediction/

│

├── data/

│   ├── customer\_churn.csv

│   └── cleaned\_customer\_churn.csv

│

├── models/

│   ├── customer\_churn\_model.joblib

│   └── model\_comparison.csv

│

├── screenshots/

│   ├── churn\_distribution.png

│   ├── churn\_by\_contract.png

│   ├── churn\_by\_internet\_service.png

│   ├── churn\_by\_payment\_method.png

│   └── ...

│

├── src/

│   ├── preprocess.py

│   └── eda.py

│

├── templates/

│   └── index.html

│

├── app.py

├── train.py

├── requirements.txt

└── README.md

Data Preprocessing



The preprocessing pipeline performed the following steps:



Loaded the raw customer dataset.

Converted TotalCharges to numeric format.

Identified missing values.

Removed records with missing TotalCharges.

Converted the Churn target into binary values:

0 = No Churn

1 = Churn

Separated numerical and categorical features.



The final cleaned dataset contained 7,032 records with no remaining missing values.



Exploratory Data Analysis



Several analyses were performed to understand customer churn behavior.



Churn by Contract



The analysis showed a strong relationship between contract type and churn.



Contract	No Churn	Churn

Month-to-month	57.29%	42.71%

One year	88.72%	11.28%

Two year	97.15%	2.85%



Customers on month-to-month contracts showed substantially higher churn rates.



Churn by Internet Service

Internet Service	No Churn	Churn

DSL	81.00%	19.00%

Fiber optic	58.11%	41.89%

No Internet	92.57%	7.43%



Fiber-optic customers showed a considerably higher churn rate in this dataset.



Churn by Payment Method



Electronic check customers showed the highest churn rate:



Payment Method	Churn

Bank transfer	16.73%

Credit card	15.25%

Electronic check	45.29%

Mailed check	19.20%

Machine Learning



Two classification models were trained and compared:



1\. Logistic Regression



Logistic Regression was used as a strong interpretable baseline for binary classification.



2\. Random Forest



Random Forest was used as an ensemble-based alternative capable of modeling more complex relationships between customer attributes.



Preprocessing Pipeline



The machine learning pipeline included:



Median imputation for numerical features

Most-frequent imputation for categorical features

StandardScaler for numerical features

OneHotEncoder for categorical features

Stratified train/test split

Balanced class weights



The dataset was split into:



80% training data

20% testing data

Model Results

Model	Accuracy	Precision	Recall	F1	ROC-AUC

Logistic Regression	72.57%	49.01%	79.68%	60.69%	83.51%

Random Forest	76.05%	54.14%	64.71%	58.95%	81.54%

Selected Model



Logistic Regression was selected based on the highest F1-score.



Although Random Forest achieved higher overall accuracy, Logistic Regression achieved:



Higher recall

Higher F1-score

Higher ROC-AUC



For a customer-retention use case, recall is particularly important because failing to identify a customer who is likely to churn can result in a missed retention opportunity.



Confusion Matrix — Logistic Regression

&#x20;                Predicted

&#x20;                No Churn   Churn





Actual No Churn     723       310

Actual Churn         76       298



The model identified 298 of 374 actual churners, resulting in a churn recall of approximately 79.68%.



Web Application



A Flask web application was developed to provide an interactive interface for the trained model.



Users can enter customer information including:



Gender

Senior citizen status

Partner/dependent status

Tenure

Phone service

Internet service

Online security

Online backup

Device protection

Technical support

Streaming services

Contract type

Billing method

Payment method

Monthly charges

Total charges



The application returns:



Churn prediction

Churn probability

Risk classification

Example Output

Prediction: Likely to Churn





Churn Probability: 78.4%





Risk Level: High Risk

How to Run

1\. Clone the repository

git clone <repository-url>

2\. Navigate to the project

cd Project\_02\_Customer\_Churn\_Prediction

3\. Create a virtual environment

python -m venv .venv

4\. Activate the environment

Windows

.venv\\Scripts\\activate

5\. Install dependencies

pip install -r requirements.txt

6\. Run preprocessing

python src/preprocess.py

7\. Run EDA

python src/eda.py

8\. Train the models

python train.py

9\. Start the Flask application

python app.py



Open:



http://127.0.0.1:5000

Key Insights



The project demonstrated several important customer churn patterns:



Month-to-month customers were substantially more likely to churn.

Two-year contract customers had the lowest churn rate.

Fiber-optic customers had a relatively high churn rate.

Electronic-check customers had the highest churn rate.

Customer tenure had a negative relationship with churn.

Monthly charges showed a positive relationship with churn.



These insights demonstrate how machine learning can support customer-retention strategies by identifying customers who may require proactive engagement.



Skills Demonstrated

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Categorical Encoding

Feature Scaling

Classification

Logistic Regression

Random Forest

Model Evaluation

Precision / Recall Analysis

F1-Score

ROC-AUC

Confusion Matrix Analysis

Scikit-learn Pipelines

Flask

Machine Learning Deployment

Business-oriented ML Analysis

Future Improvements



Possible improvements include:



Hyperparameter tuning

Cross-validation

Threshold optimization

Feature importance and model interpretability

Customer retention recommendations

Prediction history storage

Interactive analytics dashboard

Model monitoring

Deployment to a cloud platform

Integration with a customer-management system

Author



Deborah Kyalo



Machine Learning \& Python Developer



Internship



Developed as part of the Hex Software Machine Learning Internship.


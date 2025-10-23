**Credit Card Fraud Detection Using Machine Learning**

      This project is based on my published research paper: [Credit Card Fraud Detection using Machine Learning](https://doi.org/10.34218/IJMLC_03_01_002), which explores machine learning techniques for identifying fraudulent credit card transactions.
      It compares their performance based on accuracy and can be integrated into real-time applications like banking systems and e-commerce platforms.

**Overview:**

  Credit card fraud has become a serious issue due to the rapid growth of e-commerce and online transactions.
   This system applies machine learning models to classify transactions as fraudulent (1) or legitimate (0) using features from anonymized transaction data.
  The goal is to achieve high detection accuracy while minimizing false positives.

**Dataset:**

Source: Kaggle Credit Card Fraud Detection Dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Total transactions: 284,807

Fraudulent transactions: 492

Non-fraudulent transactions: 284,315

Features: 31 columns (28 PCA-transformed “V” features + Time, Amount, Class)

There are no missing values in the dataset.

**Features:**

**Time:** Time between the first and current transaction.

**Amount:** Transaction amount.

**Class:** Output variable (0 = Normal, 1 = Fraud).

**V1–V28:** Anonymized principal components from PCA transformation.


**Methodology:**

1.Data collection and preprocessing (CSV format from Kaggle).

2.Splitting data into training and testing sets (ratios: 6:4, 7:3, and 8:2).

3.Building classification models using ML algorithms.

4.Evaluating models by accuracy, based on true/false positives and negatives.

5.Comparing models via a bar graph.

**Algorithms Implemented**

**Algorithm** ->**Description** -> **Accuracy (Approx.)**

Random Forest -> Ensemble of decision trees reducing overfitting issues	-> 99.82%

Logistic Regression -> Predicts fraud probability using sigmoid-based binary classification	-> 97.78%

Decision Tree ->	Graph-based classification using entropy and information gain -> 	97.14%

Naive Bayes -> Probabilistic classifier using Bayes’ theorem -> 99.32%

Passive Aggressive ->	Online-learning algorithm for large-scale and imbalanced datasets	 -> 99.82% (Best)

**Results:**
   The **Passive Aggressive Algorithm** demonstrated the highest accuracy across multiple data split ratios (64%, 73%, 82%).

   Closely followed by **Random Forest** and **Naive Bayes**.

   The implementation demonstrates strong potential for real-world financial fraud detection.

   Bar graphs comparing model accuracy are displayed in the GUI after execution.

**Installation:**

Clone this repository and install dependencies.

git clone https://github.com/Ninisha-Balay/Credit-card-fraud-detection/blob/main/CreditCardFraud.py

cd CreditCardFraudDetection

pip install -r requirements.txt

Ensure Python 3.8+ and packages such as numpy, pandas, tkinter, matplotlib, and scikit-learn are installed.

**How to Run:**
1. Run the main Python file:

   python CreditCardFraud.py

2. Use the graphical interface to:

   . Upload Kaggle dataset (creditcard.csv)

   . Split data into selected ratios

   . Execute algorithm buttons (Random Forest, Logistic Regression, etc.)

   . View results and accuracy comparisons.

**Future Scope:**

   . Integrate deep learning architectures such as LSTM or autoencoders.

   . Real-time fraud detection using streaming pipelines (Kafka/Spark).

   . Model deployment as a microservice via Flask or FastAPI.

   . Feature optimization and combination of ensemble classifiers for improved detection.

**References:**
Key references in paper , and Kaggle datasets, as detailed in the thesis (Report.pdf).

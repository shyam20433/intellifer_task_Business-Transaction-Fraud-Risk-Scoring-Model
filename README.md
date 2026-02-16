# Business Transaction Fraud Risk Scoring Model

## 1. Background and Problem

So this is a task where the intellifer systems assigned me a task  
**Develop a classification model that assigns a risk score (0–1) to business transactions.**

Before directly jumping into the implementation, I explored the company's domain. Intellifer works in areas like:

- **AI / Machine Learning**
- **Business Analytics**
- **IoT**
- **Enterprise Solutions**

Since the task was related to risk scoring and classification, I chose to approach it from a Data Analytics / Machine Learning perspective.

Initially, I was unsure about which real-world business problem to choose. I discussed this with my course faculty and explained that I had been shortlisted for the company and assigned this task. She suggested that I choose a business domain, identify the risks involved in that business, list the possible risk factors, and then determine the features required to model that risk.

After researching real-world business problems, I decided to work on:

### **Financial Transaction Risk Classification System**

This project focuses on identifying whether a financial transaction is fraudulent or legitimate and assigning a risk score between 0 and 1.
## 2. Dataset Selection

To implement this idea, I downloaded a real-world dataset from Kaggle:

### **PaySim – Synthetic Financial Dataset for Fraud Detection**

This dataset simulates mobile financial transactions and includes information such as:

- **Transaction type**
- **Transaction amount**
- **Sender and receiver balances**
- **Fraud label** (isFraud)

I completed the entire project without using any AI code-generating tools.
## 3. Data Cleaning and Preprocessing

### 3.1 Loading and Inspecting the Dataset

I loaded the dataset and performed the following initial checks:

- Converted all column names to snake_case (lowercase and standardized naming)
- Checked dataset shape
- Verified data types
- Checked for missing values
- Checked for duplicate rows
- Examined class distribution

The dataset did not contain missing values or duplicate rows, so I proceeded to feature engineering.

I also performed summary statistics using:

```python
df.describe(include="all")
```

This helped me understand the distribution of numerical features such as amount and balances.

## 4. Feature Engineering

Since fraud detection depends heavily on behavioral patterns, I created additional meaningful features.

### 4.1 Balance Difference Features

```python
original_balance_difference = oldbalanceorg - newbalanceorig
dest_balance_diff = newbalancedest - oldbalancedest
```

These features capture:

- How much money was deducted from the sender
- How much money was added to the receiver

This helps detect abnormal money flow patterns.

### 4.2 Ratio-Based Features

```python
amount_to_origin_ratio = amount / (oldbalanceorg + 1)
amount_to_dest_ratio = amount / (oldbalancedest + 1)
```

These features measure:

- How large the transaction is relative to the sender's balance
- How large the transaction is relative to the receiver's balance

High ratios may indicate suspicious behavior such as draining the full account balance.

### 4.3 Time-Based Feature Engineering

The dataset contains a step column representing time in hours.

I converted it into:

```python
transaction_hour = step % 24
```

Then I created a behavioral feature:

```python
night_transaction = 1 if transaction_hour < 6 else 0
```

This helps detect transactions occurring during late-night hours.

After extracting useful time features, I dropped the step column.

### 4.4 Dropping Unnecessary Columns

I removed:

- `nameorig`
- `namedest`
- `isflaggedfraud`

These columns were either identifiers or could cause data leakage.

### 4.5 Encoding Categorical Feature

The type column was converted using one-hot encoding:

```python
pd.get_dummies(df, columns=["type"], drop_first=True)
```

This ensures the model does not assume any artificial ranking between transaction types.

Finally, I saved the cleaned dataset as:

**cleaned_data.csv**
## 5. Model Building

### 5.1 Splitting Data

I separated features and target:

```python
x = df.drop(columns=["isfraud"])
y = df["isfraud"]
```

Then performed stratified train-test split:

```python
train_test_split(x, y, test_size=0.2, random_state=45, stratify=y)
```

Stratification ensures fraud distribution remains consistent in both sets.
## 6. Model 1 – Logistic Regression

I trained Logistic Regression with class imbalance handling:

```python
LogisticRegression(class_weight="balanced", max_iter=1000)
```

After training, I:

- Predicted labels
- Generated probability-based risk scores using `predict_proba()`
- Printed confusion matrix
- Printed classification report

I also tested with a custom transaction input and generated:

- **Prediction** (0 = Legit, 1 = Fraud)
- **Risk score** (0–1)
- **Risk level** (Low / Medium / High)
## 7. Model 2 – Decision Tree Classifier

I trained a Decision Tree model:

```python
DecisionTreeClassifier(class_weight="balanced", random_state=42)
```

Then I:

- Evaluated using confusion matrix
- Generated classification report
- Tested with new transaction input
- Extracted fraud probability
- Assigned risk category
## 8. Risk Score Generation

The core requirement was:

> **Assign a risk score between 0 and 1.**

This was implemented using:

```python
model.predict_proba(new_transaction)[:,1]
```

This gives the probability that a transaction is fraudulent.

Based on this probability, I categorized risk levels:

- **0.0 – 0.3** → Low Risk
- **0.3 – 0.7** → Medium Risk
- **0.7 – 1.0** → High Risk
## 9. Evaluation Metrics

I evaluated models using:

- **Confusion Matrix**
- **Precision**
- **Recall**
- **F1-score**

Since fraud detection is a high-risk domain, I focused more on:

- **Minimizing False Negatives**
- **Maintaining high Recall**

This ensures fewer fraudulent transactions are missed.
## 10. Conclusion

This project successfully implements a real-world fraud risk scoring system that:

- Cleans and preprocesses financial transaction data
- Engineers meaningful behavioral features
- Trains classification models
- Assigns probability-based risk scores (0–1)
- Categorizes transactions into risk levels

This system can be deployed in:

- **Banking systems**
- **Payment gateways**
- **E-commerce platforms**
- **Financial institutions**

It demonstrates how machine learning can support automated risk management and intelligent business decision-making.
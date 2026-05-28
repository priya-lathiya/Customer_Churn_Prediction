# 🏦 Customer Churn Prediction using Artificial Neural Network (ANN)

A deep learning project that predicts whether a bank customer will churn (leave the bank) based on their demographic and account data, using a TensorFlow/Keras neural network with ROC-AUC optimized threshold selection.

---

## 📌 Problem Statement

Customer churn is a critical business problem in the banking industry. Losing a customer is far more expensive than retaining one. This project builds a binary classification model to predict whether a customer will exit (churn = 1) or stay (churn = 0), enabling the bank to take proactive retention actions.

---

## 📂 Dataset

- **Source**: [Kaggle – Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)
- **File**: `Churn_Modelling.csv`
- **Size**: 10,000 rows × 14 columns

### Features Used

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card (1 = Yes) |
| `IsActiveMember` | Is an active member (1 = Yes) |
| `EstimatedSalary` | Estimated annual salary |
| **`Exited`** | **Target — 1 = Churned, 0 = Stayed** |

> Columns `RowNumber`, `CustomerId`, and `Surname` were dropped as they carry no predictive value.

---

## 🔧 Project Workflow
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Threshold Optimization

### 1. Exploratory Data Analysis (EDA)
- Checked dataset shape, dtypes, and null/duplicate values
- Analyzed class distribution of `Exited` (target imbalance)
- Explored value counts for `Geography` and `Gender`

### 2. Data Preprocessing
- Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Applied **One-Hot Encoding** on `Geography` and `Gender` with `drop_first=True`
- Converted boolean dummy columns to integers
- Split data: **80% train / 20% test**
- Applied **StandardScaler** on features (fit on train, transform on test)

### 3. Model Architecture
Input Layer  →  Dense(3, relu)
↓
Hidden Layer →  Dense(3, relu)
↓
Output Layer →  Dense(1, sigmoid)

- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Epochs**: 50  
- **Validation Split**: 20% of training data  

### 4. Evaluation & Threshold Optimization
- Predicted churn probabilities on the test set
- Plotted **ROC Curve** and computed **AUC Score**
- Used **Youden's J Statistic** to select the optimal classification threshold:
  > `Best Threshold = argmax(TPR - FPR)`
- Generated **Classification Report** using the optimal threshold
- Plotted **Training vs Validation Loss** and **Accuracy Curves**

---

## 📊 Results

| Metric | Value |
|---|---|
| AUC Score | 0.7994 |
| Best Threshold | 0.2660 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Preprocessing, metrics, train-test split |
| TensorFlow / Keras | Neural network model |
| Matplotlib | Visualization |

---

## 🚀 How to Run

### Option 1 – Run on Kaggle (Recommended)
1. Open the notebook directly on [Kaggle](https://www.kaggle.com/)
2. Attach the dataset: `rjmanoj/credit-card-customer-churn-prediction`
3. Click **Run All**

---

## 🔧 Project Workflow

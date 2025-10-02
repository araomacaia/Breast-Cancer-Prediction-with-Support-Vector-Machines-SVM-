# Breast Cancer Prediction with Support Vector Machines (SVM)

---
## Introduction

In this project, I implemented Support Vector Machines (SVM) to predict whether a tumor is malignant (1) or benign (0) using the Breast Cancer dataset.
The main objective was to learn and apply SVMs with different kernels, evaluate their performance, and visualize decision boundaries.

---
## Workflow

  **1. Loading the Dataset**
```
df = pd.read_csv("breast-cancer.csv")
print("Shape:", df.shape)
df.head()
```

  **2. Preprocessing**

  - Dropped unnecessary columns.

  - Converted the target variable diagnosis into binary values (M=1, B=0).

  - Standardized features using StandardScaler.

  **3. Train-Test Split**

  - Splited data into training (80%) and testing (20%) sets with stratification.

  - Model Training

  - Trained Linear SVM and RBF SVM classifiers.

  - Performed GridSearchCV for hyperparameter tuning (C, gamma).

 **4. Evaluation Metrics**

  - Accuracy

  - Confusion Matrix

  - Precision, Recall, and F1-score

  **5. Visualization**

  - Plotted the decision boundary (using two features for simplicity).

---
## Results

## Linear SVM

  - Accuracy: 96.49%

***Confusion Matrix:***

|          | Predicted 0 | Predicted 1 |
| :------- | :---------: | ----------: |
| Actual 0 |   72 (TN)   |   0 (FP)    |
| Actual 1 |   4 (FN)    |   38 (TP)   |

  - **Precision (Class 0):** 0.95 | **Recall:** 1.00 | **F1:** 0.97

  - **Precision (Class 1):** 1.00 | **Recall:** 0.90 | **F1:** 0.95

----
## RBF SVM

  - Accuracy: 96.49%

***Confusion Matrix:***

|          | Predicted 0 | Predicted 1 |
| :------- | :---------: | ----------: |
| Actual 0 |   71 (TN)   |   1 (FP)    |
| Actual 1 |   3 (FN)    |   39 (TP)   |

  - **Precision (Class 0):** 0.96 | **Recall:** 0.99 | **F1:** 0.97

  - **Precision (Class 1):** 0.97 | **Recall:** 0.93 | **F1:** 0.95

----
## Hyperparameter Tuning (GridSearchCV, RBF Kernel)

- **Best Parameters:** C=1, gamma=scale, kernel=rbf

- **Best Cross-Validation Score:** 97.36%

----
## Tech Stack

  - Python

  - Pandas & NumPy → **Data manipulation**

  - Scikit-learn → **SVM, scaling, GridSearchCV, evaluation**

  - Matplotlib & Seaborn → **Visualization**

  - MLxtend → **Decision boundary plotting**

----
## Acknowledgments

These resources helped me understand both the theoretical foundations of Support Vector Machines and their practical implementation in Python.

  - Support Vector Machines – Ingo Steinwart, Andrea Christmann

  - An Idiot’s Guide to Support Vector Machines (SVMs) – R. Berwick

- Support Vector Machines – Scikit-learn Documentation

- SGD: Maximum Margin Separating Hyperplane – Scikit-learn Documentation

- SVM: Separating Hyperplane for Unbalanced Classes

- An Introduction to Support Vector Machines – N. Cristianini & J. Shawe-Taylor

- SVM: Maximum Margin Separating Hyperplane – Scikit-learn Documentation

- CS229 Lecture Notes, Part V – Support Vector Machines – Andrew Ng

- Tutorial on Support Vector Machine (SVM) – Vikramaditya Jakkula

- Lecture 2: The SVM Classifier – A. Zisserman

- Plot Classification Probability – Scikit-learn Documentation

- Plot Different SVM Classifiers in the Iris Dataset – Scikit-learn Documentation

---
📝 Author
**Arao Macaia**

AI & ML Internship Task 7 (2025)

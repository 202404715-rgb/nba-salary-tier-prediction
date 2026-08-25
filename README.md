# 🏀 NBA Players Salary Tier Prediction

A machine learning project that predicts whether an NBA player belongs to a **below-median or above-median salary tier** using player and team information.

Instead of predicting the exact salary, the project treats the problem as a **binary classification task**.

---

## 🎯 Project Goal

NBA salaries have a wide range, especially because of high-value superstar contracts.

To make the problem more suitable for the available dataset, the salary was converted into two classes using the **median salary**:

* `0` → Below Median Salary
* `1` → Above Median Salary

The main question is:

> Can player characteristics and team information help predict whether a player's salary is above or below the median?

---

## 📊 Dataset & Preparation

The data was cleaned and prepared before training the model.

Main preprocessing steps included:

* Cleaning and standardizing player height.
* Removing features with no useful information.
* Handling missing numerical values using median imputation.
* Scaling numerical features using `StandardScaler`.
* Keeping categorical features such as `team` and `college` available for CatBoost.

The preprocessing was performed as part of the modeling workflow to reduce the risk of data leakage.

---

## 🤖 Machine Learning Model

The main model used in this project is **CatBoost Classifier**.

CatBoost was selected because the dataset contains categorical variables such as:

* Team
* College

### Model Parameters

```python
model_cb = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=5,
    cat_features=categorical_features,
    random_seed=42
)
```

---

## 🔬 Model Validation

The model was evaluated using **5-Fold Stratified Cross-Validation**.

This approach keeps the class distribution relatively consistent across the different folds and provides a more reliable estimate than relying on a single train/test split.

### Cross-Validation Result

**Mean Accuracy: 64.90% ± 6.65%**

This means the model achieved an average accuracy of approximately **65% across the validation folds**.

---

## 📈 Classification Results

The final classification results were:

| Salary Tier          | Precision | Recall | F1-Score | Support |
| -------------------- | --------: | -----: | -------: | ------: |
| Below Median         |      0.65 |   0.65 |     0.65 |     224 |
| Above Median         |      0.65 |   0.65 |     0.65 |     223 |
| **Overall Accuracy** |           |        | **0.65** | **447** |

### What does this mean?

The model achieved around **65% accuracy**, with similar precision, recall, and F1-score for both salary classes.

Because the two classes are almost evenly distributed, a random 50/50 baseline would be around 50%. The model therefore captures some useful predictive patterns, but the available features are not sufficient for highly accurate salary prediction.

---

## 🔎 Feature Importance

The most important features identified by CatBoost were:

| Feature | Importance |
| ------- | ---------: |
| Age     |     26.50% |
| Team    |     16.97% |
| College |     16.63% |
| Weight  |      7.68% |
| Height  |      4.92% |

### Main Insights

**Age** was the strongest predictor, suggesting that career stage is related to salary tier in this dataset.

**Team** and **College** also contributed significantly to the model's predictions.

Physical features such as **weight and height** had lower importance compared with age, team, and college.

> Feature importance shows how useful a variable was for prediction. It does not mean that the variable directly causes salary changes.

---

## 💡 Key Findings

* Player age was the most important feature.
* Team and college provided meaningful predictive information.
* Physical characteristics alone were not strong predictors of salary tier.
* The current dataset provides some predictive signal, but its performance is limited by the available features.
* Salary is influenced by many factors that are not included in the dataset.

---

## 🚀 Future Improvements

The model could be improved by adding more information about player performance and career history, such as:

* Points Per Game (PPG)
* Assists
* Rebounds
* Minutes Played
* Games Started
* Player Efficiency Rating (PER)
* All-Star selections
* All-NBA selections
* NBA experience
* Contract information
* Previous salary

These features could provide a better representation of player value and potentially improve salary-tier prediction.

---

## ⚠️ Limitations

This project has several limitations:

* The dataset does not contain detailed performance statistics.
* Contract and market information is not available.
* Converting salary into two classes removes the exact salary information.
* The model identifies predictive relationships, not causal relationships.
* The current model accuracy of approximately 65% shows that additional features are needed for stronger predictions.

---

## 🧪 Project Workflow

```text
Raw NBA Data
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Feature Preparation
     ↓
Salary → Salary Tier
     ↓
CatBoost Classifier
     ↓
5-Fold Stratified Cross-Validation
     ↓
Model Evaluation
     ↓
Feature Importance
     ↓
Business & Sports Analytics Insights
```

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn jupyter
```

### Run the Project

1. Clone the repository.
2. Add the dataset to the project folder.
3. Open the Jupyter Notebook.
4. Run the notebook cells sequentially.
5. The notebook will clean the data, train the CatBoost model, evaluate its performance, and generate the analysis.

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 👤 Author

**Habiba Mohamed**

Data Science Student

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow for predicting NBA salary tiers, from data preparation and feature processing to model training, cross-validation, evaluation, and feature interpretation.

The results show that basic player and team information can provide useful signals for salary-tier prediction, while also highlighting the importance of richer performance and contract data for future improvements.

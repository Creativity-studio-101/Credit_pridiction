# Credit Prediction Model

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [License](#license)


## Introduction
This project builds a machine learning model to predict **whether a customer qualifies for credit approval** based on factors like **income, CoapplicantIncome, education, employment, loan history, etc**.


## Dataset
The model is trained on a dataset containing:
- **Customer Information**: Education, Occupation, Income Level, Gender
- **Loan Request Details**: Amount Requested, Duration, Credit history
- **Target Variable**: **Credit Approval** (0 = Denied, 1 = Approved)

## Technologies Used
- Python 
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn

## Usage
Run the model using:
```python
import pandas as pd
from model import CreditPredictionModel  

# Load dataset
data = pd.read_csv("CreditData.csv")

# Train model
model = CreditPredictionModel()
model.train(data)

# Make predictions
prediction = model.predict(new_customer_data)
print(prediction)
```


## Model Performance
   ####### Linear Regression #######
R² Score: 0.2530230516267622
Mean Absolute Error (MAE): 0.30260773891094145
Mean Squared Error (MSE): 0.15517349475083209
Root Mean Squared Error (RMSE): 0.39392067063157793
  ####### Decision Tree #######
Accuracy Score: 0.700507614213198
```
Classification Report:
               precision    recall  f1-score   support

           0       0.49      0.59      0.54        58
           1       0.81      0.75      0.78       139

    accuracy                           0.70       197
   macro avg       0.65      0.67      0.66       197
weighted avg       0.72      0.70      0.71       197

Confusion Matrix:
 [[ 34  24]
 [ 35 104]]
```
  ####### Random Forest #######
Accuracy Score: 0.7868020304568528
```
Classification Report:
               precision    recall  f1-score   support

           0       0.63      0.66      0.64        58
           1       0.85      0.84      0.85       139
    accuracy                           0.79       197
   macro avg       0.74      0.75      0.75       197
weighted avg       0.79      0.79      0.79       197

Confusion Matrix:
 [[ 38  20]
 [ 22 117]]
```

## Results
The model predicts **whether a customer will get credit approval** based on historical data.

## License
This project is licensed under the **MIT License**.

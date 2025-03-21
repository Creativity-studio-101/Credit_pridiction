# Credit Prediction Model

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Contributing](#contributing)
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

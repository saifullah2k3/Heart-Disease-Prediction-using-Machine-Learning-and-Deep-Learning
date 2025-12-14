# Heart Disease Prediction

This project analyzes a dataset of heart disease patient records to predict the presence of heart disease using Machine Learning and Deep Learning techniques.

## Overview

The project explores the dataset, performs data preprocessing and feature engineering, and then trains multiple models to classify patients.

### Key Classifiers
1. **Logistic Regression**: A traditional statistical machine learning model.
2. **Multi-Layer Perceptron (MLP)**: A deep learning model built with TensorFlow/Keras.

## Dataset

The dataset (`heart.csv`) contains the following features:
- `age`: Age in years
- `sex`: Sex (1 = male; 0 = female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholestoral in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by flourosopy
- `thal`: 3 = normal; 6 = fixed defect; 7 = reversable defect
- `target`: Diagnosis of heart disease (1 = presence; 0 = absence)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart_Disease_Prediction.ipynb
   ```
3. Run all cells to see the analysis and model training results.

## Results

The project compares the performance of Logistic Regression and the MLP model using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

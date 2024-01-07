# Heart Disease Prediction Project

## Overview

This project focuses on predicting the presence of heart disease in individuals based on various health-related features. The dataset used for this analysis is provided in a CSV file named "heart.csv."

## Files

- **heart.csv**: The dataset containing health-related information used for training and testing the predictive models.

- **main.py**: The main Python script implementing data analysis, model training, and evaluation.

## Requirements

Make sure you have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow (for Keras Neural Network)
  
You can install the required packages using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```

# Instructions

### Clone the Repository:

```bash
https://github.com/MankaranRooprai/heart-disease-predictor.git
```

### Run the Script:

```bash
python main.py
```

# Results

The script generates accuracy scores for different machine learning models, including:

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Neural Network

The final accuracy scores are visualized using a barplot.

# Important Notes

- The dataset contains information about individuals, including age, sex, chest pain type, and other health-related factors.
- The "target" column in the dataset indicates the presence (1) or absence (0) of heart disease.
- The Neural Network model's accuracy can be improved by adjusting hyperparameters, such as the number of epochs and nodes in the hidden layers.
- Make sure to install the required packages listed in the Requirements section before running the script.

# Acknowledgments

- The dataset used in this project is sourced from [provide the source if applicable].
- Special thanks to the developers of scikit-learn, XGBoost, and TensorFlow for their contributions to machine learning libraries.

Feel free to explore and modify the script for further analysis or improvement of predictive models.
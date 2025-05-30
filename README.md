# Titanic Data Preprocessing Engine

## Overview
A preprocessing module built using pandas and NumPy for preparing the Titanic dataset for ML modeling.

## Features
- Handles missing values and duplicates
- Extracts and consolidates titles
- Bins and scales numeric features
- Encodes categorical variables
- Outputs cleaned data in CSV and NumPy formats

## Directory Structure
titanic_project/
├── data/
│   └── train.csv            
├── notebook/
│   └── titanic_cleaning.ipynb
├── output/
│   ├── cleaned.csv
│   └── final_features.npy
├── preprocess.py
└── README.md


## Output
- `output/cleaned.csv`: Fully cleaned dataset
- `output/final_features.npy`: Model-ready feature matrix

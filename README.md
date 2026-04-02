# AMUO-Fusion: Animal-Derived Natural Medicines Unpleasant-Odor Fusion (AMUO-Fusion) model #

# Overview #
AMUO-Fusion is a machine learning model for predicting the odor classification of volatile compounds. The model combines molecular fingerprints (PubChem, MACCS, FCFP) with odor descriptor features to classify compounds into categories of unpleasant and neutral or pleasant. 

# Requirement #
Python 3.10+
Required packages (install via pip):
- pandas (>=3.0.2)
- numpy (>=2.4.4)
- scikit_learn (>=1.8.0)
- rdkit (>=2024.3.3)
- joblib (>=1.4.2)

# Data Preparation #
Prepare your input data in Excel format with the following columns (see Example data.xlsx):
- **SMILES**: Molecular structure in SMILES notation
- **Odor descriptors**: (Optional) Column-separated odor descriptors

# Running Predictions #
Modify the following paths in AMUO-Fusion_Prediction.py, then run the script:
MODEL_PATH = r"C:\...\...\AMUO-Fusion.pkl"
DATA_PATH = r"C:\...\...\Example data.xlsx"
RESULT_PATH = r"C:\...\...\Prediction results of Volatile Compounds without Odor Annotation.xlsx"

# Output #
The script generates an Excel file with the following columns:
- **SMILES**: Molecular structure in SMILES notation
- **Predicted_Class**: Predicted category (0 or 1)
- **Probability_Class0**: Probability of belonging to class 0 (neutral or pleasant odor)
- **Probability_Class1**: Probability of belonging to class 1 (unpleasant odor)
- Odor descriptors' columns



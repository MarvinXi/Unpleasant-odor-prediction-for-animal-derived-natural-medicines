import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

class RDKitECFPTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, radius=2, n_bits=1024):
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fps = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits
                )
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(self.n_bits))
        return np.array(fps)


class FCFPTransformer(RDKitECFPTransformer):
    """Transformer for generating FCFP fingerprints from SMILES strings."""

    def transform(self, X, y=None):
        fps = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=True
                )
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(self.n_bits))
        return np.array(fps)


# Set your paths here
MODEL_PATH = r"C:\...\...\AMUO-Fusion.pkl"
DATA_PATH = r"C:\...\...\Example data.xlsx"
RESULT_PATH = r"C:\...\...\Prediction results of Volatile Compounds without Odor Annotation.xlsx"

# Load model
print("🔍 Loading model...")
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
mlb = model_data['mlb']
pipelines = model_data['fingerprint_pipelines']

# Load data
print("📁 Loading data...")
data = pd.read_excel(DATA_PATH)

# Check SMILES column
if 'SMILES' not in data.columns:
    print("❌ Error: Data must contain 'SMILES' column")
    exit()

# Prepare features
print("⚙️ Preparing features...")
features = []

# Molecular fingerprints
for name, pipe in pipelines.items():
    try:
        fps = pipe.transform(data['SMILES'].tolist())
        features.append(fps)
    except:
        # Create zero matrix if error occurs
        if name == "PubChem":
            zero_fp = np.zeros((len(data), 881))
        elif name == "MACCS":
            zero_fp = np.zeros((len(data), 166))
        else:
            zero_fp = np.zeros((len(data), 1024))
        features.append(zero_fp)

# Odor features
odor_cols = [c for c in data.columns if 'odor' in c.lower() or 'Odor' in c]
if odor_cols:
    # Do not create '_odor' intermediate column, process directly
    odor_data = data[odor_cols[0]].fillna('').apply(
        lambda x: [o.strip() for o in str(x).split(';') if o.strip()]
    )
    odor_features = mlb.transform(odor_data)
else:
    odor_features = np.zeros((len(data), len(mlb.classes_)))

features.append(odor_features)

# Combine and predict
X = np.hstack(features).astype(np.int8)
predictions = model.predict(X)
probs = model.predict_proba(X)

# Save results
results = pd.DataFrame({
    'SMILES': data['SMILES'],
    'Predicted_Class': predictions,
    'Probability_Class0': probs[:, 0].round(3),
    'Probability_Class1': probs[:, 1].round(3)
})

# Add original data (do not add intermediate column '_odor')
for col in data.columns:
    if col != 'SMILES':
        results[col] = data[col]

# If there are odor description columns in the data, ensure they are properly retained
if odor_cols:
    for col in odor_cols:
        if col not in results.columns:
            results[col] = data[col]

results.to_excel(RESULT_PATH, index=False)

# Print results
print(f"✅ Prediction completed! Results saved to: {RESULT_PATH}")
print(f"📊 Statistics: Samples predicted as unpleasant odor (1) = {(predictions == 1).sum()} samples, Samples predicted as 0 = {(predictions == 0).sum()} samples")

# Set pandas display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No width limit
pd.set_option('display.max_colwidth', 30)  # Limit column width

# Select columns to display
display_cols = ['SMILES', 'Predicted_Class', 'Probability_Class1']

# Add description columns from original data
for col in data.columns:
    if col != 'SMILES' and col not in display_cols:
        # Only add columns that appear to be descriptive
        if 'Descr' in col or 'odor' in col.lower() or 'Odor' in col:
            display_cols.append(col)

## Day 3 — Python for AI

A focused day to get productive with Python for AI/ML: environment setup, Jupyter, essential libraries (NumPy, pandas, Matplotlib/Seaborn, scikit-learn), and hands-on exercises.

### Objectives
- Understand and set up an isolated Python environment
- Use Jupyter (Notebook/Lab) effectively
- Manipulate arrays with NumPy and tables with pandas
- Visualize data with Matplotlib/Seaborn
- Train a simple ML model with scikit-learn

## Prerequisites
- Python 3.10+ installed
- Git installed
- Editor: Cursor or VS Code recommended

## Quick Start
### 1) Create and activate a virtual environment
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
```

Windows (PowerShell):
```powershell
# If activation is blocked, you may need this (admin PowerShell):
Set-ExecutionPolicy -ExecutionPolicy ByPass -scope LocalMachine -Force

python -m venv .venv
. .venv\Scripts\Activate.ps1
python -V
```

### 2) Install essentials
```bash
python -m pip install --upgrade pip
pip install jupyter jupyterlab ipykernel numpy pandas matplotlib seaborn scikit-learn
python -m ipykernel install --user --name day3 --display-name "Python (day3)"
```

### 3) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Choose the "Python (day3)" kernel when creating notebooks.

## Recommended Project Layout
```
/Users/myohanhtet/Herd/ai/day3
├─ notebooks/
│  ├─ 01_numpy.ipynb
│  ├─ 02_pandas.ipynb
│  ├─ 03_viz.ipynb
│  └─ 04_sklearn.ipynb
├─ scripts/
│  └─ prepare_data.py
├─ data/
│  └─ (place raw datasets here; keep large files out of git)
├─ Readme.md
└─ Note.txt
```

## Verify your setup (Notebook cell)
```python
import sys, numpy as np, pandas as pd, sklearn, matplotlib, seaborn as sns
print(sys.version)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)
```

## Exercises
### 1) NumPy fundamentals
- Create arrays, dtypes, shapes; practice slicing, masking, broadcasting
- Implement cosine similarity for two vectors and for matrix–matrix
```python
import numpy as np

def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    # A: [n_samples, n_features]
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A_norm @ A_norm.T
```

### 2) pandas data wrangling
- Load a CSV (e.g., `data/titanic.csv` or any tabular dataset)
- Clean: handle missing values, type conversions
- Feature engineering: create numeric features, categorical encodings
- GroupBy, pivot, sorting, filtering; export a clean dataset

### 3) Visualization
- Plot distributions and relationships (hist, boxplot, pairplot)
```python
import seaborn as sns
import matplotlib.pyplot as plt

df = ...  # your DataFrame
sns.pairplot(df.select_dtypes(include=['number']))
plt.show()
```

### 4) scikit-learn mini-project
- Split data: train/test
- Train a simple model (e.g., LogisticRegression or RandomForest)
- Evaluate: accuracy, precision/recall/F1; visualize confusion matrix
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X, y = df.drop(columns=["target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
```

### 5) Stretch goals
- Implement k-NN from scratch with NumPy; compare to scikit-learn
- Use pipelines with preprocessing (scalers/encoders) + model
- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`

## Handy CLI usage
Run a script from `scripts/` inside the activated environment:
```bash
python scripts/prepare_data.py --input data/raw.csv --output data/clean.csv
```

Freeze dependencies for reproducibility:
```bash
pip freeze > requirements.txt
```

## Troubleshooting
- macOS might have multiple Pythons; prefer `python3` and ensure the venv is active (`which python`).
- Windows PowerShell execution policy can block venv activation; use the command shown above.
- If Jupyter doesn’t show the "Python (day3)" kernel: reinstall the ipykernel line and then restart Jupyter.
- SSL/Certificates (macOS): if `pip` fails due to SSL, ensure Xcode Command Line Tools are installed, or try `python -m pip install --upgrade certifi`.
- If plots don’t render: ensure `%matplotlib inline` (Notebook) or `plt.show()` (scripts) is used.

## What to submit (if applicable)
- A short write-up (bulleted) of what you learned
- Links to notebooks and key figures
- A brief description of your model, metrics, and next steps

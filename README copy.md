# Spotify Machine Learning Project

This project implements a complete, modular machine learning pipeline for analyzing Spotify track data. It covers the full workflow: data loading, preprocessing, feature engineering, dimensionality reduction (PCA/UMAP), model training (XGBoost and CatBoost), hyperparameter tuning, explainability using SHAP, and class-wise evaluation across three dataset splits (A, B, and C).  
Each stage is implemented in separate scripts to ensure clarity, reproducibility, and ease of experimentation.

---

## Project Structure

```
SPOTIFY PROJECT/
│
├── A/                                  # Split A pipeline
│   ├── spotify_30_percent.csv
│   ├── main.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── dimensionality_reduction.py
│   ├── hyperparameter_tuning.py
│   ├── models.py
│   ├── EDA.py
│   ├── SHAP.py
│   ├── classwise_performance.py
│   ├── feature_importance.py
│   ├── comparison.py
│   ├── best_xgb_params.json
│   └── final_xgboost.model
│
├── B/                                  # Split B pipeline
│   ├── spotify_30_percent_B.csv
│   ├── B_main.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── models.py
│   └── best_xgb_params.json
│
├── C/                                  # Split C pipeline
│   ├── spotify_30_percent_C.csv
│   ├── C_main.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── models.py
│   └── best_xgb_params.json
│
├── EDA/                                # Central visualizations
├── plots/                              # Saved plots
├── catboost_info/                      # CatBoost metadata
├── venv/                               # Virtual environment (local only)
│
├── baseline_retrain.py
├── final_xgboost.model
├── 30p.py
├── 30_b_c.py
├── requirements.txt
└── README.md
```

---

##  Brief Description

The project explores Spotify track characteristics using classical machine learning models.  
It compares three dataset splits (A, B, C) — each representing a 30% sampled subset — to evaluate model stability, feature robustness, and generalization.  
All preprocessing, modeling, explainability, and evaluation procedures are automated and modular.

---

##  Environment Setup 

### **1. Create a virtual environment**
Recommended for consistent package management.

```
python3 -m venv venv
source venv/bin/activate
```

### **2. Install all dependencies**
```
pip install -r requirements.txt
```

### **3. Verify installation**
```
python -c "import pandas, sklearn, xgboost, shap, umap"
```

If no errors appear, the environment is ready.

---

## Data Access

### **Google Drive link (all datasets uploaded):**  
**[Drive Folder Link ]**  
*([
(https://drive.google.com/drive/folders/1Lvtmkhd3A3E9IhGctXpJlH-AcNjoPTtd?usp=share_link)])*

### **Local Copies**  
These same CSVs are also included inside the respective A,B and C folders of the project:

- `A/spotify_30_percent.csv`
- `B/spotify_30_percent_B.csv`
- `C/spotify_30_percent_C.csv`


### **How to use the data**
Each pipeline’s main script automatically loads the corresponding CSV in its folder:
```python
df = pd.read_csv("spotify_30_percent.csv")
```
No additional configuration is required.

---

## ▶ How to Run the Code

### **Run Split A pipeline**
```
python A/main.py
```

### **Run Split B pipeline**
```
python B/B_main.py
```

### **Run Split C pipeline**
```
python C/C_main.py
```

### **Baseline Retrain**
```
python baseline_retrain.py
```

All outputs (models, plots, SHAP diagrams, metrics) are stored automatically in their respective folder paths.

---

##  Test Cases to Verify Correct Execution

### **1. Environment test**
```
python -c "import pandas, numpy, xgboost, shap, umap"
```

### **2. Preprocessing test (Split A)**
```
python -c "from A.preprocessing import preprocess; import pandas as pd; df=pd.read_csv('A/spotify_30_percent.csv'); print(preprocess(df).head())"
```

### **3. Model training smoke test**
```
python A/main.py
```
If it completes without errors and produces:
- `best_xgb_params.json`
- `final_xgboost.model`
- SHAP plots  
…then the pipeline is functioning correctly.

### **4. Dimensionality reduction test**
```
python A/dimensionality_reduction.py
```

---

##  Known Issues & Important Notes

- **Python 3.13** is **not fully compatible** with XGBoost and UMAP.  
  Use **Python 3.12** for stable execution.

- Some directories (e.g., `catboost_info` and `EDA/`) are generated automatically after running the scripts.

- SHAP plots may take a few minutes.

- When running multiple pipelines sequentially, ensure the correct virtual environment is activated.

- CSVs are available **both locally** and **on Google Drive** to avoid path failures.

---

##  Final Notes

This repository is designed for clarity and reproducibility.  
Every component — preprocessing, modeling, feature engineering, explainability, and evaluation — is modular and can be run independently or as a full pipeline.

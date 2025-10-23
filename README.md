# ICU Survival Prediction using Machine Learning

A comprehensive machine learning project for predicting patient mortality in intensive care units.

## 🎯 Project Overview

This project develops classification models to automatically identify ICU patients at risk of in-hospital mortality using real patient data from Beth Israel Deaconess Medical Center in Boston. The models leverage electronic health record (EHR) data from the first 48 hours of ICU admission to predict patient outcomes, potentially helping clinicians target interventions and improve patient management.

**Data Source**: PhysioNet.org - Real patient admissions made available for research and educational purposes.

## 🏥 Clinical Motivation

With rapid technological advancement, hospitals collect vast amounts of patient data in electronic health records. This data has the potential to:
- Identify patients at greatest risk of adverse outcomes
- Predict drug efficacy for individual patients
- Match patients with optimal treatments
- Support clinical decision-making with data-driven insights

## ✨ Key Features

- **Feature Engineering**: Transform raw timestamped EHR data into structured feature vectors
- **Missing Data Handling**: Sophisticated imputation techniques for incomplete medical records
- **Regularized Models**: L1 and L2 regularized logistic regression for robust prediction
- **Hyperparameter Tuning**: 5-fold stratified cross-validation for optimal model selection
- **Class Imbalance Solutions**: Custom class weighting to address mortality rate imbalance
- **Kernel Methods**: RBF kernels for non-linear decision boundaries
- **Multiple Metrics**: Comprehensive evaluation using 7 performance measures
- **Bootstrapped Confidence Intervals**: 95% CI estimation for test performance

## 📊 Dataset

### Overview
- **Total Patients**: 12,000 ICU admissions
- **Training Set**: 10,000 patients (2,000 for initial exploration)
- **Held-out Test Set**: 2,000 patients
- **Time Window**: First 48 hours of ICU admission
- **Data Format**: Timestamped observations in CSV files

### Variables
**Time-Invariant** (collected at admission):
- Demographics: Age, Gender, Height, Weight
- ICU Type (Coronary care, Cardiac surgery, etc.)
- Timestamp: 00:00

**Time-Varying** (measured during stay):
- Vital Signs: Heart Rate (HR), Temperature, Blood Pressure
- Lab Values: Glucose, Creatinine, Urine Output
- Clinical Scores: Glasgow Coma Scale (GCS)
- Frequency: Measured one time, many times, or not at all

### Target Labels
- **In-hospital Mortality**: {1: died, -1: survived}
- **30-day Mortality**: {1: died within 30 days, -1: survived past 30 days}

## 🛠️ Technology Stack

- **Language**: Python 3.11
- **Environment**: Anaconda
- **Core Libraries**:
  - scikit-learn 1.5.1 (Machine learning models)
  - pandas 2.2.2 (Data manipulation)
  - numpy (Numerical computing)
  - matplotlib 3.9.2 (Visualization)
  - PyYAML 6.0.1 (Configuration management)
  - tqdm 4.66.5 (Progress bars)

## 📁 Project Structure

```
project1/
├── project1.py              # Main implementation file
├── helper.py                # Data loading utilities
├── test_output.py           # Output validation script
├── data/
│   ├── files/              # Individual patient CSV files (12,000)
│   ├── labels.csv          # Patient outcomes
│   └── config.yaml         # Variable definitions
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11.9
- Anaconda or Miniconda

### Installation

1. Create conda environment with required packages:
```bash
conda create --name 445_p1 python=3.11.9 \
  matplotlib=3.9.2 \
  pandas=2.2.2 \
  pyyaml=6.0.1 \
  scikit-learn=1.5.1 \
  tqdm=4.66.5
```

2. Activate the environment:
```bash
conda activate 445_p1
```

3. Download the project dataset from Canvas:
```bash
# project1.zip contains the full dataset
unzip project1.zip
```

## 💻 Implementation Components

### 1. Feature Extraction (20 pts)

**Implemented Functions:**
- `generate_feature_vector(df)`: Transform raw patient data into feature vectors

**Feature Engineering Strategy:**
- **Time-Invariant Variables**: Use raw values, replace -1 with np.nan
- **Time-Varying Variables**: Compute max of all measurements
  - Feature naming: `max_{Variable}` (e.g., `max_HR` for heart rate)
  - Missing measurements represented as np.nan

**Example Transformation:**
```
Raw Data (48 hours of observations)
→ Feature Vector: [Age=56, Height=NaN, max_HR=122, max_temp=37, max_RespRate=NaN, ...]
```

### 2. Data Preprocessing

**Implemented Functions:**
- `impute_missing_values(X)`: Mean imputation for missing values
- `normalize_feature_matrix(X)`: Min-max normalization to [0, 1]

**Preprocessing Pipeline:**
1. Generate feature vectors for all patients
2. Aggregate into feature matrix (40 features × n patients)
3. Impute missing values using column-wise mean
4. Normalize each feature to [0, 1] range
5. Split into 80% train / 20% test

### 3. Model Selection & Hyperparameter Tuning (25 pts)

**Implemented Functions:**
- `cv_performance(clf, X, y, metric, k=5)`: K-fold cross-validation
- `select_param_logreg(X, y, metric, k, C_range, penalties)`: Grid search for logistic regression
- `select_param_RBF(X, y, metric, k, C_range, gamma_range)`: Grid search for RBF kernel

**Hyperparameter Search Space:**
- **Regularization Strength (C)**: [10⁻³, 10⁻², 10⁻¹, 1, 10, 10², 10³]
- **Penalties**: L1 (Lasso), L2 (Ridge)
- **RBF Kernel Gamma (γ)**: [0.001, 0.01, 0.1, 1, 10, 100]

**Cross-Validation Strategy:**
- 5-fold stratified CV (maintains class proportions)
- Selection criteria: Best mean performance across folds

### 4. Performance Evaluation

**Metrics Implemented:**
1. **Accuracy**: Overall correctness
2. **Precision**: Positive predictive value
3. **F1-Score**: Harmonic mean of precision and recall
4. **AUROC**: Area under ROC curve
5. **Average Precision**: Area under precision-recall curve
6. **Sensitivity (Recall)**: True positive rate
7. **Specificity**: True negative rate

**Confidence Intervals:**
- 1,000 bootstrap samples of test set
- Report median performance and 95% CI (2.5th and 97.5th percentiles)

### 5. Class Imbalance Handling (15 pts)

**Objective Function with Class Weights:**
```
min_θ ||θ||² + W_p * C * Σ(Loss for positive examples) 
              + W_n * C * Σ(Loss for negative examples)
```

**Strategies Explored:**
- Arbitrary class weights (W_n=1, W_p=50)
- Cross-validated optimal weights
- ROC curve analysis for threshold selection

### 6. Kernel Methods (20 pts)

**Implemented Functions:**
- `select_param_RBF(X, y, metric, k, C_range, gamma_range)`

**Kernel Ridge Regression:**
- RBF (Radial Basis Function) kernel for non-linear decision boundaries
- Comparison with linear models
- Hyperparameter optimization for C and γ

**Key Insight**: RBF kernels can capture non-linear relationships in medical data that linear models miss.

### 7. Challenge: 30-Day Mortality Prediction (20 pts)

**Objective**: Predict 30-day post-discharge mortality using all 10,000 training patients

**Advanced Techniques Encouraged:**
- Enhanced feature engineering:
  - Treat numerical and categorical variables differently
  - Additional summary statistics (mean, std, min, trend)
  - Temporal binning (split 48 hours into windows)
  - Feature selection
- Alternative imputation methods
- Advanced preprocessing (scaling, transformations)

**Evaluation**: AUROC and F1-score on held-out test set

## 📈 Typical Results

### Model Performance (Example)

**Baseline Logistic Regression (L2, C=1.0):**
- AUROC: ~0.75-0.80
- F1-Score: ~0.35-0.45
- Sensitivity: ~0.50-0.65
- Specificity: ~0.75-0.85

**With Class Weights (W_p=5-10):**
- Improved Sensitivity: +10-15%
- Slightly reduced Specificity: -5%
- Better balance for clinical application

### Feature Importance

**Most Predictive Features (Positive Coefficients → Higher Mortality Risk):**
- Advanced age
- Lower Glasgow Coma Scale
- Higher creatinine levels
- Elevated heart rate

**Protective Features (Negative Coefficients → Lower Mortality Risk):**
- Stable vital signs
- Normal lab values
- Certain ICU types

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- **Feature Engineering**: Transforming raw medical time-series data
- **Statistical Learning**: Regularized linear models, kernel methods
- **Model Selection**: Cross-validation, hyperparameter tuning
- **Evaluation**: Multiple metrics, confidence intervals, ROC analysis
- **Class Imbalance**: Cost-sensitive learning, class weighting
- **Medical ML**: Working with real healthcare data, clinical interpretation
- **Software Engineering**: scikit-learn ecosystem, reproducible pipelines

## 🔬 Key Machine Learning Concepts

### Regularization

**L1 (Lasso)**:
- Encourages sparsity (feature selection)
- Some coefficients become exactly zero
- Useful for interpretability

**L2 (Ridge)**:
- Encourages small coefficients
- No exact zeros (all features retained)
- Better for prediction when many features matter

### Cross-Validation
```python
# Stratified 5-fold CV maintains class proportions
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=False)
```

### Bootstrapping
```python
# 95% Confidence Interval from 1000 bootstrap samples
for _ in range(1000):
    sample_idx = np.random.choice(n, size=n, replace=True)
    X_boot, y_boot = X[sample_idx], y[sample_idx]
    scores.append(metric(clf, X_boot, y_boot))
CI = (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
```

## 📊 Visualization Examples

### Model Complexity Analysis
- **L0-norm vs C**: Shows feature selection behavior
- **ROC Curves**: Compare models with different class weights
- **Performance vs Gamma**: RBF kernel sensitivity analysis

## ⚠️ Important Considerations

### Clinical Context
- **False Negatives** (missing at-risk patients) may be more costly than False Positives
- Model should be **calibrated** for clinical decision support
- **Interpretability** is crucial for physician trust

### Data Quality
- Real patient data with missing values
- Class imbalance (more survivors than deaths)
- Time-dependent measurements require thoughtful aggregation

### Ethical Considerations
- Patient privacy (data anonymized)
- Bias in training data may affect certain populations
- Model should augment, not replace, clinical judgment

## 🧪 Testing & Validation

### Output Validation
```bash
python test_output.py -i phling.csv
```

Ensures proper format for challenge submission:
- Binary predictions (y_label)
- Real-valued risk scores (y_score)

## 📚 Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PhysioNet Challenge](https://physionet.org/)
- [ROC Curve Analysis](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Confusion Matrix Guide](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## 📧 Contact

Your Name - [GitHub](https://github.com/HaolingPu)

---

**Built with 🏥 at the University of Michigan**

*This project uses real patient data for educational purposes. All data has been de-identified to protect patient privacy.*

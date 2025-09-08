# Titanic Survival Prediction v08 (XGBoost - Accuracy Focused, No SMOTE, Sequential Execution)

## Overview
**Version**: 08  
**Primary Goal**: Develop an XGBoost model for the Titanic survival prediction task, optimized for accuracy as the primary metric, aligning with Kaggle competition requirements or business needs and learning ML.

## Key Changes from v07
- Configured `GridSearchCV` to use `scoring='accuracy'`.
- Excluded SMOTE to evaluate performance on the natural class distribution.
- Focused overfitting analysis and evaluation on accuracy and threshold-dependent metrics.
- Ensured sequential execution (`n_jobs=1`) for `XGBClassifier`, `GridSearchCV`, and `cross_val_predict` to resolve instability issues with parallel processing.

## 1. Data Preparation & Initial Setup
- **Libraries**: Utilized `numpy`, `pandas`, `xgboost`, `sklearn` (preprocessing, model selection, metrics), `re` (regular expressions), `matplotlib`, `seaborn` (visualizations), and filtered warnings for cleaner output.
- **Data Loading**: Loaded `train.csv` and `test.csv` into pandas DataFrames. Preserved `PassengerId` for submission.
- **Combined Processing**: Added a placeholder `Survived` column (-1) to the test set, concatenated with training data into `full_df` for consistent preprocessing.
- **Imbalance Handling**: No SMOTE or other sampling techniques applied. Tuned on the original class distribution (61.6% did not survive, 38.4% survived), prioritizing accuracy.

## 2. Core Model: XGBoost Classifier
- **Algorithm**: XGBoost, chosen for high performance, regularization, and ability to capture complex relationships.
- **Rationale**: Efficient, handles missing values, and performs well on tabular data.
- **Instantiation**: Used `XGBClassifier` with `random_state=42`, `use_label_encoder=False`, `eval_metric='logloss'`, and `n_jobs=1` for stable sequential execution.

## 3. Feature Engineering & Preprocessing
Applied a refined pipeline from v06/v07:
- **Text-Based Features**:
  - **Title**: Extracted from `Name` (e.g., Mr, Miss, Mrs, Master, Rare).
  - **TicketPrefix**: Extracted from `Ticket` (numeric tickets labeled 'NUM', rare prefixes as 'RARE_TICKET').
  - **Cabin-Based**: `Deck` (first letter of `Cabin`, 'M' for missing), `HasCabin` (binary), `CabinMultiple` (count of cabins).
- **Family-Based Features**:
  - `FamilySize`: `SibSp + Parch + 1`.
  - `IsAlone`: Binary (1 if `FamilySize` is 1).
- **Numerical Transformations**:
  - `Fare_log`: Log-transformed `Fare` (`np.log1p`).
  - `FarePerPerson`: `Fare` divided by `TicketFrequency`.
  - `FarePerPerson_log`: Log-transformed `FarePerPerson`.
- **Missing Values**:
  - `Age`: Imputed with median per `Title` group, fallback to overall median.
  - `Embarked`: Filled with mode.
  - `Fare`: Imputed with median per `Pclass`.
- **Binning**:
  - `AgeBin`: Categorized into 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'.
  - `FareBin_log`: Quantile-based bins ('Low', 'Medium', 'High', 'VeryHigh') using `pd.qcut`.
- **Interaction Features**:
  - `Pclass_Sex`: Concatenation of `Pclass` and `Sex`.
  - `AgeBin_Pclass`: Concatenation of `AgeBin` and `Pclass`.
- **Encoding**:
  - **Label Encoding**: Applied to `AgeBin` (`AgeBin_Labeled`) and `FareBin_log` (`FareBin_log_Labeled`).
  - **One-Hot Encoding**: Applied to `Embarked`, `Sex`, `Title`, `Deck`, `TicketPrefix`, `Pclass_Sex`, `AgeBin_Pclass`.
- **Dropped Columns**: Removed original columns (`Name`, `Ticket`, `Cabin`, `PassengerId`, etc.) after transformation.

## 4. Data Finalization
- **Splitting**: Split `full_df` into `X_train`, `X_test`, and `Y_train` (from `train_df`).
- **Column Realignment**: Ensured `X_train` and `X_test` have identical columns, filling missing columns with 0.

## 5. Hyperparameter Tuning
- **Estimator**: `XGBClassifier` with `n_jobs=1`.
- **GridSearchCV**:
  - **Parameters**:
    - `n_estimators`: [100, 200, 300, 400]
    - `max_depth`: [1]
    - `learning_rate`: [0.03, 0.05]
    - `subsample`: [0.7, 0.8]
    - `colsample_bytree`: [0.7, 0.9]
    - `gamma`: [0.05, 0.1]
    - `reg_alpha`: [0, 0.01]
    - `reg_lambda`: [0.1, 1.0]
    - `min_child_weight`: [1, 3]
  - **Cross-Validation**: `StratifiedKFold` (n_splits=5, shuffle=True, random_state=42).
  - **Execution**: Sequential (`n_jobs=1`) to avoid parallelism errors.
  - **Scoring**: `accuracy`.
- **Results**:
  - **Best CV Accuracy**: 0.8395
  - **Best Parameters**:
    - `colsample_bytree`: 0.9
    - `gamma`: 0.05
    - `learning_rate`: 0.03
    - `max_depth`: 1
    - `min_child_weight`: 1
    - `n_estimators`: 400
    - `reg_alpha`: 0
    - `reg_lambda`: 0.1
    - `subsample`: 0.8

## 6. Model Diagnostics
- **Execution**: All `cross_val_predict` operations used `n_jobs=1`.
- **Accuracy Comparison**:
  - Full Training Accuracy: 0.8361
  - CV Accuracy: 0.8395
  - Difference: -0.0034 (negligible, indicates no overfitting).
  - CV Accuracy Std Dev: 0.0077.
- **Learning Curves**:
  - Training and CV accuracy converged at ~0.83-0.84 with ~700 examples.
  - Minimal gap confirms good generalization.
- **Validation Curves** (for `max_depth`):
  - `max_depth=1` optimal, with training and CV accuracy ~0.83-0.84.
  - Higher depths increased training accuracy but not CV accuracy.
- **Classification Report** (CV, Class 1: Survived):
  - Precision: 0.82
  - Recall: 0.75
  - F1-Score: 0.78
  - Accuracy: 0.84
- **Confusion Matrix** (CV): [[492, 57], [86, 256]]
  - True Negatives: 492
  - False Positives: 57
  - False Negatives: 86
  - True Positives: 256
- **ROC Curve & AUC** (CV):
  - Training AUC: 0.8925
  - CV AUC: 0.8723
  - Small gap (0.0202) confirms strong discriminative power.

## 7. Feature Importances
- **Top Features**: `Sex_female` (0.188), `Title_Mr` (0.141), `Pclass` (0.107), `HasCabin` (0.085), `Deck_M` (0.048).
- **Interpretation**: Aligns with domain knowledge (gender, title, class, cabin data as key predictors).

## 8. Prediction & Submission
- Predicted survival on `X_test` using `best_model`.
- Generated `titanic_prediction_v08.csv` with `PassengerId` and `Survived` predictions.

## 9. Python Libraries
- `pandas`, `numpy`, `re`, `xgboost`, `scikit-learn` (preprocessing, model selection, metrics), `matplotlib`, `seaborn`, `joblib`.
- Note: `imblearn` not used in v08.

## 10. Key Achievements
- **Accuracy**: Achieved 0.8395 CV accuracy.
- **Generalization**: Negligible overfitting, confirmed by learning/validation curves and metrics.
- **Hyperparameters**: Identified optimal parameters for accuracy and generalization.
- **Stability**: Resolved parallelism issues with sequential execution.
- **Diagnostics**: Comprehensive analysis (learning curves, validation curves, classification reports, ROC AUC).

This v08 model is a robust, well-tuned predictor for Titanic survival, optimized for accuracy and reliably executed.

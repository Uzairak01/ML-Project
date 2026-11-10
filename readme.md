# Bank Customer Churn Prediction

This project builds a machine learning model to predict whether a bank customer is likely to leave the bank (churn). It uses a customer dataset with demographic, account, and behavioral features and applies preprocessing and classification techniques to evaluate predictive performance.

## Project Goal

Predict the `Exited` target variable:
- `1` = customer churned
- `0` = customer remained active

## Dataset

The project uses the dataset stored in:
- `DataSet/Churn_Modelling.csv`

Key features include:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target)

## Project Directory Structure
```text
GroupXX_ML_Project/
├── Code/
│   ├── ML_Project_Churn.ipynb    # Main Jupyter Notebook with complete workflow
│   └── requirements.txt         # Required Python packages and versions
├── Dataset/
│   ├── Churn_Modelling.csv      # Raw CSV dataset
│   └── Dataset Information.txt  # Attribute metadata and descriptions
├── Presentation/
│   └── ML_Project_Presentation.pptx # Final viva & presentation slide deck
├── Report/
│   ├── ML_Project_Report.pdf     # Full IEEE-formatted academic project report
│   ├── EDA_Target_Distribution.png
│   ├── EDA_Categorical_Features.png
│   ├── EDA_Numerical_Distributions.png
│   ├── EDA_Correlation_Matrix.png
│   └── Model_Evaluation_Plots.png
└── README.md                    # Project documentation
```

## Bank Customer Churn Prediction

This project builds a machine learning model to predict whether a bank customer is likely to leave the bank (churn). It uses a customer dataset with demographic, account, and behavioral features and applies preprocessing and classification techniques to evaluate predictive performance.

## Project Goal

Predict the `Exited` target variable:
- `1` = customer churned
- `0` = customer remained active

## Dataset

The project uses the dataset stored in:
- `DataSet/Churn_Modelling.csv`

Key features include:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target)

## Project Structure

- `Code/ML_Project_Churn.ipynb` — main notebook with preprocessing, model training, and evaluation
- `Code/dependency_requirements.txt` — project dependencies
- `DataSet/Churn_Modelling.csv` — churn dataset
- `DataSet/Dataset_Information.txt` — dataset notes
- `Report/` — project report outputs
- `Presentations/` — presentation materials

## Workflow

The notebook follows a standard machine learning workflow:

1. Load the churn dataset
2. Remove irrelevant identifier columns such as `RowNumber`, `CustomerId`, and `Surname`
3. Split the data into training and testing sets using stratification
4. Separate numeric and categorical variables
5. Apply preprocessing:
   - `StandardScaler` for numeric columns
   - `OneHotEncoder` for categorical columns
6. Train a classifier and compare baseline performance
7. Evaluate the model with metrics such as:
   - classification report
   - confusion matrix
   - ROC AUC score

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Setup

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate     # On Windows
pip install -r Code/dependency_requirements.txt
```

## Run the Project

Open the notebook:

```bash
cd "Bank_churn_ML_Project/Code"
jupyter notebook ML_Project_Churn.ipynb
```

## Current Work Completed

The project has already completed the core machine learning workflow and model evaluation stage:

- Loaded the churn dataset and cleaned irrelevant identifier columns: `RowNumber`, `CustomerId`, and `Surname`
- Split the data into training and testing sets using stratification to avoid class imbalance leakage
- Separated numerical and categorical variables and built a preprocessing pipeline using:
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features
- Trained and compared multiple classification models:
  - Baseline Dummy Classifier
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - XGBoost
- Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC

### Best Results Observed

From the current notebook execution:

- Random Forest achieved the highest performance:
  - Accuracy: `0.8683`
  - Precision: `0.7842`
  - Recall: `0.4877`
  - F1-Score: `0.6014`
  - ROC-AUC: `0.8652`
- XGBoost also performed strongly:
  - Accuracy: `0.8637`
  - Precision: `0.7371`
  - Recall: `0.5139`
  - F1-Score: `0.6056`
  - ROC-AUC: `0.8487`

### Hyperparameter Tuning

The notebook also includes tuning for the top models:

- Tuned Random Forest parameters: `{'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 200}`
- Tuned XGBoost parameters: `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}`

These tuning runs confirm that the project has moved beyond basic model training and is now focused on performance optimization and result interpretation.

## Notes

This project is suitable for learning:
- feature engineering
- train/test splitting
- preprocessing pipelines
- class imbalance handling considerations
- classification model evaluation
- model comparison and hyperparameter tuning

## License

This project is intended for educational and learning purposes.

- `Code/ML_Project_Churn.ipynb` — main notebook with preprocessing, model training, and evaluation
- `DataSet/Churn_Modelling.csv` — churn dataset
- `DataSet/Dataset_Information.txt` — dataset notes
- `Report/` — project report outputs
- `Presentations/` — presentation materials

## Workflow

The notebook follows a standard machine learning workflow:

1. Load the churn dataset
2. Remove irrelevant identifier columns such as `RowNumber`, `CustomerId`, and `Surname`
3. Split the data into training and testing sets using stratification
4. Separate numeric and categorical variables
5. Apply preprocessing:
   - `StandardScaler` for numeric columns
   - `OneHotEncoder` for categorical columns
6. Train a classifier and compare baseline performance
7. Evaluate the model with metrics such as:
   - classification report
   - confusion matrix
   - ROC AUC score

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Setup

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate     # On Windows
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run the Project

Open the notebook:

```bash
cd "Bank_churn_ML_Project/Code"
jupyter notebook ML_Project_Churn.ipynb
```

## Notes

This project is suitable for learning:
- feature engineering
- train/test splitting
- preprocessing pipelines
- class imbalance handling considerations
- classification model evaluation

## License

This project is intended for educational and learning purposes.

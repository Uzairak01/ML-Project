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

## Project Structure

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

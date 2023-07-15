# Loan Status Prediction

This repository contains code for predicting loan status using machine learning techniques. The dataset used for training and prediction is stored in the `loan_prediction.csv` file.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- tkinter (for GUI)

## Getting Started
1. Clone the repository: `git clone https://github.com/your-username/loan-status-prediction.git`
2. Navigate to the project directory: `cd loan-status-prediction`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the code: `python loan_prediction.py`

## Code Description
The code performs the following steps:

1. Load the dataset using pandas:
   ```python
   import pandas as pd
   data = pd.read_csv('loan_prediction.csv')
   ```

2. Data exploration and preprocessing:
   - Display the first few rows of the dataset:
     ```python
     data.head()
     ```
   - Display the last few rows of the dataset:
     ```python
     data.tail()
     ```
   - Get the dimensions of the dataset (number of rows and columns):
     ```python
     data.shape
     ```
   - Print the number of rows and columns:
     ```python
     print("Number of Rows:",data.shape[0])
     print("Number of Columns:",data.shape[1])
     ```
   - Display information about the dataset:
     ```python
     data.info()
     ```
   - Check for missing values and calculate the percentage of missing values for each column:
     ```python
     data.isnull().sum()
     data.isnull().sum() * 100 / len(data)
     ```
   - Remove the 'Loan_ID' column:
     ```python
     data = data.drop('Loan_ID', axis=1)
     ```
   - Remove rows with missing values in specific columns:
     ```python
     columns = ['Gender', 'Dependents', 'LoanAmount', 'Loan_Amount_Term']
     data = data.dropna(subset=columns)
     ```
   - Fill missing values in the 'Self_Employed' and 'Credit_History' columns:
     ```python
     data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
     data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
     ```
   - Map categorical variables to numerical values:
     ```python
     data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).astype('int')
     data['Married'] = data['Married'].map({'Yes': 1, 'No': 0}).astype('int')
     data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype('int')
     data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0}).astype('int')
     data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 2, 'Urban': 1}).astype('int')
     data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0}).astype('int')
     ```

3. Split the data into features (X) and target variable (y):
   ```python
   X = data.drop('Loan_Status', axis=1)
   y = data['Loan_Status']
   ```

4. Perform feature scaling using StandardScaler:
   ```python
   from sklearn.preprocessing import StandardScaler
   st = StandardScaler()
   X[cols] = st.fit_transform(X[cols])
   ```

5. Split the data into training and testing sets and evaluate multiple classifiers:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.model_selection import cross_val_score
   from sklearn.metrics import accuracy_score
   import numpy as np

   model_df = {}

   def model_val(model, X, y):
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       print(f"{model} accuracy is {accuracy_score(y_test, y_pred)}")
       score = cross_val_score(model, X, y, cv=5)
       print(f"{model} Avg cross val score is {np.mean(score)}")
       model_df[model] = round(np.mean(score) * 100, 2)

   # Logistic Regression
   model = LogisticRegression()
   model_val(model, X, y)

   # Support Vector Machine
   model = svm.SVC()
   model_val(model, X, y)

   # Decision Tree Classifier
   model = DecisionTreeClassifier()
   model_val(model, X, y)

   # Random Forest Classifier
   model = RandomForestClassifier()
   model_val(model, X, y)

   # Gradient Boosting Classifier
   model = GradientBoostingClassifier()
   model_val(model, X, y)
   ```

6. Perform hyperparameter tuning using RandomizedSearchCV:
   - For Logistic Regression:
     ```python
     log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ['liblinear']}
     rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, n_iter=20, cv=5, verbose=True)
     rs_log_reg.fit(X, y)
     rs_log_reg.best_score_
     rs_log_reg.best_params_
     ```

   - For Support Vector Machine:
     ```python
     svc_grid = {'C': [0.25, 0.50, 0.75, 1], "kernel": ["linear"]}
     rs_svc = RandomizedSearchCV(svm.SVC(), param_distributions=svc_grid, cv=5, n_iter=20, verbose=True)
     rs_svc.fit(X, y)
     rs_svc.best_score_
     rs_svc.best_params_
     ```

   - For Random Forest Classifier:
     ```python
     rf_grid = {'n_estimators': np.arange(10, 1000, 10), 'max_features': ['auto', 'sqrt'],
                'max_depth': [None, 3, 5, 10, 20, 30], 'min_samples_split': [2, 5, 20, 50, 100],
                'min_samples_leaf': [1, 2, 5, 10]}
     rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)
     rs_rf.fit(X, y)
     rs_rf.best_score_
     rs_rf.best_params_
     ```

7. Train the Random Forest Classifier with the best hyperparameters:
   ```python
   X = data.drop('Loan_Status', axis=1)
   y = data['Loan_Status']

   rf = RandomForestClassifier(n_estimators=270, min_samples_split=5, min_samples_leaf=5,
                                max_features='sqrt', max_depth=5)
   rf.fit(X, y)
   ```

8. Save the trained model:
   ```python
   import joblib
   joblib.dump(rf, 'loan_status_predict')
   ```

9. Predict loan status for new data:
   ```python
   model = joblib.load('loan_status_predict')

   df = pd.DataFrame({
       'Gender': 1,
       'Married': 1,
       'Dependents': 2,
       'Education': 0,
       'Self_Employed': 0,
       'ApplicantIncome': 2889,
       'CoapplicantIncome': 0.0,
       'LoanAmount': 45,
       'Loan_Amount_Term': 180,
       'Credit_History': 0,
       'Property_Area': 1
   }, index=[0])

   result = model.predict(df)

   if result == 1:
       print("Loan Approved")
   else:
       print("Loan Not Approved")
   ```

10. GUI Application:
    - The code also provides a graphical user interface (GUI) using tkinter library.
    - Enter the values in the input fields and click the "Predict" button to see the loan status prediction.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

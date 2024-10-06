from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score

class ModelTrain:
    def __init__(self, config: ModelTrainConfig) -> None:
        self.config = config
        
    def model_preprocess(self):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        num_pipeline = Pipeline([
            ('power_transform', PowerTransformer(method='yeo-johnson')),  # Power transformation
            ('scale', StandardScaler())  # Standard scaling
        ])

        preprocess = ColumnTransformer([
            ('num_columns', num_pipeline, num_columns)
        ])
        
        return preprocess
    
    def train(self):
        # Load training and test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        target_col = 'Outcome'
        
        # Prepare features and target variable
        X_train = train_data.drop(columns=target_col, axis=1)
        X_test = test_data.drop(columns=target_col, axis=1)
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        # Preprocess the data
        preprocess_obj = self.model_preprocess()
        X_train = preprocess_obj.fit_transform(X_train)
        X_test = preprocess_obj.transform(X_test)

        # Set up Logistic Regression
        log = LogisticRegression(class_weight='balanced', solver='saga')  # 'saga' works well for large datasets

        # Define a broader parameter grid for tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Regularization strengths
            'penalty': ['l1', 'l2', 'elasticnet'],  # Include 'elasticnet' if you want to explore mixed penalty
            'max_iter': [100, 200, 300, 500],  # Maximum iterations
            'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
        }

        # Stratified K-Fold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=log, param_grid=param_grid, 
                                   scoring='accuracy', cv=skf, verbose=1, n_jobs=-1)
        
        # Fit the model using GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Log best parameters and score
        logging.info(f'Best parameters: {best_params}')
        logging.info(f'Best cross-validation score: {best_score:.4f}')

        # Train the model with the best parameters on the entire training set
        best_log = grid_search.best_estimator_
        best_log.fit(X_train, y_train)

        # Save the best model and preprocessing object
        joblib.dump(best_log, self.config.model_path)
        joblib.dump(preprocess_obj, self.config.preprocess_path)
        logging.info('Best model saved')
        logging.info('Preprocessing file saved')
        
        # Evaluate model performance on the test data
        pred = best_log.predict(X_test)
        test_accuracy = accuracy_score(y_test, pred)
        logging.info(f'Test accuracy: {test_accuracy:.4f}')

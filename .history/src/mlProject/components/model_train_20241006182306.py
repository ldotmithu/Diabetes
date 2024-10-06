from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import joblib

class ModelTrain:
    def __init__(self, config: ModelTrainConfig) -> None:
        self.config = config

    def model_preprocess(self):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                       'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Define a pipeline that includes feature selection
        num_pipeline = Pipeline([
            ('power_transform', PowerTransformer(method='yeo-johnson')),  
            ('scale', StandardScaler()),
            ('feature_selection', SelectFromModel(LogisticRegression(C=0.01, solver='liblinear')))  # Feature selection
        ])

        preprocess = ColumnTransformer([
            ('num_columns', num_pipeline, num_columns)
        ])

        return preprocess
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        target_col = 'Outcome'
        
        X_train = train_data.drop(columns=target_col, axis=1)
        X_test = test_data.drop(columns=target_col, axis=1)
        y_train = train_data[target_col]
        y_test = test_data[target_col]

        # Preprocess the data
        preprocess_obj = self.model_preprocess()
        X_train = preprocess_obj.fit_transform(X_train, y_train)
        X_test = preprocess_obj.transform(X_test)

        # Hyperparameter tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 300]
        }

        # Use GridSearchCV with the logistic regression model
        grid = GridSearchCV(LogisticRegression(class_weight='balanced', n_jobs=-1), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Fit the best model
        best_model.fit(X_train, y_train)

        # Save the model and preprocess object
        joblib.dump(best_model, self.config.model_path)
        joblib.dump(preprocess_obj, self.config.preprocess_path)
        logging.info('Model saved')
        logging.info('Preprocess file saved')

        # Predictions and evaluation
        pred = best_model.predict(X_test)
        logging.info(classification_report(y_test, pred))
        logging.info("ROC AUC Score: {}".format(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])))

if __name__ == "__main__":
    config = ModelTrainConfig()
    model_train = ModelTrain(config)
    model_train.train()

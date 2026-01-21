import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def train_model():
    print("Loading dataset...")
    df = load_data('model/titanic.csv')
    
    # 1. Feature Selection
    # Selected features: Pclass, Sex, Age, Fare, Embarked
    # Target: Survived
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    target = 'Survived'
    
    X = df[features]
    y = df[target]
    
    # 2. Preprocessing
    # Numeric features: Age, Fare
    numeric_features = ['Age', 'Fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features: Pclass, Sex, Embarked
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Model Definition
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train
    print("Training model...")
    clf.fit(X_train, y_train)

    # 5. Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 6. Save Model
    model_path = 'model/titanic_survival_model.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    # 7. Demonstrate Reload
    print("\nVerifying reload...")
    loaded_model = joblib.load(model_path)
    if hasattr(loaded_model, 'predict'):
        print("Model reloaded successfully.")
        # Test prediction with a sample
        sample = pd.DataFrame([{
            'Pclass': 3,
            'Sex': 'male',
            'Age': 22.0,
            'Fare': 7.25,
            'Embarked': 'S'
        }])
        prediction = loaded_model.predict(sample)
        print(f"Test Prediction for sample: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")

if __name__ == "__main__":
    train_model()

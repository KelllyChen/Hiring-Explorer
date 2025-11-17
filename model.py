import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def train_model(df):
    y =df['HiringDecision']
    X =  df.drop('HiringDecision',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train simple RF model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv("recruitment_data.csv")
    model, X_train, X_test, y_test = train_model(df)

    # save model
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save training features for SHAP explainer later
    with open("X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    print("Model and training data saved.")
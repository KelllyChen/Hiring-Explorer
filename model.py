import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os   

def train_model(df):
    y =df['HiringDecision']
    X =  df.drop('HiringDecision',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train simple RF model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv("data/recruitment_data.csv")
    model, X_train, X_test, y_test = train_model(df)

    print(classification_report(y_test, model.predict(X_test)))

    os.makedirs("data", exist_ok=True)

    # save model
    with open("data/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save training features for SHAP explainer later
    with open("data/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open("data/X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)

    with open("data/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    print("Model and data saved.")
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/processed/cleaned_heart_failure.csv")
MODEL_PATH = Path("models/best_model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_model_name = None
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{name}")
        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1-score :", round(f1, 4))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = pipe

    print(f"\nBest model based on F1-score: {best_model_name}")

    joblib.dump(best_model, MODEL_PATH)
    print("Saved best model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
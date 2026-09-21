import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = os.path.join("dataset", "sensor_training_data.csv")
MODEL_PATH = os.path.join("models", "aquatwin_rf_model.joblib")

FEATURES = ["temp_c", "humidity_pct", "water_level_pct", "delta_level", "delta_humidity"]
LABEL = "label"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(df)

    X = df[FEATURES]
    y = df[LABEL]

    # Dataset is intentionally tiny (demo dummy data), so we train on the
    # full set and hold out a small slice just to show an evaluation step.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out split:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy on held-out split: {accuracy_score(y_test, y_pred):.2f}")

    print("\nFeature importances:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:16s} {imp:.3f}")

    # Retrain on the FULL dataset before saving, so the demo model has seen
    # every example (fine for a small illustrative dataset like this one).
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

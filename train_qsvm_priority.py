import joblib
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


DATASET_PATH = "survivor_priority_dataset.csv"
MODEL_PATH = "priority_qsvm.pkl"
SCALER_PATH = "priority_scaler.pkl"

FEATURE_COLUMNS = [
    "distance_from_camera",
    "body_visibility_pct",
    "movement_detected",
    "nearby_hazards",
    "time_since_last_movement",
]

PRIORITY_NAMES = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def main():
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["priority"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_map = ZZFeatureMap(
        feature_dimension=len(FEATURE_COLUMNS),
        reps=2,
    )

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)

    print("Training QSVM survivor priority model...", flush=True)
    qsvc.fit(X_train, y_train)

    accuracy = qsvc.score(X_test, y_test)
    predictions = qsvc.predict(X_test)

    print("Accuracy:", round(accuracy, 4), flush=True)
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[PRIORITY_NAMES[i] for i in [0, 1, 2]],
            zero_division=0,
        )
    )

    joblib.dump(qsvc, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Saved:", MODEL_PATH, flush=True)
    print("Saved:", SCALER_PATH, flush=True)


if __name__ == "__main__":
    main()

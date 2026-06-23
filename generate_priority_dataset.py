import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "distance_from_camera",
    "body_visibility_pct",
    "movement_detected",
    "nearby_hazards",
    "time_since_last_movement",
]

PRIORITY_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def assign_priority(row):
    distance = row["distance_from_camera"]
    visibility = row["body_visibility_pct"]
    moving = row["movement_detected"]
    hazard = row["nearby_hazards"]
    still_time = row["time_since_last_movement"]

    risk_score = (
        (0.38 * distance)
        + (0.42 * (1.0 - visibility))
        + (0.48 * (1 - moving))
        + (0.62 * hazard)
        + (0.58 * still_time)
    )

    if risk_score >= 1.0:
        return 2

    if risk_score >= 0.55:
        return 1

    return 0


def main():
    np.random.seed(42)

    rows = []

    for _ in range(120):
        distance_from_camera = np.random.beta(2.0, 2.5)
        body_visibility_pct = np.random.beta(2.8, 1.7)
        movement_detected = np.random.choice([0, 1], p=[0.38, 0.62])
        nearby_hazards = np.random.choice([0, 1], p=[0.78, 0.22])

        if movement_detected:
            time_since_last_movement = np.random.beta(1.0, 8.0)
        else:
            time_since_last_movement = np.random.beta(2.5, 2.0)

        row = {
            "distance_from_camera": distance_from_camera,
            "body_visibility_pct": body_visibility_pct,
            "movement_detected": movement_detected,
            "nearby_hazards": nearby_hazards,
            "time_since_last_movement": time_since_last_movement,
        }
        row["priority"] = assign_priority(row)
        row["priority_name"] = PRIORITY_LABELS[row["priority"]]
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["priority", "priority_name"])
    df.to_csv("survivor_priority_dataset.csv", index=False)

    print(df.head())
    print("Dataset saved: survivor_priority_dataset.csv")
    print(df["priority_name"].value_counts())


if __name__ == "__main__":
    main()

from ultralytics import YOLO


DATASET_YAML = "datasets/fire_smoke_yolov8/data.yaml"
BASE_MODEL = "yolov8m.pt"
PROJECT_DIR = "runs/fire_smoke"
RUN_NAME = "yolov8m_fire_smoke"


def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATASET_YAML,
        epochs=3,
        imgsz=320,
        batch=2,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=2,
        workers=0,
    )
    model.val(
        data=DATASET_YAML,
        imgsz=320,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_val",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

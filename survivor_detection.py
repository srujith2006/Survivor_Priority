import argparse
import csv
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from ultralytics import YOLO


MODEL_PATH = "yolov8m.pt"
HAZARD_MODEL_PATH = "runs/detect/runs/fire_smoke/yolov8m_fire_smoke/weights/best.pt"
PRIORITY_MODEL_PATH = "priority_qsvm.pkl"
PRIORITY_SCALER_PATH = "priority_scaler.pkl"

MOBILE_IP = "192.168.1.103:8080"
OUTPUT_PATH = "survivor_detection_output.mp4"
REPORT_DIR = "reports"

CONFIDENCE_THRESHOLD = 0.45
HAZARD_CONFIDENCE_THRESHOLD = 0.95
HAZARD_OVERLAP_THRESHOLD = 0.18
GPS_INTERVAL_SECONDS = 1
MOTION_PIXELS_THRESHOLD = 0.018
TRACK_DISTANCE_THRESHOLD = 90
STILL_TIME_CAP_SECONDS = 30

PERSON_CLASS_ID = 0

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

PRIORITY_COLORS = {
    0: (40, 190, 70),
    1: (0, 215, 255),
    2: (0, 0, 255),
}

HAZARD_COLORS = {
    "fire": (0, 80, 255),
    "smoke": (180, 180, 180),
}

REPORT_PRIORITY_RANK = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


class SurvivorTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, center, movement_detected, now):
        best_id = None
        best_distance = TRACK_DISTANCE_THRESHOLD

        for track_id, track in self.tracks.items():
            distance = np.linalg.norm(np.array(center) - np.array(track["center"]))
            if distance < best_distance:
                best_id = track_id
                best_distance = distance

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
            self.tracks[best_id] = {
                "center": center,
                "last_seen": now,
                "last_movement": now,
            }

        track = self.tracks[best_id]
        track["center"] = center
        track["last_seen"] = now

        if movement_detected:
            track["last_movement"] = now

        self._remove_stale(now)
        return best_id, now - track["last_movement"]

    def _remove_stale(self, now):
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > 5
        ]

        for track_id in stale_ids:
            del self.tracks[track_id]


class SurvivorReportWriter:
    def __init__(self, report_dir):
        self.report_dir = Path(report_dir)
        self.image_dir = self.report_dir / "images"
        self.events = {}
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def maybe_record(
        self,
        frame,
        box,
        track_id,
        priority,
        features,
        hazard_name,
        latitude,
        longitude,
        frame_count,
        now,
    ):
        existing = self.events.get(track_id)
        existing_priority = existing["priority"] if existing else -1

        if existing and priority < existing_priority:
            return

        if existing and priority == existing_priority:
            old_area = existing["box_width"] * existing["box_height"]
            new_area = max(box[2] - box[0], 1) * max(box[3] - box[1], 1)
            if new_area <= old_area:
                return

        crop_path = self._save_survivor_image(frame, box, track_id, frame_count)
        priority_name = PRIORITY_NAMES[priority]
        hazard = hazard_name if features["nearby_hazards"] else "clear"

        self.events[track_id] = {
            "survivor_id": track_id,
            "priority": priority,
            "priority_name": priority_name,
            "qml_priority": priority_name,
            "latitude": latitude,
            "longitude": longitude,
            "gps_available": latitude is not None and longitude is not None,
            "map_url": map_link(latitude, longitude),
            "image_path": str(crop_path),
            "frame": frame_count,
            "captured_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "movement": "moving" if features["movement_detected"] else "still",
            "nearby_hazard": hazard,
            "distance_from_camera": round(features["distance_from_camera"], 3),
            "body_visibility_pct": round(features["body_visibility_pct"], 3),
            "time_since_last_movement": round(
                features["time_since_last_movement"], 3
            ),
            "box_width": max(box[2] - box[0], 1),
            "box_height": max(box[3] - box[1], 1),
        }

    def _save_survivor_image(self, frame, box, track_id, frame_count):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame

        image_path = self.image_dir / f"survivor_{track_id}_frame_{frame_count}.jpg"
        cv2.imwrite(str(image_path), crop)
        return image_path

    def write_outputs(self):
        events = self.sorted_events()
        self._write_json(events)
        self._write_csv(events)
        self._write_map(events)
        self._write_report(events)
        return {
            "json": self.report_dir / "survivors.json",
            "csv": self.report_dir / "survivors.csv",
            "map": self.report_dir / "survivor_map.html",
            "report": self.report_dir / "survivor_report.html",
            "count": len(events),
        }

    def sorted_events(self):
        return sorted(
            self.events.values(),
            key=lambda event: (
                -REPORT_PRIORITY_RANK[event["priority_name"]],
                event["survivor_id"],
            ),
        )

    def _write_json(self, events):
        with (self.report_dir / "survivors.json").open("w", encoding="utf-8") as file:
            json.dump(events, file, indent=2)

    def _write_csv(self, events):
        fieldnames = [
            "survivor_id",
            "priority_name",
            "qml_priority",
            "latitude",
            "longitude",
            "gps_available",
            "map_url",
            "image_path",
            "frame",
            "captured_at",
            "movement",
            "nearby_hazard",
            "distance_from_camera",
            "body_visibility_pct",
            "time_since_last_movement",
        ]

        with (self.report_dir / "survivors.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow({field: event.get(field) for field in fieldnames})

    def _write_report(self, events):
        rows = "\n".join(report_row(event, self.report_dir) for event in events)
        if not rows:
            rows = (
                "<tr><td colspan=\"9\">No survivors were recorded in this run.</td></tr>"
            )

        report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QML Survivor Priority Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #5d6970; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d5dde2; padding: 8px; text-align: left; }}
    th {{ background: #edf2f5; }}
    img {{ max-width: 180px; max-height: 140px; object-fit: cover; }}
    .High {{ color: #b00020; font-weight: 700; }}
    .Medium {{ color: #8a5a00; font-weight: 700; }}
    .Low {{ color: #126b2f; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>QML Survivor Priority Report</h1>
  <div class="meta">Generated {html.escape(datetime.now().isoformat(timespec="seconds"))}</div>
  <p><a href="survivor_map.html">Open survivor GPS map</a></p>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Captured Survivor</th>
        <th>QML Priority</th>
        <th>GPS</th>
        <th>Map</th>
        <th>Movement</th>
        <th>Hazard</th>
        <th>Frame</th>
        <th>Captured At</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""

        (self.report_dir / "survivor_report.html").write_text(
            report_html,
            encoding="utf-8",
        )

    def _write_map(self, events):
        points = [
            event
            for event in events
            if event["latitude"] is not None and event["longitude"] is not None
        ]
        points_json = json.dumps(
            [
                {
                    "id": event["survivor_id"],
                    "priority": event["priority_name"],
                    "lat": event["latitude"],
                    "lon": event["longitude"],
                    "image": relative_path(event["image_path"], self.report_dir),
                    "hazard": event["nearby_hazard"],
                    "movement": event["movement"],
                }
                for event in points
            ]
        )
        center_lat = points[0]["latitude"] if points else 0
        center_lon = points[0]["longitude"] if points else 0
        map_note = (
            "GPS markers are shown below."
            if points
            else "No GPS coordinates were available for this run."
        )

        map_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Survivor GPS Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    header {{ padding: 14px 18px; background: #172026; color: white; }}
    #map {{ height: calc(100vh - 82px); min-height: 420px; }}
    .note {{ margin-top: 4px; color: #dce5ea; }}
    .popup img {{ width: 150px; max-height: 120px; object-fit: cover; display: block; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <strong>Survivor GPS Map</strong>
    <div class="note">{html.escape(map_note)}</div>
  </header>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {points_json};
    const map = L.map('map').setView([{center_lat}, {center_lon}], points.length ? 17 : 2);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const colors = {{ High: '#b00020', Medium: '#b87500', Low: '#167c3a' }};
    const bounds = [];
    for (const point of points) {{
      const marker = L.circleMarker([point.lat, point.lon], {{
        radius: 9,
        color: colors[point.priority] || '#255f85',
        fillColor: colors[point.priority] || '#255f85',
        fillOpacity: 0.85
      }}).addTo(map);
      marker.bindPopup(
        `<div class="popup"><strong>Survivor ${{point.id}}</strong><br>` +
        `Priority: ${{point.priority}}<br>` +
        `GPS: ${{point.lat}}, ${{point.lon}}<br>` +
        `Movement: ${{point.movement}}<br>` +
        `Hazard: ${{point.hazard}}<br>` +
        `<img src="${{point.image}}" alt="Survivor ${{point.id}}"></div>`
      );
      bounds.push([point.lat, point.lon]);
    }}
    if (bounds.length > 1) {{
      map.fitBounds(bounds, {{ padding: [40, 40] }});
    }}
  </script>
</body>
</html>
"""

        (self.report_dir / "survivor_map.html").write_text(
            map_html,
            encoding="utf-8",
        )


def mobile_video_urls(mobile_ip):
    return [
        f"http://{mobile_ip}/video",
        f"http://{mobile_ip}/videofeed",
        f"http://{mobile_ip}/mjpegfeed",
    ]


def map_link(latitude, longitude):
    if latitude is None or longitude is None:
        return ""

    return f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=18/{latitude}/{longitude}"


def relative_path(path, base_dir):
    return Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix()


def report_row(event, report_dir):
    image_path = html.escape(relative_path(event["image_path"], report_dir))
    priority_name = html.escape(event["priority_name"])
    gps_text = (
        f"{event['latitude']}, {event['longitude']}"
        if event["gps_available"]
        else "unavailable"
    )
    map_cell = (
        f"<a href=\"{html.escape(event['map_url'])}\">Open map</a>"
        if event["map_url"]
        else "unavailable"
    )

    return f"""<tr>
  <td>{event["survivor_id"]}</td>
  <td><img src="{image_path}" alt="Survivor {event["survivor_id"]}"></td>
  <td class="{priority_name}">{priority_name}</td>
  <td>{html.escape(gps_text)}</td>
  <td>{map_cell}</td>
  <td>{html.escape(event["movement"])}</td>
  <td>{html.escape(event["nearby_hazard"])}</td>
  <td>{event["frame"]}</td>
  <td>{html.escape(event["captured_at"])}</td>
</tr>"""


def get_gps(gps_url, last_lat, last_lon):
    try:
        gps = requests.get(gps_url, timeout=1).json()
        return gps["latitude"], gps["longitude"]
    except Exception:
        return last_lat, last_lon


def open_stream(source, mobile_ip):
    if source == "mobile":
        for url in mobile_video_urls(mobile_ip):
            cap = cv2.VideoCapture(url)

            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print("Connected:", url)
                    return cap

            cap.release()

        print("Unable to connect to mobile video stream.")
        sys.exit(1)

    camera_index = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Unable to open source:", source)
        sys.exit(1)

    return cap


def resolve_hazard_model_path(path):
    requested_path = Path(path)

    if requested_path.exists():
        return requested_path

    candidates = [
        Path("runs/detect/runs/fire_smoke/yolov8m_fire_smoke/weights/best.pt"),
        Path("runs/fire_smoke/yolov8m_fire_smoke/weights/best.pt"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def expanded_box(box, width, height, scale=1.45):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    box_w = (x2 - x1) * scale
    box_h = (y2 - y1) * scale

    return clamp_box(
        (
            int(center_x - box_w / 2),
            int(center_y - box_h / 2),
            int(center_x + box_w / 2),
            int(center_y + box_h / 2),
        ),
        width,
        height,
    )


def estimate_distance_from_camera(box, frame_shape):
    frame_h, _ = frame_shape[:2]
    box_h = max(box[3] - box[1], 1)
    height_ratio = np.clip(box_h / max(frame_h, 1), 0.0, 1.0)
    return float(1.0 - height_ratio)


def estimate_body_visibility(box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_area = max(x2 - x1, 1) * max(y2 - y1, 1)
    frame_area = max(frame_w * frame_h, 1)
    area_score = np.clip(box_area / (frame_area * 0.35), 0.0, 1.0)

    touches_edge = x1 <= 2 or y1 <= 2 or x2 >= frame_w - 3 or y2 >= frame_h - 3
    edge_penalty = 0.7 if touches_edge else 1.0

    return float(area_score * edge_penalty)


def detect_motion(frame_gray, previous_gray, box):
    if previous_gray is None:
        return 1

    x1, y1, x2, y2 = box
    current_roi = frame_gray[y1:y2, x1:x2]
    previous_roi = previous_gray[y1:y2, x1:x2]

    if current_roi.size == 0 or previous_roi.size == 0:
        return 0

    diff = cv2.absdiff(current_roi, previous_roi)
    _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_ratio = cv2.countNonZero(threshold) / float(threshold.size)

    return int(changed_ratio >= MOTION_PIXELS_THRESHOLD)


def box_intersection_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    b_area = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)

    return intersection / float(b_area)


def extract_hazard_detections(hazard_results, frame_shape, min_confidence):
    detections = []
    frame_h, frame_w = frame_shape[:2]

    if hazard_results is None:
        return detections

    for result in hazard_results:
        names = result.names

        for detected_box in result.boxes:
            class_id = int(detected_box.cls[0])
            name = names[class_id]
            confidence = float(detected_box.conf[0])

            if confidence < min_confidence:
                continue

            raw_box = tuple(map(int, detected_box.xyxy[0]))

            detections.append(
                {
                    "name": name,
                    "confidence": confidence,
                    "box": clamp_box(raw_box, frame_w, frame_h),
                }
            )

    return detections


def detect_nearby_hazards_from_yolo(person_region, hazard_detections):
    nearby = [
        detection
        for detection in hazard_detections
        if box_intersection_ratio(person_region, detection["box"])
        >= HAZARD_OVERLAP_THRESHOLD
    ]

    if not nearby:
        return 0, "clear"

    strongest = max(nearby, key=lambda detection: detection["confidence"])
    return 1, strongest["name"]


def detect_nearby_hazards_by_color(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return 0, "clear"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    fire_mask = cv2.inRange(hsv, (0, 80, 120), (35, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 55), (180, 75, 210))

    fire_ratio = cv2.countNonZero(fire_mask) / float(fire_mask.size)
    smoke_ratio = cv2.countNonZero(smoke_mask) / float(smoke_mask.size)

    if fire_ratio > 0.12:
        return 1, "fire"

    if smoke_ratio > 0.72:
        return 1, "smoke"

    return 0, "clear"


def extract_priority_features(
    frame,
    frame_gray,
    previous_gray,
    box,
    still_seconds,
    hazard_detections,
    use_color_fallback,
):
    height, width = frame.shape[:2]
    hazard_region = expanded_box(box, width, height)
    nearby_hazards, hazard_name = detect_nearby_hazards_from_yolo(
        hazard_region,
        hazard_detections,
    )

    if not nearby_hazards and use_color_fallback:
        nearby_hazards, hazard_name = detect_nearby_hazards_by_color(
            frame,
            hazard_region,
        )

    features = {
        "distance_from_camera": estimate_distance_from_camera(box, frame.shape),
        "body_visibility_pct": estimate_body_visibility(box, frame.shape),
        "movement_detected": detect_motion(frame_gray, previous_gray, box),
        "nearby_hazards": nearby_hazards,
        "time_since_last_movement": float(
            np.clip(still_seconds / STILL_TIME_CAP_SECONDS, 0.0, 1.0)
        ),
    }

    return features, hazard_name


def predict_priority(qsvm, scaler, features):
    feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    scaled_features = scaler.transform(feature_frame)
    return int(qsvm.predict(scaled_features)[0])


def draw_hazard_boxes(frame, hazard_detections):
    for detection in hazard_detections:
        x1, y1, x2, y2 = detection["box"]
        name = detection["name"]
        confidence = detection["confidence"]
        color = HAZARD_COLORS.get(name, (255, 255, 255))
        label = f"{name} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_label(frame, box, priority, track_id, features, hazard_name, gps_text):
    x1, y1, x2, y2 = box
    color = PRIORITY_COLORS[priority]
    priority_name = PRIORITY_NAMES[priority]
    movement = "moving" if features["movement_detected"] else "still"
    hazard = hazard_name if features["nearby_hazards"] else "clear"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id} | {priority_name} | {movement} | {hazard}"
    gps_label = f"GPS: {gps_text}"

    for index, text in enumerate([label, gps_label]):
        label_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            2,
        )
        label_y = max(y1 - 12 - (index * 24), label_size[1] + 10)

        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - baseline),
            (x1 + label_size[0], label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO + QSVM survivor priority detection"
    )
    parser.add_argument(
        "--source",
        default="mobile",
        help="Use 'mobile', a webcam index like 0, or a video file path.",
    )
    parser.add_argument(
        "--mobile-ip",
        default=MOBILE_IP,
        help="IP Webcam host and port, for example 192.168.1.103:8080.",
    )
    parser.add_argument(
        "--gps-host",
        default="127.0.0.1",
        help="Host for the GPS server that provides latest coordinates.",
    )
    parser.add_argument(
        "--gps-port",
        type=int,
        default=5000,
        help="Port for the GPS server.",
    )
    parser.add_argument(
        "--gps-path",
        default="latest",
        help="Path for GPS JSON endpoint (for example latest or gps.json).",
    )
    parser.add_argument(
        "--disable-gps",
        action="store_true",
        help="Do not poll the GPS server while detecting survivors.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Annotated output video path.",
    )
    parser.add_argument(
        "--report-dir",
        default=REPORT_DIR,
        help="Directory for survivor images, map, JSON, CSV, and HTML report.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Write output without opening a display window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process until the source ends.",
    )
    parser.add_argument(
        "--hazard-model",
        default=HAZARD_MODEL_PATH,
        help="Trained fire/smoke YOLO model path.",
    )
    parser.add_argument(
        "--hazard-conf",
        type=float,
        default=HAZARD_CONFIDENCE_THRESHOLD,
        help="Minimum confidence for fire/smoke detections.",
    )
    parser.add_argument(
        "--enable-color-fallback",
        action="store_true",
        help="Enable HSV color fallback when no trained hazard box is nearby.",
    )
    parser.add_argument(
        "--show-hazard-boxes",
        action="store_true",
        help="Draw separate fire/smoke boxes for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(MODEL_PATH)
    hazard_model_path = resolve_hazard_model_path(args.hazard_model)
    hazard_model = YOLO(hazard_model_path) if hazard_model_path else None
    qsvm = joblib.load(PRIORITY_MODEL_PATH)
    scaler = joblib.load(PRIORITY_SCALER_PATH)

    if hazard_model is None:
        print("Fire/smoke YOLO model not found.")
    else:
        print("Loaded fire/smoke YOLO model:", hazard_model_path)
        print("Hazard confidence threshold:", args.hazard_conf)

    if args.enable_color_fallback:
        print("HSV color fallback enabled.")

    cap = open_stream(args.source, args.mobile_ip)
    gps_path = args.gps_path.strip("/")
    gps_url = f"http://{args.gps_host}:{args.gps_port}/{gps_path}"

    tracker = SurvivorTracker()
    report_writer = SurvivorReportWriter(args.report_dir)
    previous_gray = None
    out = None
    last_lat = None
    last_lon = None
    last_gps = 0

    print("Running YOLO + QSVM survivor priority detection. Press q to quit.")
    frame_count = 0
    total_survivor_hazards = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (width, height),
            )

        now = time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not args.disable_gps and now - last_gps > GPS_INTERVAL_SECONDS:
            last_lat, last_lon = get_gps(gps_url, last_lat, last_lon)
            last_gps = now

        gps_text = (
            f"{last_lat},{last_lon}"
            if last_lat is not None and last_lon is not None
            else "unavailable"
        )

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        hazard_results = (
            hazard_model(frame, conf=args.hazard_conf, verbose=False)
            if hazard_model
            else None
        )
        hazard_detections = extract_hazard_detections(
            hazard_results,
            frame.shape,
            args.hazard_conf,
        )
        if args.show_hazard_boxes:
            draw_hazard_boxes(frame, hazard_detections)

        for result in results:
            for detected_box in result.boxes:
                class_id = int(detected_box.cls[0])

                if class_id != PERSON_CLASS_ID:
                    continue

                raw_box = tuple(map(int, detected_box.xyxy[0]))
                box = clamp_box(raw_box, frame.shape[1], frame.shape[0])
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                movement_detected = detect_motion(frame_gray, previous_gray, box)
                track_id, still_seconds = tracker.update(center, movement_detected, now)
                features, hazard_name = extract_priority_features(
                    frame,
                    frame_gray,
                    previous_gray,
                    box,
                    still_seconds,
                    hazard_detections,
                    args.enable_color_fallback,
                )
                features["movement_detected"] = movement_detected
                total_survivor_hazards += int(features["nearby_hazards"])

                priority = predict_priority(qsvm, scaler, features)
                draw_label(frame, box, priority, track_id, features, hazard_name, gps_text)
                report_writer.maybe_record(
                    frame,
                    box,
                    track_id,
                    priority,
                    features,
                    hazard_name,
                    last_lat,
                    last_lon,
                    frame_count,
                    now,
                )

        previous_gray = frame_gray
        out.write(frame)

        if not args.no_window:
            cv2.imshow("YOLO + QSVM Survivor Priority", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()

    if out:
        out.release()

    cv2.destroyAllWindows()
    report_outputs = report_writer.write_outputs()
    print("Output saved:", args.output)
    print("Frames processed:", frame_count)
    print("Survivor labels with nearby hazard:", total_survivor_hazards)
    print("Survivors recorded:", report_outputs["count"])
    print("Survivor report saved:", report_outputs["report"])
    print("Survivor map saved:", report_outputs["map"])


if __name__ == "__main__":
    main()

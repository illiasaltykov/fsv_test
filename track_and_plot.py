import cv2
import torch
import pandas as pd
import numpy as np
import folium
import pysrt
import re
import sys



VIDEO_FILE = 'video2.MP4'
SRT_FILE = 'video2.SRT'
OUTPUT_MAP = 'car_paths_map.html'


BRACKET_RE = re.compile(r"\[([^]]+)\]")
FRAME_CNT_RE = re.compile(r"FrameCnt:\s*(\d+)")


def parse_srt_data_block(text: str):
    data = {}
    text_no_html = re.sub(r'<[^>]+>', '', text).strip()
    frame_match = FRAME_CNT_RE.search(text_no_html)
    if not frame_match:
        return None
    data['FrameCnt'] = int(frame_match.group(1))

    matches = BRACKET_RE.findall(text_no_html)
    for match in matches:
        if ':' not in match:
            continue
        parts = match.split(':', 1)
        key = parts[0].strip()
        value_str = parts[1].strip()

        try:
            if key == 'rel_alt':
                alt_parts = value_str.split()
                if len(alt_parts) >= 1: data['rel_alt'] = float(alt_parts[0])
            elif key == 'gb_yaw':
                pose_parts = value_str.split()
                if len(pose_parts) >= 1: data['gb_yaw'] = float(pose_parts[0])
                if len(pose_parts) >= 3: data['gb_pitch'] = float(pose_parts[2])
                if len(pose_parts) >= 5: data['gb_roll'] = float(pose_parts[4])
            elif key not in ['shutter', 'color_md', 'ae_meter_md']:
                data[key] = float(value_str)
        except (ValueError, IndexError):
            pass

    return data


def parse_srt_file(srt_path: str) -> pd.DataFrame:
    print(f"Parsing SRT file: {srt_path}...")
    try:
        subs = pysrt.open(srt_path, encoding='utf-8')
    except Exception:
        try:
            subs = pysrt.open(srt_path, encoding='latin-1')
        except Exception as e_default:
            print(f"Error: Could not open SRT file: {e_default}")
            return pd.DataFrame()

    all_data = [data for sub in subs if (data := parse_srt_data_block(sub.text))]
    if not all_data:
        print("Error: Could not find any valid pose data in SRT file.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data).set_index('FrameCnt').sort_index()
    df = df.dropna(subset=['latitude', 'longitude', 'rel_alt', 'gb_yaw', 'focal_len'])
    print(f"Successfully parsed and filtered {len(df)} frames of pose data.")
    return df



def load_models():
    """
    Loads the YOLOv5 model for car detection.
    """
    print("Loading YOLOv5 model...")
    # 'model' is a function that takes an image and returns detections
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("YOLOv5 model loaded.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This script requires PyTorch and an internet connection to download the model.")
        return None


def detect_and_track_cars(frame, model):
    """
    Simulated function for detection and tracking.

    Returns:
        A list of tuples: [(track_id, x, y), ...]
        where (x, y) is the bottom-center of the car's bounding box.
    """
    # 1. Detect
    results = model(frame)
    df = results.pandas().xyxy[0]
    # Class 2 is 'car' in the COCO dataset
    cars = df[(df['class'] == 2) & (df['confidence'] > 0.5)]

    # 2. Track (Simplified)
    tracked_cars = []
    for i, row in cars.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        x_center = (x1 + x2) / 2
        y_bottom = y2
        track_id = i
        tracked_cars.append((track_id, x_center, y_bottom))

    return tracked_cars



def project_pixel_to_gps(px, py, frame_pose, frame_width, frame_height):
    """
    Projects a pixel coordinate (px, py) to a GPS coordinate (lat, lon).
    """
    try:
        F_MM = frame_pose['focal_len']
        H_M = frame_pose['rel_alt']
        YAW_DEG = frame_pose['gb_yaw']

        SENSOR_WIDTH_MM = 35.0

        fx_px = (F_MM / SENSOR_WIDTH_MM) * frame_width

        cx = frame_width / 2
        cy = frame_height / 2

        # Pixel to Camera Coordinate System
        # (X_cam, Y_cam) is the point on the "image sensor" in pixels
        X_cam = px - cx
        Y_cam = py - cy

        # Camera Coords to Ground Plane (Meters)
        # This assumes a straight-down pitch (-90 deg)
        meters_per_pixel = H_M / fx_px

        # (dx, dy) is the offset in meters in the *camera's* frame
        # (Y is forward, X is right)
        dx_cam_m = X_cam * meters_per_pixel
        dy_cam_m = Y_cam * meters_per_pixel

        # Rotate to World Frame (North/East)
        # Rotate the (dx, dy) vector by the drone's yaw
        yaw_rad = np.deg2rad(YAW_DEG)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        # We map this to North/East based on yaw.
        dNorth = dy_cam_m * cos_yaw - dx_cam_m * sin_yaw
        dEast = dy_cam_m * sin_yaw + dx_cam_m * cos_yaw

        # Convert Meter Offset to Lat/Lon Offset
        lat_offset_deg = dNorth / 111111.0
        lon_offset_deg = dEast / (111111.0 * np.cos(np.deg2rad(frame_pose['latitude'])))

        # Add to Drone's GPS Position
        car_lat = frame_pose['latitude'] + lat_offset_deg
        car_lon = frame_pose['longitude'] + lon_offset_deg

        return (car_lat, car_lon)

    except Exception as e:
        print(f"Warning: Projection failed. {e}")
        return None


def main():
    # 1. Parse SRT data
    drone_data = parse_srt_file(SRT_FILE)
    if drone_data.empty:
        sys.exit("Exiting. Could not parse SRT file.")

    # 2. Load ML Model
    # model = load_models()
    # if model is None:
    #     sys.exit("Exiting. Could not load ML model.")
    print("--- SIMULATION MODE ---")
    print("ML model loading is skipped.")
    print("Car detection will be faked to demonstrate the pipeline.")

    # 3. Open Video
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        sys.exit(f"Error: Could not open video file {VIDEO_FILE}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened: {frame_width}x{frame_height}, {total_frames} frames.")

    # Dictionary to store paths: { "car_1": [(lat, lon), ...], ... }
    car_paths = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Only process every 10th frame to speed things up
        if frame_idx % 10 != 0:
            continue

        # Get the pose data for this *exact* frame
        if frame_idx not in drone_data.index:
            continue  # Skip if we have no pose data

        current_pose = drone_data.loc[frame_idx]

        # This is where the AI models would run ---
        # tracked_cars = detect_and_track_cars(frame, model)
        tracked_cars = [
            (1, frame_width * 0.5, frame_height * 0.6),  # Car 1
            (2, frame_width * 0.7, frame_height * 0.55)  # Car 2
        ]

        # Projection Step
        for track_id, x, y in tracked_cars:
            gps_coord = project_pixel_to_gps(x, y, current_pose, frame_width, frame_height)

            if gps_coord:
                if track_id not in car_paths:
                    car_paths[track_id] = []
                car_paths[track_id].append(gps_coord)

        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    print(f"Video processing complete. Found {len(car_paths)} (simulated) car paths.")

    # 5. Plotting
    if not car_paths:
        print("No car paths were generated. Plotting drone path only.")

    # Get drone path for plotting
    drone_gps_points = drone_data[['latitude', 'longitude']].dropna().values.tolist()

    m = folium.Map(location=drone_gps_points[0], zoom_start=18)

    # Plot drone path
    folium.PolyLine(
        drone_gps_points, color='red', weight=5, opacity=0.7, popup='Drone Path'
    ).add_to(m)

    # Plot car paths
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    for i, (track_id, path) in enumerate(car_paths.items()):
        if len(path) > 1:  # Only plot paths with more than one point
            folium.PolyLine(
                path,
                color=colors[i % len(colors)],
                weight=3,
                opacity=0.8,
                popup=f'Car {track_id}'
            ).add_to(m)

    m.save(OUTPUT_MAP)
    print(f"\nSUCCESS: Map of drone and (simulated) car paths saved to '{OUTPUT_MAP}'")


if __name__ == "__main__":
    main()
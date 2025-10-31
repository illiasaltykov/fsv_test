#  Drone-Based Vehicle Trajectory Tracking

This project analyzes drone footage and the accompanying `.SRT` telemetry file to **detect**, **track**, and **map real-world GPS trajectories** of moving vehicles.

##  Features

* **Telemetry Parsing:** Extracts GPS coordinates, altitude, yaw (heading), and focal length from a custom `.SRT` file for each video frame.  
* **Object Detection:** Uses a **YOLOv5** model to detect vehicles in each video frame (requires `torch`).  
* **Object Tracking:** Applies a basic tracker (simulated in the code, designed for integration with SORT) to assign unique IDs to each vehicle.  
* **Pixel-to-GPS Projection:** Converts 2D pixel coordinates (vehicle positions in the frame) into real-world GPS coordinates `(latitude, longitude)`.  
* **Visualization:** Generates an interactive **HTML map** (`.html`) using **Folium**, displaying both the drone’s trajectory and all tracked vehicle paths.

## ️ How It Works

The processing pipeline consists of the following steps:

1. **Data Parsing:**  
   The script reads the `video2.SRT` file and loads telemetry data into a `pandas.DataFrame`, using `FrameCnt` (frame number) as the index.  
   This enables instant lookup of drone position data for any given frame.  

2. **Initialization:**  
   Loads the `video2.MP4` video using OpenCV and (optionally) loads the YOLOv5 model.  

3. **Frame Processing:**  
   For each video frame:  
   * Retrieves the current frame’s `FrameCnt`.  
   * Finds matching telemetry data (GPS, altitude, yaw) from the DataFrame.  
   * Runs YOLOv5 on the frame to detect bounding boxes for cars.  
   * Passes detections to the tracker to update trajectories.  

4. **Coordinate Projection:**  
   For each tracked vehicle, its 2D pixel position `(x, y)` is converted to GPS coordinates.  
   The `project_pixel_to_gps` function uses the drone’s altitude (`rel_alt`), focal length (`focal_len`), and yaw (`gb_yaw`) to calculate real-world displacements, which are then translated into latitude and longitude offsets.  

5. **Result Saving:**  
   After processing the video, all collected GPS trajectories (both drone and vehicles) are plotted on a Folium map and saved as `car_paths_map.html`.

##  Installation

You’ll need **Python 3.8+** and the following dependencies:

```bash
# Install all required dependencies
pip install opencv-python-headless torch torchvision pandas folium pysrt numpy scipy

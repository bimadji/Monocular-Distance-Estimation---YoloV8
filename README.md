# YOLO-Based Monocular Distance Estimation with Webcam

This project uses **YOLO object detection** and a **single (monocular) camera**—such as a built-in laptop webcam—to estimate the distance between the camera and a detected object in real time.

---

## Features

-  Real-time object detection with YOLO (ONNX model)
-  Distance estimation based on object size in pixels
-  Simple one-time calibration using a known object distance
-  Works with laptop webcams (V4L2 compatible)
-  Automatic saving of calibration (`focal.txt`)
-  Optional logging to CSV for analysis/plotting

---

## Prerequisites

- OpenCV 4.x with CUDA 12 support (optional but recommended)
- ONNX Runtime
- C++17 compiler (e.g., `g++`, `clang++`)
- Webcam (USB or built-in)

---

## Building the Application

1. Make sure you have the prerequisites installed
2. Create a build directory and navigate to it:

```bash
mkdir build
cd build
```

3. Run CMake and build:

```bash
cmake ..
make
```

---

## Running the Application

From the build directory, run:

```bash
./inferenceTest
```

---

## How It Works

This program uses the pinhole camera model for monocular distance estimation.

**Formula:**

```
Distance (cm) = (Real Object Height (cm) × Focal Length (px)) / Object Height in Image (px)
```

- **Real Object Height (cm)**: Must be manually defined in the code based on the object you are detecting.
- **Object Height in Image (px)**: Automatically computed using the height of the bounding box detected by YOLO.
- **Focal Length (px)**: Estimated during calibration and saved to `focal.txt`.

---

## Calibration Guide

You need to calibrate the camera once for each object type or detection setup.

### Steps:

1. **Prepare an object** with known height (e.g., a phone = 15 cm, or a human face = 23 cm).
2. **Update the constant** in `main.cpp`:

   ```cpp
   const float REAL_HEIGHT_CM = 15.0f; // Set to your object's real height
   ```

3. **Place the object** in front of the camera at a known distance (e.g., 50 cm).
4. **Run the program**. Ensure the object is detected (bounding box visible).
5. **Press the `c` key** in the OpenCV window.
6. **Enter the real distance** in cm when prompted (e.g., `50`).

The system will calculate the focal length and store it in `focal.txt`. You only need to do this once unless your camera or object changes.

---

## Customization

### Change Object Type (e.g., Phone, Face, Bottle)

Update this line in your code:

```cpp
const float REAL_HEIGHT_CM = 15.0f; // Example: phone
```

> Use a height that matches your object in the real world, in centimeters.

### Change Detection Model

Make sure to update your model path in:

```cpp
Inference inf("your_model.onnx", Size(640, 640), "classes.txt", true);
```

Ensure your YOLO model is trained to detect the object you are trying to measure.

---

## Notes

- Calibration is needed only once unless the camera or object type changes.
- Ensure the whole object fits inside the frame during detection.
- The camera must remain stationary during usage.

---

## Contributor

**Bima Adji Kusuma**  
 bimaadjikusuma@gmail.com

---

## Further Development Ideas

You can expand this project with:

-  Multi-object tracking with individual distance estimation
-  Real-time distance plotting
-  Calibration profiles per object type
-  GUI using Qt or ImGui
-  3D pose estimation from bounding boxes

---

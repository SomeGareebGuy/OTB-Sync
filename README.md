# OTB Chess Vision Builder - Project Overview

## Project Goal

Build a foundational computer vision system to detect a physical chessboard from an IP Webcam stream, calibrate its geometry, generate grid coordinates, and prepare for future move detection.

**Scope**: Steps 1-8 focus on DETECTION, CALIBRATION, and GRID GENERATION only.

**Out of Scope**: ML, piece recognition, move prediction, chess engines

---

## Architecture Overview

```bash
┌─────────────────────────────────────────────────────────────┐
│                     Chess Vision System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: IP Webcam                                          │ 
│  ├─ Phone camera → WiFi stream                              │
│  ├─ HTTP MJPEG format                                       │
│  └─ Real-time frame capture (15-30 FPS)                     │
│      └─ OUTPUT: Live video feed                             │
│                                                             │
│  Step 2: Chessboard Detection (Next)                        │
│  ├─ Edge detection (Canny)                                  │
│  ├─ Contour finding                                         │
│  ├─ Corner detection (largest quadrilateral)                │
│  └─ OUTPUT: 4 corner coordinates (pixel space)              │
│                                                             │
│  Step 3: Perspective Calibration                            │
│  ├─ Homography matrix calculation                           │
│  ├─ Perspective warp to top-down view                       │
│  ├─ Manual corner adjustment tool                           │
│  └─ OUTPUT: Warp transformation matrix (saved)              │
│                                                             │
│  Step 4: Grid Generation                                    │
│  ├─ Divide 64 squares (8x8)                                 │
│  ├─ Pixel → Board coordinate mapping                        │
│  ├─ Grid visualization overlay                              │
│  └─ OUTPUT: Grid coordinates JSON                           │
│                                                             │
│  Step 5-8: Future (Not in this phase)                       │
│  ├─ Occupancy detection                                     │
│  ├─ Board state tracking                                    │
│  ├─ Configuration persistence                               │
│  └─ Next phase foundation                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
| ----------- | ----------- | --------- |
| **Streaming** | IP Webcam (Android app) | Real-time camera over WiFi |
| **Video I/O** | OpenCV (cv2) | Frame capture and display |
| **Image Processing** | OpenCV + NumPy | Edge detection, contours, transforms |
| **Mathematics** | NumPy | Matrix operations, homography |
| **Configuration** | JSON files | Persistent calibration storage |
| **Language** | Python 3.7+ | All processing |

**NOT USED**: TensorFlow, PyTorch, YOLO, or any deep learning

---

## File Structure

```bash
OTB-Sync/
│
├── main.py                          # Step 1: IP Webcam connection
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── STEP1_SETUP.md                   # Step 1 setup instructions
├── TROUBLESHOOTING_STEP1.md         # Step 1 troubleshooting guide
├── STEP2_SETUP.md                   # Step 2 detection guide
├── STEP2_GETTING_STARTED.md         # Step 2 quick start guide
├── STEP2_COMPLETE.md                # Step 2 complete reference
├── STEP2_QUICK_REFERENCE.md         # Step 2 quick reference
├── TROUBLESHOOTING_STEP2.md         # Step 2 troubleshooting guide
├── TUNING_GUIDE.md                  # Step 2 parameter tuning
├── STEP3_SETUP.md                   # Step 3 calibration guide
├── STEP3_GETTING_STARTED.md         # Step 3 quick start guide
├── STEP3_COMPLETE.md                # Step 3 complete reference
├── TROUBLESHOOTING_STEP3.md        # Step 3 troubleshooting guide
│
├── calibration_config.json          # (Created) Saved camera config
│
├── screenshot_*.jpg                 # (Created) Captured frames
```

---

## Step-by-Step Milestones

### Step 1: IP Webcam Connection (CURRENT)

**Objective**: Establish real-time connection to phone camera

**Deliverables**:

- [x] IPWebcamReader class with frame capture
- [x] Real-time video display window
- [x] FPS monitoring
- [x] Configuration persistence (JSON)
- [x] User-friendly interface with controls
- [x] Comprehensive troubleshooting guide

**Success Criteria**:

- [ ] Script runs without errors
- [ ] Live video displays in window
- [ ] FPS counter shows 15+ frames/sec
- [ ] Can save screenshots with 's' key
- [ ] Can save config with 'c' key (generates `calibration_config.json`)

**Time Estimate**: 15-30 minutes (setup + testing)

---

### Step 2: Chessboard Detection (Next)

**Objective**: Automatically detect board corners in video frames

**Concepts**:

- Canny edge detection (finds strong intensity changes)
- Contour finding (traces object boundaries)
- Quadrilateral detection (finds largest 4-sided shape)
- Reliability filtering (handles lighting changes)

**Deliverables**:

- Edge detection with adaptive thresholding
- Contour analysis and filtering
- Corner detection (largest quadrilateral)
- Visualization (draw corners on frame)
- Persistence across frames (smooth tracking)

**Output**: 4 corner coordinates (x,y pixels) of detected board

**Time Estimate**: 30-45 minutes

---

### Step 3: Perspective Calibration

**Objective**: Transform board from camera angle to top-down view

**Concepts**:

- Homography matrix (perspective transformation math)
- cv2.getPerspectiveTransform (calculates warp matrix)
- cv2.warpPerspective (applies transformation)
- Manual corner refinement (UI for adjustment)

**Deliverables**:

- Homography matrix calculation
- Perspective warp implementation
- Live warped board preview
- Manual calibration interface
- Calibration persistence (save transformation)

**Output**: Top-down board image + transformation matrix

**Time Estimate**: 30-40 minutes

---

### Step 4: Grid Generation

**Objective**: Create 64 coordinate mappings (8×8 board)

**Concepts**:

- Dividing warped board into equal regions
- Pixel → board coordinate mapping
- Grid visualization
- Square identification (a1-h8 notation)

**Deliverables**:

- Grid lines overlay
- Coordinate mapping (pixel to square)
- Grid visualization in live feed
- Calibration export (grid_config.json)

**Output**: Complete grid coordinate system

**Time Estimate**: 20-30 minutes

---

### Steps 5-8: Occupancy & Persistence (Future)

Will build on Steps 1-4 foundations for future move detection

---

## How to Proceed

### RIGHT NOW (Step 1)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare phone
# - Install IP Webcam app
# - Start server
# - Note IP address

# 3. Run the script
python main.py

# 4. Enter phone IP when prompted
# 5. Watch live video
# 6. Press 'c' to save config
# 7. Press 'q' to quit
```

### AFTER Step 1

Once you confirm live video working, we'll build Step 2 (chessboard detection) in the same `main.py` file, adding the board detection logic on top of the video stream.

---

## Key Design Principles

### 1. **Modular Architecture**

Each step is self-contained but builds on previous:

```bash
Frame Capture (Step 1)
    ↓
Board Detection (Step 2)
    ↓
Perspective Warping (Step 3)
    ↓
Grid Generation (Step 4)
```

### 2. **Configuration Persistence**

All calibration data saved to JSON:

```json
{
  "ip_address": "192.168.1.100",
  "resolution": {"width": 1280, "height": 720},
  "board_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "homography_matrix": [...],
  "grid_coordinates": {...}
}
```

Each step reads + appends to this file.

### 3. **Visual Debugging**

Every step displays:

- Live video with annotations
- Detected features (corners, contours, grid)
- Debug information (coordinates, matrices, FPS)

**Why?** You can see exactly what's working/broken at each stage.

### 4. **Robustness**

- Handles lighting variations
- Continues detection even with partial board visibility
- Graceful error recovery
- User can manually adjust calibration

---

## OpenCV Concepts You'll Learn

| Concept | Used In | Purpose |
| --------- | --------- | --------- |
| **cv2.VideoCapture** | Step 1 | Stream video from IP Webcam |
| **cv2.Canny** | Step 2 | Edge detection |
| **cv2.findContours** | Step 2 | Find object boundaries |
| **cv2.approxPolyDP** | Step 2 | Simplify contours to corners |
| **cv2.goodFeaturesToTrack** | Step 2 (alt) | Corner detection |
| **cv2.getPerspectiveTransform** | Step 3 | Calculate perspective matrix |
| **cv2.warpPerspective** | Step 3 | Apply perspective transform |
| **cv2.drawContours** | All | Visualization |
| **cv2.line, cv2.circle** | All | Draw grid/points |

No ML frameworks needed for any of these!

---

## Common Pitfalls to Avoid

1. Trying ML (YOLO, etc.) for Step 2
   - Use contour detection instead

2. Complex piece recognition logic
   - Focus only on board geometry

3. Skipping calibration persistence
   - Save all parameters to JSON

4. No visualization during development
   - Always show debug windows

5. Hardcoding coordinates
   - Use configuration files

---

## Success Definition

You'll know this project is successful when:

 **Step 1 Complete**: Video stream displays in real-time with FPS > 15
 **Step 2 Complete**: Board corners automatically detected and highlighted
 **Step 3 Complete**: Board appears rectangular in transformed view
 **Step 4 Complete**: 64 grid squares visualized with coordinates
 **Step 5+**: Foundation ready for occupancy/move detection

---

## Quick Reference: OpenCV + NumPy Syntax

```python
# Reading frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame is numpy array: (height, width, 3) - BGR color

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding polygon
eps = 0.02 * cv2.arcLength(contour, True)
corners = cv2.approxPolyDP(contour, eps, True)  # corners is shape (4, 1, 2)

# Perspective transform
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(frame, M, (width, height))

# Drawing
cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
cv2.imshow("window_name", frame)
```

---

## Debugging Workflow

When something doesn't work:

1. **Check data types**

   ```python
   print(f"Type: {type(var)}, Shape: {var.shape}, Dtype: {var.dtype}")
   ```

2. **Visualize intermediate steps**

   ```python
   cv2.imshow("step_name", image)
   cv2.waitKey(0)
   ```

3. **Print key values**

   ```python
   print(f"Corners: {corners}")
   print(f"FPS: {fps}")
   ```

4. **Check ranges**

   ```python
   print(f"Min: {image.min()}, Max: {image.max()}")
   ```

---

## Next Actions

1. Install dependencies: `pip install -r requirements.txt`
2. Set up phone and IP Webcam app
3. Run `main.py` and verify live video
4. Save configuration with 'c' key
5. Proceed to Step 2: Chessboard Detection

Once Step 1 is complete, I'll build Step 2 on top of this foundation.

---

## Resources

### IP Webcam Documentation

- App: <https://play.google.com/store/apps/details?id=com.pas.webcam>
- FAQ: <http://192.168.X.X:8080> (visit the web interface)

### OpenCV Documentation

- Main: <https://docs.opencv.org/>
- Python: <https://docs.opencv.org/4.8.1/d6/d00/tutorial_py_root.html>
- Video I/O: <https://docs.opencv.org/4.8.1/dd/d43/tutorial_py_video_display.html>

### NumPy (for coordinate math)

- Official: <https://numpy.org/doc/stable/>

---

**Status**: Step 1 code ready. Waiting for your setup and test results.

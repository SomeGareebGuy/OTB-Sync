# Step 1 Quick Start Guide

## 5-Minute Setup

### On Your Android Phone

1. **Install IP Webcam**
   - Open Play Store → Search "IP Webcam" → Install

2. **Start the Server**
   - Open IP Webcam app
   - Tap "Start server" button
   - **Note the IP address** displayed (e.g., `192.168.1.100`)
   - Keep app open

3. **Verify Connection**
   - Open browser on laptop
   - Visit: `http://192.168.1.100:8080`
   - Should see web interface with live preview

### On Your Laptop

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python main.py

# 3. When prompted, enter phone IP
# Example: 192.168.1.100

# 4. You should see live video with FPS counter

# 5. Controls:
#    q = quit
#    s = save screenshot
#    c = save config
```

---

## How to Know It's Working

You'll see:

- Live video from phone camera in a window
- FPS counter in top-left (15-30 FPS)
- Frame counter updating
- Console shows: `[+] Connected successfully!`
- Console shows: `[+] Frame resolution: WIDTHxHEIGHT`

---

## Most Common Issues

### "Connection error"

- Check phone IP address (copy exactly from app)
- Verify both on same WiFi network
- Try in browser first: `http://192.168.1.100:8080`

### "Connected but no frames"

- Restart IP Webcam app (Stop → Start server)
- Wait 3 seconds then try again

### Very low FPS (< 5)

- Move closer to WiFi router
- Reduce video quality in IP Webcam settings
- Close other apps using WiFi

---

## Files Created After Success

After pressing 'c' in the video window:

```bash
calibration_config.json     # Camera configuration (saves IP + resolution)
screenshot_1.jpg            # Frame captures (after pressing 's')
screenshot_2.jpg
...
```

---

## Troubleshooting Checklist

```bash
☐ IP Webcam installed on phone
☐ IP Webcam server running (showing "Stop server" button)
☐ Phone and laptop on same WiFi
☐ Can visit http://192.168.X.X:8080 in browser
☐ Python dependencies installed (pip install -r requirements.txt)
☐ Running: python main.py
☐ Entering correct IP address when prompted
```

If still stuck, see **TROUBLESHOOTING_STEP1.md** for detailed diagnostics.

---

## What's Happening (Technical)

1. **IP Webcam app** runs HTTP server on phone
2. **OpenCV** connects to `http://192.168.1.X:8080/video`
3. **Frames stream** as MJPEG (Motion JPEG) over WiFi
4. **Python reads** frames in a loop (15-30 FPS)
5. **Display window** shows video in real-time

The connection is over WiFi - no cables needed!

---

## Next Steps (After Step 1)

Once live video is working:

1. Confirm FPS > 15
2. Save configuration with 'c' key
3. Ready for Step 2: Chessboard Detection
   - We'll add board corner detection
   - Build on this same framework

---

## Camera Positioning Tips

For chessboard detection (later steps):

- **Overhead angle**: Phone mounted above board (tripod recommended)
- **Board filling frame**: Board should fill most of the video area
- **Good lighting**: Avoid shadows or extreme glare
- **Stable**: Use tripod so camera doesn't move

For now, just get video working - positioning can be adjusted later.

## If Everything Fails

1. **Restart both devices**
   - Phone: Power off → on
   - Laptop: WiFi off → on

2. **Restart WiFi**
   - Phone: Forget WiFi → reconnect
   - Laptop: Disconnect → reconnect
   - May get new IP address

3. **Reinstall IP Webcam**
   - Uninstall app
   - Reinstall from Play Store
   - Start fresh

4. **Check with browser first**
   - Always test `http://192.168.X.X:8080` in browser before Python
   - If browser works, network is fine (problem is Python setup)
   - If browser fails, network problem (fix WiFi connection)

5. **Full diagnostic**
   - See TROUBLESHOOTING_STEP1.md for detailed flowchart

---

## Expected Performance

| Metric | Expected | Poor | Critical |
| -------- | ---------- | ------ | ---------- |
| **FPS** | 20-30 | 5-15 | < 5 |
| **Latency** | < 500ms | 500-2000ms | > 2000ms |
| **Resolution** | 1280×720+ | 640×480 | < 640×480 |
| **Connection** | Stable | Occasional drops | Frequent drops |

Anything in "Expected" column = you're good for Step 2!

## Configuration Saved

When you press 'c', this file is created:

**calibration_config.json:**

```json
{
  "ip_address": "192.168.1.100",
  "port": 8080,
  "stream_url": "http://192.168.1.100:8080/video",
  "resolution": {
    "width": 1280,
    "height": 720
  }
}
```

Next time you run the script, it will ask to reuse this config!

---

## Success Criteria

Step 1 is **DONE** when:

- [x] Script runs without errors
- [x] Live video displays in window
- [x] FPS counter shows 15+ FPS
- [x] Can save screenshots (press 's')
- [x] Can save config (press 'c')
- [x] calibration_config.json created

---

## Code Structure Overview

```bash
main.py
├── IPWebcamReader
│   ├── connect()           # Establish connection
│   ├── read_frame()        # Get next frame
│   ├── disconnect()        # Clean up
│   └── get_fps()           # Return FPS
│
├── CalibrationManager
│   ├── load_config()       # Load from JSON
│   └── save_config()       # Save to JSON
│
└── display_live_feed()
    └── Main loop
        ├── Read frame
        ├── Draw FPS/info
        ├── Display window
        └── Handle keys (q, s, c)
```

Simple, modular, reusable for Step 2!

---

Ready? Let's go! Run: `python main.py`

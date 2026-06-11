import cv2
import time
import json
import os
from pathlib import Path
import threading

import numpy as np
from board_detection import (
    BoardDetector,
    draw_board_corners,
    draw_debug_images,
    warp_board,
    generate_board_grid,
    draw_board_grid,
    safe_warp_board,
    compute_square_intensity_map,
    compute_square_average_intensity_map,
    compare_intensity_maps,
    draw_occupancy_map,
    TUNING_PROFILES
)

class IPWebcamReader:
    """
    Manages connection to IP Webcam stream and frame capture.

    Uses a background thread to continuously read frames from the HTTP MJPEG
    stream so the main loop remains responsive to `cv2.waitKey` and user input.
    """

    def __init__(self, ip_address: str, port: int = 8080):
        self.ip_address = ip_address
        self.port = port
        self.stream_url = f"http://{ip_address}:{port}/video"
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

        # Threading/state for non-blocking frame reads
        self._thread = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._last_ret = False

    def connect(self) -> bool:
        """
        Establish connection and start background reader thread.
        """
        print(f"[*] Connecting to {self.stream_url}...")

        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify the stream with a single blocking read
            ret, frame = self.cap.read()
            if not ret:
                print("[!] Failed to read first frame")
                if self.cap is not None:
                    self.cap.release()
                return False

            # Store the initial frame
            with self._frame_lock:
                self._last_frame = frame
                self._last_ret = True

            print(f"[+] Connected successfully!")
            print(f"[+] Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
            self.frame_count = 0
            self.last_time = time.time()

            # Start background reader
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

            return True

        except Exception as e:
            print(f"[!] Connection error: {e}")
            return False

    def _reader_loop(self):
        """Continuously read frames from the capture in a background thread."""
        while self._running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # Small sleep to avoid busy loop on intermittent network failures
                    time.sleep(0.01)
                    continue

                with self._frame_lock:
                    # Store latest frame for non-blocking access
                    self._last_frame = frame
                    self._last_ret = True
                    self.frame_count += 1

                # Update FPS periodically
                self._update_fps()

            except Exception as e:
                print(f"[!] Reader thread error: {e}")
                time.sleep(0.5)

    def read_frame(self):
        """
        Non-blocking retrieval of the latest frame read by the background thread.

        Returns:
            tuple: (success: bool, frame: numpy.ndarray)
        """
        if not self._running:
            return False, None

        with self._frame_lock:
            if not self._last_ret or self._last_frame is None:
                return False, None
            # Return a reference to the last frame; copy if you need isolation
            return True, self._last_frame

    def _update_fps(self):
        """Calculate frames per second (thread-safe usage expected)."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

    def disconnect(self):
        """Stop background reader and release camera resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        print("[+] Disconnected from webcam")

    def get_fps(self) -> float:
        return self.fps


class CalibrationManager:
    """
    Manages persistent configuration storage for camera/board setup.
    
    Stores:
    - IP Webcam address and port
    - Board corner coordinates
    - Camera calibration parameters
    
    Future: Will store perspective transform matrices and grid calibration
    """
    
    CONFIG_FILE = "calibration_config.json"
    
    @staticmethod
    def load_config() -> dict:
        """Load configuration from file."""
        if os.path.exists(CalibrationManager.CONFIG_FILE):
            try:
                with open(CalibrationManager.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                print(f"[+] Loaded configuration from {CalibrationManager.CONFIG_FILE}")
                return config
            except Exception as e:
                print(f"[!] Error loading config: {e}")
        
        return {}
    
    @staticmethod
    def save_config(config: dict):
        """Save configuration to file."""
        try:
            with open(CalibrationManager.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[+] Saved configuration to {CalibrationManager.CONFIG_FILE}")
        except Exception as e:
            print(f"[!] Error saving config: {e}")


def display_live_feed(
    ip_address: str,
    duration: int = 0,
    saved_corners: np.ndarray | None = None,
    saved_occupancy_baseline: list[float] | None = None,
) -> bool:
    """
    Display live feed from IP Webcam with optional chessboard detection.
    
    Args:
        ip_address (str): Phone's IP address
        duration (int): How long to display (seconds, 0 = unlimited)
    
    Returns:
        bool: True if successful, False if failed
    
    Controls:
        - Press 'q' to quit
        - Press 's' to take a screenshot
        - Press 'c' to save current camera config
        - Press 'b' to toggle board detection ON/OFF
        - Press 'd' to toggle debug visualizations (edges, contours, etc.)
        - Press 'p' to print current corner coordinates
        - Press '+' to increase Canny low threshold
        - Press '-' to decrease Canny low threshold
        - Press '>' to increase Canny high threshold
        - Press '<' to decrease Canny high threshold
    """
    
    # Initialize reader
    reader = IPWebcamReader(ip_address)
    
    if not reader.connect():
        print("[!] Failed to connect to IP Webcam")
        return False
    
    # Initialize board detector
    detector = BoardDetector(
        canny_low=50,
        canny_high=150,
        min_contour_area=0.1,
        epsilon_factor=0.02,
        debug_mode=False
    )

    if saved_corners is not None:
        detector.last_corners = saved_corners.copy()
        print("[*] Using saved board corners as fallback calibration")

    if saved_occupancy_baseline is not None:
        occupancy_baseline = saved_occupancy_baseline.copy()
        print("[*] Loaded saved occupancy baseline from configuration")

    def get_window_property(window_name: str, prop: int):
        try:
            return cv2.getWindowProperty(window_name, prop)
        except cv2.error:
            return None

    def is_window_open(window_name: str) -> bool:
        prop = get_window_property(window_name, cv2.WND_PROP_VISIBLE)
        return prop is not None and prop >= 1

    def safe_destroy_window(window_name: str) -> None:
        if is_window_open(window_name):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    print("\n[*] Live feed started")
    print("[*] Controls:")
    print("    'q'   = quit")
    print("    's'   = screenshot")
    print("    'c'   = save config")
    print("    'b'   = toggle board detection")
    print("    'd'   = toggle debug visualizations")
    print("    'p'   = print corner coordinates")
    print("    '+'   = increase Canny low threshold")
    print("    '-'   = decrease Canny low threshold")
    print("    '>'   = increase Canny high threshold")
    print("    '<'   = decrease Canny high threshold")
    print("    'w'   = toggle warped top-down view")
    print("    'g'   = toggle grid overlay on warped view")
    print("    'l'   = toggle square labels on warped grid")
    print("    'x'   = increase warp padding")
    print("    'z'   = decrease warp padding")
    print("    'v'   = increase vertical zoom")
    print("    'n'   = decrease vertical zoom")
    print("    'i'   = increase diagonal zoom")
    print("    'u'   = decrease diagonal zoom")
    print("    'f'   = toggle lock/freeze detected board corners for warp")
    print("    't'   = set occupancy calibration baseline")
    print("    'o'   = toggle occupancy overlay")
    print("    '['   = lower occupancy threshold")
    print("    ']'   = raise occupancy threshold")
    print("[*] Press Ctrl+C to stop\n")
    
    start_time = time.time()
    frame_num = 0
    board_detection_enabled = True
    debug_visualization_enabled = False
    warp_enabled = False
    grid_enabled = False
    show_labels = False
    occupancy_enabled = False
    occupancy_threshold = 18.0
    occupancy_baseline = None
    occupied_squares = []
    last_warped_board = None
    # Move-detection state
    move_baseline = None
    last_grid = None
    last_intensity_map = None
    detected_corners = None
    board_found = False
    warp_size = 800
    warp_padding = 40
    warp_scale_vert = 1.0
    warp_scale_diag = 1.0
    lock_board = False
    locked_corners = None
    warp_window_created = False
    main_window_name = "IP Webcam - Chess Vision Builder"
    
    try:
        while True:
            # Check timeout if specified
            if duration > 0 and (time.time() - start_time) > duration:
                print(f"[*] Timeout after {duration} seconds")
                break
            
            # Read frame
            ret, frame = reader.read_frame()
            
            if not ret or frame is None:
                print("[!] Failed to read frame")
                break
            
            frame_num += 1
            display_frame = frame.copy()

            # ===== BOARD DETECTION STAGE =====
            if board_detection_enabled:
                success, corners, debug_data = detector.detect_board_corners(frame) # type: ignore
                
                if success:
                    board_found = True
                    detected_corners = corners
                    
                    # Draw detected corners on the display frame
                    display_frame = draw_board_corners(display_frame, corners) # type: ignore
                else:
                    board_found = False
                    detected_corners = None

                # Draw debug images if enabled
                if debug_visualization_enabled and debug_data:
                    draw_debug_images(display_frame, debug_data)

            # ===== WARPED BOARD VIEW =====
            if warp_enabled:
                if warp_window_created:
                    warp_window_prop = get_window_property("Warped Board", cv2.WND_PROP_VISIBLE)
                    if warp_window_prop is not None and warp_window_prop < 0:
                        warp_enabled = False
                        warp_window_created = False
                        print("[*] Warped board view disabled after window close")

                if warp_enabled:
                    if warp_window_created:
                        warp_window_prop = get_window_property("Warped Board", cv2.WND_PROP_VISIBLE)
                        if warp_window_prop is not None and warp_window_prop < 0:
                            warp_enabled = False
                            warp_window_created = False
                            print("[*] Warped board view disabled after window close")

                    # Determine source corners: respect locked state if set
                    if lock_board and locked_corners is not None:
                        warp_source_corners = locked_corners
                    else:
                        warp_source_corners = detected_corners if detected_corners is not None else detector.last_corners

                    if warp_source_corners is not None:
                        try:
                            # compute effective scales
                            scale_x = float(warp_scale_diag)
                            scale_y = float(warp_scale_diag * warp_scale_vert)
                            warped, _ = warp_board(frame, warp_source_corners, output_size=warp_size, padding=warp_padding, scale_x=scale_x, scale_y=scale_y)
                        except Exception as e:
                            print(f"[!] Warp failure: {e}")
                            warped = None

                        if warped is not None:
                            # generate grid matching the warped image size and padding
                            out_w = int(round(warp_size * warp_scale_diag))
                            out_h = int(round(warp_size * warp_scale_diag * warp_scale_vert))
                            grid = generate_board_grid(output_size=(out_w, out_h), board_size=8, offset=(warp_padding, warp_padding))

                            # Update last grid and intensity map for move detection
                            try:
                                last_grid = grid
                                last_intensity_map = compute_square_intensity_map(warped, grid)
                            except Exception:
                                last_grid = None
                                last_intensity_map = None

                            if grid_enabled:
                                warped = draw_board_grid(warped, grid, show_labels=show_labels)

                            last_warped_board = warped.copy()

                            if occupancy_enabled:
                                if occupancy_baseline is None:
                                    print("[!] Occupancy baseline not set. Press 't' to calibrate.")
                                else:
                                    current_intensities = compute_square_intensity_map(warped, grid)
                                    occupied_squares = compare_intensity_maps(
                                        occupancy_baseline, current_intensities,
                                        threshold=occupancy_threshold
                                    )
                                    warped = draw_occupancy_map(warped, grid, occupied_squares)

                            cv2.namedWindow("Warped Board", cv2.WINDOW_NORMAL)
                            cv2.imshow("Warped Board", warped) # type: ignore
                            warp_window_created = True
                        else:
                            safe_destroy_window("Warped Board")
                            warp_window_created = False
                    else:
                        print("[!] Warp enabled but no valid corner set yet")
                        safe_destroy_window("Warped Board")
                        warp_window_created = False
            else:
                safe_destroy_window("Warped Board")
                warp_window_created = False

            # ===== OVERLAY TEXT =====
            # FPS display
            fps_text = f"FPS: {reader.get_fps():.1f}"
            cv2.putText(
                display_frame, fps_text, # type: ignore
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            ) # type: ignore
            
            # Frame counter
            frame_text = f"Frame: {frame_num}"
            cv2.putText(
                display_frame, frame_text, # type: ignore
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            ) # pyright: ignore[reportCallIssue]
            
            # Board detection status
            status_text = "Board: FOUND" if board_found else "Board: Not found"
            status_color = (0, 255, 0) if board_found else (0, 0, 255)
            cv2.putText(
                display_frame, status_text, # type: ignore
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            ) # pyright: ignore[reportCallIssue]
            
            # Detection mode indicator
            detection_indicator = "Detection: ON" if board_detection_enabled else "Detection: OFF"
            detection_color = (0, 255, 0) if board_detection_enabled else (0, 165, 255)
            cv2.putText(
                display_frame, detection_indicator, # type: ignore
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, detection_color, 2
            ) # type: ignore
            
            # Debug mode indicator
            if debug_visualization_enabled:
                cv2.putText(
                    display_frame, "Debug: ON", # type: ignore
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                ) # type: ignore

            # Warp status indicator
            warp_indicator = "Warp: ON" if warp_enabled else "Warp: OFF"
            warp_color = (0, 255, 0) if warp_enabled else (0, 165, 255)
            cv2.putText(
                display_frame, warp_indicator, # type: ignore
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, warp_color, 2
            ) # type: ignore

            cv2.putText(
                display_frame, f"Padding: {warp_padding}", # type: ignore
                (10, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            ) # type: ignore

            cv2.putText(
                display_frame, f"V-zoom: {warp_scale_vert:.2f}", # type: ignore
                (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 255), 2
            ) # type: ignore

            cv2.putText(
                display_frame, f"Diag-zoom: {warp_scale_diag:.2f}", # type: ignore
                (10, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 255), 2
            ) # type: ignore

            lock_text = "LOCKED" if lock_board else "UNLOCKED"
            lock_color = (0, 165, 255) if lock_board else (200, 200, 200)
            cv2.putText(
                display_frame, f"Board: {lock_text}", # type: ignore
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, lock_color, 2
            ) # type: ignore

            # Grid status indicator
            if grid_enabled:
                cv2.putText(
                    display_frame, "Grid: ON", # type: ignore
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                ) # type: ignore

            if occupancy_enabled:
                cv2.putText(
                    display_frame, f"Occupancy: ON ({len(occupied_squares)} squares)", # type: ignore
                    (10, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                ) # type: ignore
                cv2.putText(
                    display_frame, f"Threshold: {occupancy_threshold:.1f}", # type: ignore
                    (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 0), 2
                ) # type: ignore

            # Canny threshold display (when adjustable)
            canny_text = f"Canny: {detector.canny_low}-{detector.canny_high}"
            cv2.putText(
                display_frame, canny_text, # type: ignore
                (display_frame.shape[1] - 280, 30), # type: ignore
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 0), 2
            ) # type: ignore
            
            # Display frame
            cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(main_window_name, display_frame) #type: ignore
            
            # Close when main window is manually closed
            main_window_prop = get_window_property(main_window_name, cv2.WND_PROP_VISIBLE)
            if main_window_prop is not None and main_window_prop < 0:
                print("[*] Main window closed by user, exiting")
                break

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[*] Quit requested by user")
                break
            
            elif key == ord('s'):
                # Save screenshot
                filename = f"screenshot_{frame_num}.jpg"
                cv2.imwrite(filename, frame) #type: ignore
                print(f"[+] Screenshot saved: {filename}")

            elif key == ord('x'):
                warp_padding += 5
                print(f"[*] Warp padding: {warp_padding}")

            elif key == ord('z'):
                warp_padding = max(0, warp_padding - 5)
                print(f"[*] Warp padding: {warp_padding}")

            elif key == ord('v'):
                warp_scale_vert = min(3.0, warp_scale_vert + 0.02)
                print(f"[*] Vertical zoom: {warp_scale_vert:.2f}")

            elif key == ord('n'):
                warp_scale_vert = max(0.2, warp_scale_vert - 0.02)
                print(f"[*] Vertical zoom: {warp_scale_vert:.2f}")

            elif key == ord('i'):
                warp_scale_diag = min(3.0, warp_scale_diag + 0.02)
                print(f"[*] Diagonal zoom: {warp_scale_diag:.2f}")

            elif key == ord('u'):
                warp_scale_diag = max(0.2, warp_scale_diag - 0.02)
                print(f"[*] Diagonal zoom: {warp_scale_diag:.2f}")

            elif key == ord('f'):
                # Toggle lock/freeze of detected corners used for warping
                if not lock_board:
                    # Engage lock: capture current detected corners if available
                    if detected_corners is not None:
                        locked_corners = detected_corners.copy()
                        lock_board = True
                        print("[*] Board lock engaged (using current detected corners)")
                    elif detector.last_corners is not None:
                        locked_corners = detector.last_corners.copy()
                        lock_board = True
                        print("[*] Board lock engaged (using last known corners)")
                    else:
                        print("[!] Cannot lock board: no corners available")
                else:
                    lock_board = False
                    locked_corners = None
                    print("[*] Board lock released")
            
            elif key == ord('c'):
                # Save camera configuration
                config = {
                    "ip_address": reader.ip_address,
                    "port": reader.port,
                    "stream_url": reader.stream_url,
                    "resolution": {
                        "width": frame.shape[1], #type: ignore
                        "height": frame.shape[0] #type: ignore
                    }
                }
                
                # Add board corners if detected
                if board_found and detected_corners is not None:
                    config["board_corners"] = detected_corners.tolist()
                    config["board_detected"] = True
                    print("[+] Board corners saved to configuration")
                else:
                    config["board_detected"] = False

                if occupancy_baseline is not None:
                    config["occupancy_baseline"] = occupancy_baseline
                    config["occupancy_threshold"] = occupancy_threshold
                    print("[+] Occupancy baseline saved to configuration")

                CalibrationManager.save_config(config)
                print("[+] Camera config saved")
            
            elif key == ord('b'):
                # Toggle board detection
                board_detection_enabled = not board_detection_enabled
                status = "ENABLED" if board_detection_enabled else "DISABLED"
                print(f"[*] Board detection {status}")
            
            elif key == ord('d'):
                # Toggle debug visualization
                debug_visualization_enabled = not debug_visualization_enabled
                detector.debug_mode = debug_visualization_enabled
                status = "ENABLED" if debug_visualization_enabled else "DISABLED"
                print(f"[*] Debug visualization {status}")
                if not debug_visualization_enabled:
                    # Close debug windows
                    safe_destroy_window("Edges (Canny)")
                    safe_destroy_window("Detected Contours")
                    safe_destroy_window("Grayscale")
            
            elif key == ord('p'):
                # Print corner coordinates
                if board_found and detected_corners is not None:
                    print("\n[*] Detected board corners:")
                    labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                    for i, (corner, label) in enumerate(zip(detected_corners, labels)):
                        print(f"    {label:15s}: ({corner[0]:7.1f}, {corner[1]:7.1f})")
                    print()
                else:
                    print("[!] Board not detected - no corners to print")
            
            elif key == ord('+'):
                # Increase Canny low threshold
                detector.canny_low = min(255, detector.canny_low + 5)
                print(f"[*] Canny low threshold: {detector.canny_low}")
            
            elif key == ord('-'):
                # Decrease Canny low threshold
                detector.canny_low = max(0, detector.canny_low - 5)
                print(f"[*] Canny low threshold: {detector.canny_low}")
            
            elif key == ord('>'):
                # Increase Canny high threshold
                detector.canny_high = min(255, detector.canny_high + 5)
                print(f"[*] Canny high threshold: {detector.canny_high}")
            
            elif key == ord('<'):
                # Decrease Canny high threshold
                detector.canny_high = max(0, detector.canny_high - 5)
                print(f"[*] Canny high threshold: {detector.canny_high}")

            elif key == ord('t'):
                if last_warped_board is not None and board_found:
                    occupancy_baseline = compute_square_intensity_map(
                        last_warped_board,
                        generate_board_grid(output_size=warp_size, board_size=8, offset=(warp_padding, warp_padding))
                    )
                    print("[*] Occupancy baseline calibrated")
                else:
                    print("[!] Cannot calibrate occupancy baseline until a warped board is available")

            elif key == ord('m'):
                # Move detection: set baseline or compare to previous baseline
                if last_warped_board is None or last_grid is None or last_intensity_map is None:
                    print("[!] No warped board available to set/compare move baseline")
                else:
                    current_map = compute_square_average_intensity_map(last_warped_board, last_grid)
                    if move_baseline is None:
                        move_baseline = dict(current_map)
                        print("[*] Move baseline set. Make your move and press 'm' again to detect.")
                    else:
                        # Compute absolute and relative differences with a more robust intensity metric
                        diffs = {sq: current_map[sq] - move_baseline.get(sq, 0.0) for sq in current_map}
                        rel = {sq: (diffs[sq]) / (abs(move_baseline.get(sq, 0.0)) + 1.0) for sq in current_map}

                        # Sensitivity thresholds tuned for translucent pieces
                        abs_threshold = 4.0   # intensity units
                        rel_threshold = 0.04  # ~4% relative change

                        significant = []
                        for sq in diffs:
                            if abs(diffs[sq]) >= abs_threshold or abs(rel[sq]) >= rel_threshold:
                                significant.append((sq, diffs[sq], rel[sq]))

                        if not significant:
                            abs_threshold_relaxed = abs_threshold / 2.0
                            rel_threshold_relaxed = rel_threshold / 2.0
                            for sq in diffs:
                                if abs(diffs[sq]) >= abs_threshold_relaxed or abs(rel[sq]) >= rel_threshold_relaxed:
                                    significant.append((sq, diffs[sq], rel[sq]))

                        if not significant:
                            print("[*] No significant square changes detected (try lowering thresholds further)")
                        else:
                            # Identify the strongest positive and negative changes separately
                            positives = [item for item in significant if item[1] > 0]
                            negatives = [item for item in significant if item[1] < 0]
                            origin = None
                            dest = None

                            if positives and negatives:
                                best_pos = max(positives, key=lambda x: abs(x[1]))
                                best_neg = max(negatives, key=lambda x: abs(x[1]))
                                if abs(best_pos[1]) >= abs(best_neg[1]):
                                    origin, dest = best_pos[0], best_neg[0]
                                else:
                                    origin, dest = best_neg[0], best_pos[0]

                            if origin and dest and origin != dest:
                                print(f"[+] Move detected: {origin} -> {dest}")
                            else:
                                significant.sort(key=lambda x: abs(x[1]), reverse=True)
                                top = ', '.join([f"{s}({d:+.1f},{r:+.2f})" for s, d, r in significant[:6]])
                                print(f"[+] Significant changes: {top}")

                        move_baseline = dict(current_map)

            elif key == ord('o'):
                occupancy_enabled = not occupancy_enabled
                status = "ENABLED" if occupancy_enabled else "DISABLED"
                print(f"[*] Occupancy overlay {status}")
                if occupancy_enabled and occupancy_baseline is None:
                    print("[!] Occupancy baseline not set. Press 't' to set it when a valid warped board is visible.")

            elif key == ord('['):
                occupancy_threshold = max(1.0, occupancy_threshold - 1.0)
                print(f"[*] Occupancy threshold: {occupancy_threshold:.1f}")

            elif key == ord(']'):
                occupancy_threshold = min(255.0, occupancy_threshold + 1.0)
                print(f"[*] Occupancy threshold: {occupancy_threshold:.1f}")

            elif key == ord('w'):
                warp_enabled = not warp_enabled
                status = "ENABLED" if warp_enabled else "DISABLED"
                print(f"[*] Warped board view {status}")
                if not warp_enabled:
                    safe_destroy_window("Warped Board")
                    warp_window_created = False

            elif key == ord('g'):
                grid_enabled = not grid_enabled
                status = "ENABLED" if grid_enabled else "DISABLED"
                print(f"[*] Grid overlay {status}")
            
            elif key == ord('l'):
                show_labels = not show_labels
                status = "ENABLED" if show_labels else "DISABLED"
                print(f"[*] Grid labels {status}")
    
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    
    finally:
        reader.disconnect()
        cv2.destroyAllWindows()
    
    print(f"[+] Processed {frame_num} frames")
    return True


def main():
    """
    Main entry point for Step 1 & 2: IP Webcam + Chessboard Detection.
    
    Flow:
    1. Check for saved configuration
    2. Get IP address from user or use saved config
    3. Display live feed with optional board detection
    4. Save successful connection details and any detected corners
    
    Step 2 is integrated and can be toggled with 'b' key during streaming.
    """
    
    print("="*60)
    print("Step 1-3: IP Webcam + Chessboard Detection + Warp Calibration")
    print("Chess Vision Builder - Foundational CV")
    print("="*60)
    print()
    
    # Try to load existing configuration
    config = CalibrationManager.load_config()
    
    if config and "ip_address" in config:
        print(f"[*] Found saved configuration")
        print(f"    IP Address: {config['ip_address']}")
        print(f"    Resolution: {config['resolution']['width']}x{config['resolution']['height']}")
        
        use_saved = input("\n[?] Use saved IP address? (y/n): ").strip().lower()
        
        if use_saved == 'y':
            ip_address = config['ip_address']
        else:
            ip_address = input("[?] Enter phone's IP address (e.g., 192.168.1.100): ").strip()
    else:
        print("[*] No saved configuration found")
        print("\n[*] To find your phone's IP address:")
        print("    1. Open IP Webcam app on phone")
        print("    2. Look at the server message, or")
        print("    3. Check Settings → Wi-Fi → IP address")
        print()
        ip_address = input("[?] Enter phone's IP address (e.g., 192.168.1.100): ").strip()
    
    if not ip_address:
        print("[!] No IP address provided. Exiting.")
        return
    
    saved_corners = None
    saved_occupancy_baseline = None

    if config:
        if config.get("board_detected") and "board_corners" in config:
            saved_corners = np.array(config["board_corners"], dtype=np.float32)
        if "occupancy_baseline" in config:
            saved_occupancy_baseline = config["occupancy_baseline"]

    # Display live feed
    success = display_live_feed(
        ip_address,
        saved_corners=saved_corners,
        saved_occupancy_baseline=saved_occupancy_baseline,
    )
    
    if success:
        print("\n[+] Step 1-3 Complete!")
        print("[*] Board detection and perspective calibration are integrated and ready for use")
        print("[*] Next: Proceed to Step 4 - Grid Generation")
    else:
        print("\n[!] Steps 1 & 2 Failed - Check connection and try again")


if __name__ == "__main__":
    main()

"""
Step 2: Chessboard Corner Detection & Board Boundary Identification
Chess Vision Builder - Foundational Computer Vision

This module detects the physical chessboard's outer boundaries and corner coordinates
using OpenCV edge detection, contour analysis, and quadrilateral filtering.

Key Concepts:
- Canny edge detection for boundary extraction
- Contour detection and filtering by area/shape
- Perspective distortion handling (board at angles)
- Corner coordinate ordering (top-left, top-right, bottom-left, bottom-right)
"""
# type: ignore

import cv2
import numpy as np
from typing import Tuple, Optional, List


class BoardDetector:
    """
    Detects chessboard corners and boundaries in a frame.
    
    Uses multi-stage pipeline:
    1. Canny edge detection
    2. Contour detection
    3. Quadrilateral filtering
    4. Corner extraction and ordering
    
    Attributes:
        canny_threshold_low (int): Lower threshold for Canny edge detection
        canny_threshold_high (int): Upper threshold for Canny edge detection
        min_contour_area (float): Minimum contour area to consider (% of frame)
        epsilon_factor (float): Contour approximation precision
        debug_mode (bool): If True, return intermediate images for visualization
    """
    
    def __init__(self, 
                 canny_low: int = 50, 
                 canny_high: int = 150,
                 min_contour_area: float = 0.1,
                 epsilon_factor: float = 0.02,
                 smoothing_alpha: float = 0.2,
                 debug_mode: bool = False):
        """
        Initialize board detector with tuning parameters.
        
        Args:
            canny_low (int): Lower Canny threshold (typically 30-100)
            canny_high (int): Upper Canny threshold (typically 100-200)
            min_contour_area (float): Minimum contour area as % of frame (0.0-1.0)
            epsilon_factor (float): Contour approximation precision (0.01-0.05)
            smoothing_alpha (float): Temporal smoothing strength (0.0-1.0)
            debug_mode (bool): Return intermediate images for debugging
        
        Tuning Tips:
        - Increase canny_low/high if detecting too much noise
        - Decrease if board edges are faint
        - Increase min_contour_area if detecting wrong objects
        - Decrease epsilon_factor for more precise corner detection
        - Decrease smoothing_alpha for more stable, slower corner movement
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.smoothing_alpha = smoothing_alpha
        self.debug_mode = debug_mode
        self.last_corners = None  # Cache for temporal smoothing
        self.smoothed_corners = None
        self.missed_frames = 0
        self.max_missing_frames = 15
        
    def detect_board_corners(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], dict]:
        """
        Detect the 4 corners of the chessboard.
        
        Args:
            frame (np.ndarray): Input image from webcam (BGR format)
        
        Returns:
            Tuple:
            - success (bool): Whether board was detected
            - corners (np.ndarray): 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                   Ordered as: [top-left, top-right, bottom-right, bottom-left]
            - debug_data (dict): Intermediate images for visualization (if debug_mode=True)
        
        Returns:
            (False, None, {}) if board not detected
        """
        
        debug_data = {} if self.debug_mode else None
        
        # ===== STAGE 1: Edge Detection (Canny) =====
        # Convert to grayscale (intensity only, no color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        # Kernel size (5,5) provides light smoothing without losing edge sharpness
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        # - Edges with gradient > canny_high are strong edges
        # - Edges with gradient < canny_low are discarded
        # - Edges between low/high are kept if connected to strong edges (hysteresis)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        if self.debug_mode:
            debug_data['gray'] = gray # type: ignore
            debug_data['blurred'] = blurred # type: ignore
            debug_data['edges'] = edges # type: ignore
        
        # ===== STAGE 2: Contour Detection =====
        # Find all contours (closed boundaries) in the edge image
        # cv2.RETR_EXTERNAL: Only retrieve extreme outer contours (ignore internal edges)
        # cv2.CHAIN_APPROX_SIMPLE: Compress contours (store only corner points)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            if self.debug_mode:
                debug_data['contours_img'] = frame.copy() # type: ignore
            return False, None, debug_data #type: ignore
        
        # ===== STAGE 3: Contour Filtering =====
        # Frame area for filtering
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.min_contour_area
        
        # Filter contours by:
        # 1. Area (board is large)
        # 2. Shape (board is quadrilateral = 4 corners)
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filter
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            # epsilon: maximum distance from contour point to approximation line
            # Smaller epsilon = more accurate, but may detect noise
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Shape filter: must be quadrilateral (4 corners)
            if len(approx) == 4:
                candidates.append({
                    'contour': contour,
                    'approx': approx,
                    'area': area
                })
        
        if len(candidates) == 0:
            self.missed_frames += 1
            if self.last_corners is not None and self.missed_frames <= self.max_missing_frames:
                if self.debug_mode:
                    debug_data['contours_img'] = frame.copy() # type: ignore
                    debug_data['fallback'] = True # type: ignore
                return True, self.last_corners.copy(), debug_data #type: ignore
            if self.debug_mode:
                debug_data['contours_img'] = frame.copy() # type: ignore
            return False, None, debug_data #type: ignore
        
        self.missed_frames = 0
        # Select largest quadrilateral (most likely the board)
        best_candidate = max(candidates, key=lambda x: x['area'])
        corners = best_candidate['approx'].reshape(4, 2).astype(np.float32)
        
        # ===== STAGE 4: Corner Ordering =====
        # Sort corners consistently: top-left, top-right, bottom-right, bottom-left
        # This is critical for perspective transforms in Step 3
        corners_ordered = self._order_corners(corners)
        
        # Apply temporal smoothing so corners do not jitter frame-to-frame
        corners_smoothed = self._smooth_corners(corners_ordered)
        self.last_corners = corners_smoothed
        
        if self.debug_mode:
            # Draw contours on image for visualization
            contours_img = frame.copy()
            cv2.drawContours(contours_img, [best_candidate['contour']], 0, (0, 255, 0), 2)
            debug_data['contours_img'] = contours_img # type: ignore
            debug_data['best_contour_area'] = best_candidate['area'] # type: ignore
            debug_data['candidate_count'] = len(candidates) # type: ignore
        
        return True, corners_smoothed, debug_data # type: ignore
    
    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left.
        
        This ensures corners have predictable ordering regardless of perspective angle.
        
        Args:
            corners (np.ndarray): 4 corner points (may be in any order)
        
        Returns:
            np.ndarray: 4 corners in order [TL, TR, BR, BL]
        """
        # Use sum/difference method for robust rectangle corner ordering
        # even when the board is rotated in the frame.
        corners = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)

        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).reshape(4)

        ordered[0] = corners[np.argmin(sums)]  # top-left has smallest x+y
        ordered[2] = corners[np.argmax(sums)]  # bottom-right has largest x+y
        ordered[1] = corners[np.argmin(diffs)] # top-right has smallest x-y
        ordered[3] = corners[np.argmax(diffs)] # bottom-left has largest x-y

        return ordered

    def _smooth_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Smooth corner positions over time using exponential moving average.

        Args:
            corners (np.ndarray): Newly detected ordered corners

        Returns:
            np.ndarray: Smoothed corner coordinates
        """
        if self.smoothed_corners is None:
            self.smoothed_corners = corners.copy()
            return self.smoothed_corners

        alpha = np.clip(self.smoothing_alpha, 0.0, 1.0)
        self.smoothed_corners = (
            self.smoothed_corners * (1.0 - alpha)
            + corners * alpha
        )
        return self.smoothed_corners


def warp_board(frame: np.ndarray,
               corners: np.ndarray,
               output_size: int = 800,
               padding: int = 0,
               scale_x: float = 1.0,
               scale_y: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp the detected board into a top-down square view.

    Args:
        frame (np.ndarray): Original color frame
        corners (np.ndarray): Ordered board corners [TL, TR, BR, BL]
        output_size (int): Output warp size in pixels

    Returns:
        warped (np.ndarray): Top-down board image
        matrix (np.ndarray): Perspective transform matrix
    """
    if corners is None:
        raise ValueError("Expected 4 ordered corners for warping, got None")

    # Ensure corners is a numpy array and has shape (4,2)
    corners_arr = np.asarray(corners)
    try:
        corners_arr = corners_arr.reshape(4, 2)
    except Exception:
        raise ValueError(f"Expected corners shape (4,2), got {corners_arr.shape}")

    # OpenCV requires float32 inputs for getPerspectiveTransform
    if corners_arr.dtype != np.float32:
        corners_arr = corners_arr.astype(np.float32)

    padding = max(0, int(padding))

    # Allow non-square warps via independent scaling factors.
    # scale_x/scale_y are multipliers applied to the base output_size.
    sx = max(0.01, float(scale_x))
    sy = max(0.01, float(scale_y))
    out_w = max(1, int(round(output_size * sx)))
    out_h = max(1, int(round(output_size * sy)))

    total_w = out_w + padding * 2
    total_h = out_h + padding * 2

    destination = np.array([
        [padding, padding],
        [padding + out_w - 1, padding],
        [padding + out_w - 1, padding + out_h - 1],
        [padding, padding + out_h - 1]
    ], dtype=np.float32)

    try:
        matrix = cv2.getPerspectiveTransform(corners_arr, destination)
    except Exception as e:
        raise RuntimeError(f"getPerspectiveTransform failed: {e}")

    warped = cv2.warpPerspective(frame, matrix, (total_w, total_h))
    return warped, matrix


def generate_board_grid(output_size: int = 800,
                        board_size: int = 8,
                        offset: tuple[int, int] = (0, 0)) -> dict:
    """
    Generate a grid of square regions for the warped board.

    Args:
        output_size (int): Width/height of the warped board image
        board_size (int): Number of squares per side (8 for chess)

    Returns:
        dict: Mapping of square name to region coordinates
    """
    # Support either a scalar output_size (square) or a tuple/list (width, height)
    if isinstance(output_size, (tuple, list)):
        out_w, out_h = int(output_size[0]), int(output_size[1])
    else:
        out_w = out_h = int(output_size)

    ox, oy = offset
    step_x = out_w / board_size
    step_y = out_h / board_size
    files = 'abcdefgh'
    ranks = list(range(board_size, 0, -1))

    grid = {}
    for rank_idx, rank in enumerate(ranks):
        for file_idx, file in enumerate(files):
            x1 = int(file_idx * step_x + ox)
            y1 = int(rank_idx * step_y + oy)
            x2 = int((file_idx + 1) * step_x + ox)
            y2 = int((rank_idx + 1) * step_y + oy)
            square = f"{file}{rank}"
            grid[square] = {
                'top_left': (x1, y1),
                'bottom_right': (x2, y2),
                'center': (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            }
    return grid


def draw_board_grid(frame: np.ndarray,
                    grid: dict,
                    show_labels: bool = False,
                    color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Draw the 8x8 board grid on a warped board image.

    Args:
        frame (np.ndarray): Warped board image
        grid (dict): Square regions returned by generate_board_grid
        show_labels (bool): Whether to draw square names in each cell
        color (tuple): Line color

    Returns:
        np.ndarray: Image with grid overlay
    """
    overlay = frame.copy()
    x_coords = [region['top_left'][0] for region in grid.values()] + [region['bottom_right'][0] for region in grid.values()]
    y_coords = [region['top_left'][1] for region in grid.values()] + [region['bottom_right'][1] for region in grid.values()]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    board_width = max_x - min_x
    board_height = max_y - min_y
    step_x = board_width / 8
    step_y = board_height / 8

    # Draw horizontal and vertical grid lines for the inner board region.
    for i in range(9):
        xs = int(min_x + i * step_x)
        ys = int(min_y + i * step_y)
        cv2.line(overlay, (xs, min_y), (xs, max_y), color, 1)
        cv2.line(overlay, (min_x, ys), (max_x, ys), color, 1)

    if show_labels:
        for square, data in grid.items():
            cx, cy = data['center']
            cv2.putText(overlay, square, (cx - 18, cy + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return overlay


def compute_square_intensity_map(warped: np.ndarray, grid: dict) -> dict:
    """
    Compute grayscale intensity for each square on the warped board.

    Args:
        warped (np.ndarray): Warped board image
        grid (dict): Square grid mapping

    Returns:
        dict: Mapping of square name to mean intensity
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    intensity_map = {}

    for square, region in grid.items():
        x1, y1 = region['top_left']
        x2, y2 = region['bottom_right']
        square_img = gray[y1:y2, x1:x2]
        if square_img.size == 0:
            intensity_map[square] = 0.0
        else:
            intensity_map[square] = float(square_img.mean())

    return intensity_map


def compute_square_average_intensity_map(warped: np.ndarray, grid: dict) -> dict:
    """
    Compute a more robust intensity metric per square using mean and median.

    This helps with translucent pieces where the square intensity change may be
    subtle or affected by the underlying board color.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    average_map = {}

    for square, region in grid.items():
        x1, y1 = region['top_left']
        x2, y2 = region['bottom_right']
        square_img = gray[y1:y2, x1:x2]
        if square_img.size == 0:
            average_map[square] = 0.0
        else:
            mean_val = float(square_img.mean())
            median_val = float(np.median(square_img))
            average_map[square] = 0.5 * mean_val + 0.5 * median_val

    return average_map


def compare_intensity_maps(reference: dict,
                           current: dict,
                           threshold: float = 18.0) -> list:
    """
    Compare two intensity maps and report squares that changed more than threshold.

    Args:
        reference (dict): Baseline square intensities
        current (dict): Current square intensities
        threshold (float): Difference threshold to consider a square changed

    Returns:
        list: List of square names with changes
    """
    changed = []
    for square, current_value in current.items():
        reference_value = reference.get(square)
        if reference_value is None:
            continue
        if abs(current_value - reference_value) >= threshold:
            changed.append(square)
    return changed


def draw_occupancy_map(frame: np.ndarray,
                       grid: dict,
                       occupied_squares: list,
                       color: tuple = (0, 0, 255),
                       thickness: int = 2) -> np.ndarray:
    """
    Highlight occupied or changed squares on a warped board image.

    Args:
        frame (np.ndarray): Warped board image
        grid (dict): Square grid mapping
        occupied_squares (list): Square names to highlight
        color (tuple): Box color
        thickness (int): Rectangle thickness

    Returns:
        np.ndarray: Frame with occupancy overlay
    """
    overlay = frame.copy()
    for square in occupied_squares:
        region = grid.get(square)
        if region is None:
            continue
        x1, y1 = region['top_left']
        x2, y2 = region['bottom_right']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(overlay, square, (x1 + 5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return overlay


def safe_warp_board(frame: np.ndarray,
                    corners: np.ndarray,
                    output_size: int = 800) -> Optional[np.ndarray]:
    """
    Attempt to warp the board and return None on failure.
    """
    try:
        warped, _ = warp_board(frame, corners, output_size=output_size)
        return warped
    except Exception:
        return None


def get_board_grid_from_warp(output_size: int = 800,
                             board_size: int = 8) -> dict:
	return generate_board_grid(output_size=output_size, board_size=board_size)


def draw_board_grid_overlay(frame: np.ndarray,
                            output_size: int = 800,
                            board_size: int = 8,
                            show_labels: bool = False) -> np.ndarray:
    """
    Wrapper that generates and draws the board grid overlay.
    """
    grid = generate_board_grid(output_size=output_size, board_size=board_size)
    return draw_board_grid(frame, grid, show_labels=show_labels)


def get_square_region(grid: dict, square: str) -> Optional[dict]:
    """
    Return the region coordinates for a named square.
    """
    return grid.get(square)


def get_square_name_at_point(point: tuple, board_size: int = 8, output_size: int = 800) -> Optional[str]:
    """
    Map a point in warped board coordinates to a square name.
    """
    x, y = point
    step = output_size / board_size
    file_idx = int(x // step)
    rank_idx = int(y // step)

    if 0 <= file_idx < board_size and 0 <= rank_idx < board_size:
        file = 'abcdefgh'[file_idx]
        rank = str(board_size - rank_idx)
        return f"{file}{rank}"
    return None


def get_square_image(warped: np.ndarray, square: str, grid: dict) -> Optional[np.ndarray]:
    """
    Extract the image region for a named square.
    """
    region = get_square_region(grid, square)
    if region is None:
        return None
    x1, y1 = region['top_left']
    x2, y2 = region['bottom_right']
    return warped[y1:y2, x1:x2]


def extract_square_intensity(warped: np.ndarray, square: str, grid: dict) -> Optional[float]:
    """
    Compute the mean grayscale intensity for a named square.
    """
    square_img = get_square_image(warped, square, grid)
    if square_img is None:
        return None
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def draw_board_corners(frame: np.ndarray,
                      corners: np.ndarray,
                      thickness: int = 3,
                      radius: int = 8) -> np.ndarray:
    """
    Draw detected corners and connecting lines on frame.
    
    Args:
        frame (np.ndarray): Image to draw on
        corners (np.ndarray): 4 corner coordinates
        thickness (int): Line thickness
        radius (int): Corner circle radius
    
    Returns:
        np.ndarray: Frame with drawn corners
    """
    
    result = frame.copy()
    
    # Draw connecting lines between corners
    # Blue lines connecting the board boundary
    corners_int = corners.astype(np.int32)
    cv2.polylines(result, [corners_int], True, (255, 0, 0), thickness)
    
    # Draw corner circles with labels
    colors = [
        (0, 255, 0),      # Top-left: Green
        (0, 165, 255),    # Top-right: Orange
        (0, 0, 255),      # Bottom-right: Red
        (255, 255, 0)     # Bottom-left: Cyan
    ]
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
        x, y = corner.astype(int)
        
        # Draw filled circle
        cv2.circle(result, (x, y), radius, color, -1)
        
        # Draw outline
        cv2.circle(result, (x, y), radius, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(result, label, (x + 15, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result


def draw_debug_images(frame: np.ndarray,
                     debug_data: dict) -> None:
    """
    Display debug images in separate windows.
    
    Args:
        frame (np.ndarray): Original frame
        debug_data (dict): Debug images from detector
    """
    
    if 'edges' in debug_data:
        cv2.imshow("Edges (Canny)", debug_data['edges'])
    
    if 'contours_img' in debug_data:
        cv2.imshow("Detected Contours", debug_data['contours_img'])
    
    if 'gray' in debug_data:
        cv2.imshow("Grayscale", debug_data['gray'])


# Example tuning profiles for different lighting/board conditions
TUNING_PROFILES = {
    'bright': {
        'canny_low': 80,
        'canny_high': 200,
        'min_contour_area': 0.15,
        'epsilon_factor': 0.02
    },
    'dim': {
        'canny_low': 30,
        'canny_high': 100,
        'min_contour_area': 0.10,
        'epsilon_factor': 0.02
    },
    'high_contrast': {
        'canny_low': 50,
        'canny_high': 150,
        'min_contour_area': 0.12,
        'epsilon_factor': 0.015
    },
    'low_contrast': {
        'canny_low': 20,
        'canny_high': 80,
        'min_contour_area': 0.08,
        'epsilon_factor': 0.03
    }
}

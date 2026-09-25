"""
=============================================================================
  Anomaly Detection in Residential Security Using Facial Recognition
=============================================================================
  Academic Prototype | Temporal Logic Gate Security System

  Pipeline Overview:
  +------------------------------------------------------------------+
  |  Webcam  ->  Frame Downscale (25%)  ->  Face Detection/Encoding  |
  |          ->  Known-Face Comparison (tol=0.6)                     |
  |          ->  Temporal Logic Gate (23:00 - 05:00)                 |
  |          ->  Alert (Beep + Snapshot) | Annotated Display         |
  +------------------------------------------------------------------+

  Author  : Academic Prototype
  Platform: Windows (primary) | Linux / macOS (fallback audio)
  Deps    : opencv-python, face_recognition, numpy
=============================================================================
"""

import cv2
import face_recognition
import numpy as np
import os
import sys
import time
import platform
from datetime import datetime


# =============================================================================
#  SECTION 1 - CONFIGURATION CONSTANTS
# =============================================================================

KNOWN_FACES_DIR = "known_faces"   # Directory containing resident portrait images
ALERTS_DIR      = "alerts"        # Directory where intruder snapshots are saved

SCALE_FACTOR    = 0.25            # Downscale factor for faster face detection
TOLERANCE       = 0.6             # Face-distance threshold (lower = stricter match)
FRAME_THICKNESS = 2               # Bounding-box line thickness (px)
FONT_SCALE      = 0.65            # Label font scale
FONT_THICKNESS  = 2               # Label font stroke thickness

# Temporal Logic Gate: hours that define "off-hours / unusual window"
START_HOUR = 23   # 11 PM  (inclusive)
END_HOUR   = 5    # 05 AM  (exclusive upper bound)

# Non-blocking cooldown: minimum seconds between consecutive alerts
ALERT_COOLDOWN_SEC = 3

# Alert tone parameters (Windows winsound.Beep)
BEEP_FREQ_HZ  = 1000   # Frequency in Hertz
BEEP_DURATION = 500    # Duration in milliseconds


# =============================================================================
#  SECTION 2 - ENVIRONMENT SETUP
# =============================================================================

def ensure_directories():
    """
    Create required working directories if they do not already exist.
    Called once at startup before any I/O operations.
    """
    for directory in (KNOWN_FACES_DIR, ALERTS_DIR):
        os.makedirs(directory, exist_ok=True)
        print(f"[SETUP] Directory verified: '{directory}/'")


# =============================================================================
#  SECTION 3 - DATA MANAGEMENT: LOAD KNOWN RESIDENT ENCODINGS
# =============================================================================

def load_known_faces(directory):
    """
    Scan `directory` for portrait images, compute 128-dimensional facial
    encodings using Dlib's ResNet model, and return them as parallel lists.

    Args:
        directory (str): Path to the folder containing resident images (Name.jpg).

    Returns:
        known_encodings (list): One 128-d numpy vector per resident.
        known_names     (list): Corresponding resident name strings.

    Notes:
        - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
        - Files with no detectable face are skipped with a warning.
        - The filename stem (without extension) is used as the resident's name.
    """
    known_encodings = []
    known_names     = []

    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files   = [f for f in os.listdir(directory)
                     if f.lower().endswith(supported_ext)]

    if not image_files:
        print(f"[WARNING] No portrait images found in '{directory}/'. "
              "Add resident images named 'Name.jpg' to enable recognition.")
        return known_encodings, known_names

    print(f"[SETUP] Loading {len(image_files)} resident image(s)...")

    for filename in image_files:
        filepath    = os.path.join(directory, filename)
        # Strip extension to derive the resident's display name
        person_name = os.path.splitext(filename)[0]

        try:
            # face_recognition.load_image_file returns an RGB numpy array
            image     = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                # No face detected in the portrait -- skip gracefully
                print(f"[WARNING] No face detected in '{filename}'. "
                      "Ensure the image contains a clear frontal face.")
                continue

            # Use the first (primary) face encoding found in the portrait
            known_encodings.append(encodings[0])
            known_names.append(person_name)
            print(f"  [OK] Loaded: {person_name}")

        except Exception as exc:
            print(f"[ERROR] Failed to load '{filename}': {exc}")

    print(f"[SETUP] {len(known_names)} resident encoding(s) ready.\n")
    return known_encodings, known_names


# =============================================================================
#  SECTION 4 - TEMPORAL LOGIC GATE
# =============================================================================

def is_off_hours(start=START_HOUR, end=END_HOUR):
    """
    Evaluate the Temporal Logic Gate.

    Determines whether the current wall-clock hour falls within the
    designated high-risk surveillance window (e.g., 23:00 - 05:00).

    The gate handles the midnight-crossing case:
        start=23, end=5  -> True for hours {23, 0, 1, 2, 3, 4}

    Args:
        start (int): Hour that begins the off-hours window (inclusive, 0-23).
        end   (int): Hour that ends  the off-hours window (exclusive, 0-23).

    Returns:
        bool: True  -> anomaly should be escalated (HIGH PRIORITY).
              False -> log only, no alert triggered.
    """
    current_hour = datetime.now().hour

    if start > end:
        # Window crosses midnight  (e.g., 23 -> 05)
        return current_hour >= start or current_hour < end
    else:
        # Window within same calendar day (e.g., 01 -> 06)
        return start <= current_hour < end


# =============================================================================
#  SECTION 5 - ACTION LAYER: ALERT MECHANISMS
# =============================================================================

def play_alert_sound():
    """
    Trigger an audible alarm in a non-blocking manner.

    Strategy (cross-platform):
        Windows -> winsound.Beep(freq, duration)
        macOS   -> os.system('afplay /System/Library/Sounds/Sosumi.aiff')
        Linux   -> ASCII bell via print() fallback
    """
    system = platform.system()
    if system == "Windows":
        try:
            import winsound
            winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION)
        except Exception:
            pass  # Silent fallback if audio device unavailable
    elif system == "Darwin":
        os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
    else:
        # Linux: attempt ASCII bell via print (works in some terminals)
        print("\a", end="", flush=True)


def save_alert_snapshot(frame, alerts_dir):
    """
    Persist the current (full-resolution) video frame to disk as evidence.

    Args:
        frame      (np.ndarray): The original (unscaled) BGR frame from OpenCV.
        alerts_dir (str)       : Target directory path for saved snapshots.

    Returns:
        filepath (str): Path of the saved image file.

    File naming convention: intruder_YYYYMMDD_HHMMSS.jpg
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"intruder_{timestamp}.jpg"
    filepath  = os.path.join(alerts_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


# =============================================================================
#  SECTION 6 - DECISION ENGINE: FACE COMPARISON
# =============================================================================

def identify_face(live_encoding, known_encodings, known_names, tolerance=TOLERANCE):
    """
    Compare a live face encoding against all known resident encodings.

    Uses Euclidean (L2) face distance to find the best match, which is
    more robust than a simple boolean compare_faces result.

    Args:
        live_encoding   (np.ndarray): 128-d encoding from the current video frame.
        known_encodings (list)      : Pre-computed encodings of known residents.
        known_names     (list)      : Corresponding resident name labels.
        tolerance       (float)     : Maximum acceptable face distance for a match.

    Returns:
        name     (str) : Resident name if matched, else "Unknown".
        is_known (bool): True if a resident was identified, False otherwise.
    """
    if not known_encodings:
        return "Unknown", False

    # Compute L2 distances from all known encodings to the live face
    face_distances = face_recognition.face_distance(known_encodings, live_encoding)

    # Select the encoding with the minimum distance (closest match)
    best_match_idx = int(np.argmin(face_distances))

    if face_distances[best_match_idx] <= tolerance:
        # Match found within acceptable tolerance
        return known_names[best_match_idx], True
    else:
        return "Unknown", False


# =============================================================================
#  SECTION 7 - ANNOTATION: DRAW HUD OVERLAY ON FRAME
# =============================================================================

def draw_face_annotation(frame, top, right, bottom, left, name, is_known, is_alert):
    """
    Render a bounding box and name label onto the video frame (in-place).

    Color coding:
        Green  (0, 220, 0)   -> Known resident
        Amber  (0, 200, 255) -> Unknown person during normal hours (no alert)
        Red    (0, 0, 220)   -> Unknown person during off-hours (HIGH ALERT)

    Args:
        frame            (np.ndarray): BGR video frame to annotate.
        top, right, bottom, left (int): Bounding box corners (full-res coords).
        name     (str) : Text label to display above the box.
        is_known (bool): Whether the face matched a resident.
        is_alert (bool): Whether the Temporal Logic Gate is active.
    """
    # Select box color based on identity + temporal gate state
    if is_known:
        color = (0, 220, 0)        # Green  -> Resident
        label = name
    elif is_alert:
        color = (0, 0, 220)        # Red    -> Intruder (off-hours)
        label = "UNKNOWN - ALERT"
    else:
        color = (0, 200, 255)      # Amber  -> Unknown (daytime)
        label = "Unknown"

    # Draw bounding box around the detected face
    cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)

    # Filled label background for readability
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, FONT_THICKNESS
    )
    cv2.rectangle(
        frame,
        (left, top - text_h - 14),
        (left + text_w + 8, top),
        color, -1  # Filled rectangle (background for text)
    )

    # Name / status label (white text on coloured background)
    cv2.putText(
        frame, label,
        (left + 4, top - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA
    )


def draw_hud(frame, face_count, alert_active):
    """
    Render a Heads-Up Display (HUD) status bar at the top of the frame.

    Displays: current system time, gate status, and face count.

    Args:
        frame        (np.ndarray): BGR video frame to annotate.
        face_count   (int)       : Number of faces detected this frame.
        alert_active (bool)      : Whether the temporal gate is currently open.
    """
    now_str    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    gate_str   = "GATE: OFF-HOURS [!]" if alert_active else "GATE: Normal Hours"
    status_str = f"{now_str}  |  {gate_str}  |  Faces: {face_count}"

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    hud_color = (0, 60, 220) if alert_active else (180, 180, 180)
    cv2.putText(
        frame, status_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        hud_color, 1, cv2.LINE_AA
    )


# =============================================================================
#  SECTION 8 - MAIN SURVEILLANCE LOOP
# =============================================================================

def run_surveillance(known_encodings, known_names):
    """
    Main real-time surveillance loop.

    Captures frames from the default webcam, runs the full detection and
    recognition pipeline on each frame, applies the Temporal Logic Gate,
    triggers alerts when warranted, and renders the annotated feed.

    Exits cleanly when the user presses 'q'.

    Args:
        known_encodings (list): Pre-loaded resident face encodings.
        known_names     (list): Corresponding resident name labels.
    """
    # Open the default system webcam (device index 0)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Cannot access webcam. "
              "Ensure the camera is connected and not in use by another app.")
        sys.exit(1)

    print("[INFO] Webcam opened successfully. Press 'q' to quit.\n")

    # ---- Cooldown state -------------------------------------------------
    # Tracks the last alert time using monotonic clock to enforce cooldown.
    # Monotonic time is immune to system-clock changes (e.g., NTP sync).
    last_alert_time = 0.0

    # ---- Main capture loop ----------------------------------------------
    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("[ERROR] Failed to read frame from webcam. Exiting...")
            break

        # ---- LAYER 2: Localised Computation -----------------------------
        # Downscale to 25% of original resolution for fast CPU inference.
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        # face_recognition expects RGB; OpenCV delivers BGR -> must convert
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face bounding boxes in the downscaled frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Extract 128-d Dlib encoding for each detected face region
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # ---- LAYER 3: Temporal Logic Gate (evaluated once per frame) ----
        alert_active = is_off_hours(START_HOUR, END_HOUR)

        # ---- LAYER 4 + 5: Decision Engine & Action Layer ----------------
        for (top, right, bottom, left), live_encoding in zip(face_locations, face_encodings):

            # Scale bounding-box coordinates back to original resolution
            # (inverse of SCALE_FACTOR; e.g., x4 for 0.25 downscale)
            scale  = int(1 / SCALE_FACTOR)
            top    *= scale
            right  *= scale
            bottom *= scale
            left   *= scale

            # Identify face: compare against all known resident encodings
            name, is_known = identify_face(live_encoding, known_encodings, known_names)

            # Annotate frame with colored box + label
            draw_face_annotation(
                frame, top, right, bottom, left,
                name, is_known,
                is_alert=(not is_known and alert_active)
            )

            # --- Trigger alert: Unknown + Off-Hours + Cooldown expired ---
            if (not is_known) and alert_active:
                now_mono = time.monotonic()

                if (now_mono - last_alert_time) >= ALERT_COOLDOWN_SEC:
                    # Log alert to console
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[ALERT] Unknown person detected at {ts}! "
                          f"(off-hours gate OPEN)")

                    # Trigger audio alarm (cross-platform)
                    play_alert_sound()

                    # Save full-resolution timestamped snapshot as evidence
                    saved_path = save_alert_snapshot(frame, ALERTS_DIR)
                    print(f"[ALERT] Snapshot saved -> {saved_path}")

                    # Update cooldown timestamp (non-blocking)
                    last_alert_time = now_mono

        # ---- HUD Overlay ------------------------------------------------
        draw_hud(frame, face_count=len(face_locations), alert_active=alert_active)

        # ---- Render the annotated security feed -------------------------
        cv2.imshow("Residential Security Feed - Anomaly Detection", frame)

        # Exit on 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received. Shutting down...")
            break

    # ---- Cleanup --------------------------------------------------------
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Session ended.")


# =============================================================================
#  SECTION 9 - ENTRY POINT
# =============================================================================

def main():
    """
    Orchestrates the full startup sequence:
        1. Verify / create working directories.
        2. Load and encode known resident portraits.
        3. Launch the real-time surveillance loop.
    """
    print("=" * 65)
    print("  Anomaly Detection in Residential Security")
    print("  Using Facial Recognition  |  Temporal Logic Gate")
    print("=" * 65)
    print(f"  Off-hours window : {START_HOUR:02d}:00 - {END_HOUR:02d}:00")
    print(f"  Match tolerance  : {TOLERANCE}")
    print(f"  Alert cooldown   : {ALERT_COOLDOWN_SEC}s")
    print("=" * 65 + "\n")

    # Step 1: Ensure working directories exist
    ensure_directories()

    # Step 2: Load known resident encodings from known_faces/
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    # Step 3: Launch real-time surveillance
    run_surveillance(known_encodings, known_names)


if __name__ == "__main__":
    main()
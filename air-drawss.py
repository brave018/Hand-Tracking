# ============================================================
#  air-drawss.py  –  Real-Time Hand Gesture Air-Drawing App
#  MediaPipe 0.10+ Tasks API compatible
#  Run: python air-drawss.py   |   Quit: press 'q'
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import urllib.request

from mediapipe.tasks.python            import BaseOptions
from mediapipe.tasks.python.vision    import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
    HandLandmarksConnections,
)
from mediapipe                         import Image, ImageFormat

# ── 2. CONSTANTS ─────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Color palette (BGR)
COLORS = {
    "Neon Green": (57, 255, 20),
    "Pink":       (147, 20, 255),
    "Cyan":       (255, 255, 0),
    "Orange":     (0, 165, 255),
}
COLOR_NAMES  = list(COLORS.keys())
COLOR_VALUES = list(COLORS.values())

ERASER_COLOR     = (255, 255, 255)
ERASER_THICKNESS = 30

THICKNESS_LEVELS = {"THIN": 2, "MEDIUM": 6, "THICK": 14}
THICKNESS_NAMES  = list(THICKNESS_LEVELS.keys())
THICKNESS_VALUES = list(THICKNESS_LEVELS.values())

# Toolbar geometry
TB_WIDTH       = 70
TB_PAD         = 20
TB_SWATCH_R    = 18
TB_SWATCH_GAP  = 12
TB_BTN_SIZE    = 44
TB_SECTION_GAP = 18

DWELL_TIME = 0.8

FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = 0.45
FONT_MD = 0.55

# MediaPipe landmark indices
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]   # PIP for index/middle/ring/pinky; MCP for thumb


# ── 3. MODEL BOOTSTRAP ────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[Setup] Downloading hand_landmarker.task (~7.8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[Setup] Model saved -> {MODEL_PATH}")
    else:
        print(f"[Setup] Model found -> {MODEL_PATH}")


# ── 4. HELPER FUNCTIONS ───────────────────────────────────────

def landmark_px(lm, w, h):
    """Convert a normalized landmark to pixel (x, y)."""
    return int(lm.x * w), int(lm.y * h)


def is_finger_extended_norm(lm_list, tip_idx, pip_idx) -> bool:
    """Return True if tip.y < pip.y (tip is above PIP in image coords)."""
    return lm_list[tip_idx].y < lm_list[pip_idx].y


def get_gesture(lm_list) -> str:
    """
    lm_list: list of NormalizedLandmark from HandLandmarkerResult
    Returns: 'DRAW' | 'PAUSE' | 'NONE'
    """
    index_up  = is_finger_extended_norm(lm_list, 8,  6)
    middle_up = is_finger_extended_norm(lm_list, 12, 10)
    ring_up   = is_finger_extended_norm(lm_list, 16, 14)
    pinky_up  = is_finger_extended_norm(lm_list, 20, 18)

    if index_up and middle_up and ring_up and pinky_up:
        return "PAUSE"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "DRAW"
    return "NONE"


def redraw_canvas(strokes: list, shape: tuple) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for stroke in strokes:
        pts, color, thick = stroke["points"], stroke["color"], stroke["thickness"]
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(canvas, pts[0], max(1, thick // 2), color, -1, cv2.LINE_AA)
    return canvas


def build_toolbar_zones(frame_h: int, frame_w: int) -> dict:
    zones = {}
    x0    = frame_w - TB_WIDTH
    y     = TB_PAD + 10

    for i in range(len(COLOR_NAMES)):
        cx = x0 + TB_WIDTH // 2
        cy = y + TB_SWATCH_R
        zones[f"color_{i}"] = (cx - TB_SWATCH_R, cy - TB_SWATCH_R,
                                cx + TB_SWATCH_R, cy + TB_SWATCH_R)
        y += TB_SWATCH_R * 2 + TB_SWATCH_GAP

    y += TB_SECTION_GAP

    for i in range(len(THICKNESS_NAMES)):
        zones[f"thick_{i}"] = (x0 + 4, y, x0 + TB_WIDTH - 4, y + 28)
        y += 34

    y += TB_SECTION_GAP

    for name in ("UNDO", "REDO", "ERASER", "CLEAR"):
        bx1 = x0 + (TB_WIDTH - TB_BTN_SIZE) // 2
        zones[name] = (bx1, y, bx1 + TB_BTN_SIZE, y + TB_BTN_SIZE)
        y += TB_BTN_SIZE + 10

    return zones


# ── Icon drawers ──────────────────────────────────────────────

def _draw_undo_icon(img, cx, cy, col):
    pts = np.array([[cx+10,cy-4],[cx-2,cy-4],[cx-2,cy-10],
                    [cx-12,cy],[cx-2,cy+10],[cx-2,cy+4],[cx+10,cy+4]], np.int32)
    cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[cx-12,cy],[cx-4,cy-5],[cx-4,cy+5]], np.int32)], col)


def _draw_redo_icon(img, cx, cy, col):
    pts = np.array([[cx-10,cy-4],[cx+2,cy-4],[cx+2,cy-10],
                    [cx+12,cy],[cx+2,cy+10],[cx+2,cy+4],[cx-10,cy+4]], np.int32)
    cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[cx+12,cy],[cx+4,cy-5],[cx+4,cy+5]], np.int32)], col)


def _draw_eraser_icon(img, cx, cy, col):
    cv2.rectangle(img, (cx-12, cy-7), (cx+12, cy+7), col, 2)
    cv2.line(img, (cx-2, cy-7), (cx-2, cy+7), col, 1)
    cv2.putText(img, "E", (cx-5, cy+5), FONT, 0.45, col, 1, cv2.LINE_AA)


def _draw_clear_icon(img, cx, cy, col):
    cv2.rectangle(img, (cx-9, cy-5),  (cx+9,  cy+11), col, 2)
    cv2.rectangle(img, (cx-11, cy-9), (cx+11, cy-5),  col, 2)
    cv2.rectangle(img, (cx-4, cy-13), (cx+4,  cy-9),  col, 2)
    for dx in (-4, 0, 4):
        cv2.line(img, (cx+dx, cy-2), (cx+dx, cy+8), col, 1)


def draw_toolbar(frame: np.ndarray, state: dict, zones: dict):
    h, w = frame.shape[:2]
    x0   = w - TB_WIDTH
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, TB_PAD), (w-1, h-TB_PAD), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, TB_PAD), (w-1, h-TB_PAD), (80, 80, 80), 1)

    # Color swatches
    for i, (name, bgr) in enumerate(zip(COLOR_NAMES, COLOR_VALUES)):
        z  = zones[f"color_{i}"]
        cx = (z[0] + z[2]) // 2
        cy = (z[1] + z[3]) // 2
        cv2.circle(frame, (cx, cy), TB_SWATCH_R, bgr, -1, cv2.LINE_AA)
        if i == state["color_idx"] and not state["eraser_mode"]:
            cv2.circle(frame, (cx, cy), TB_SWATCH_R + 3, (255, 255, 255), 2, cv2.LINE_AA)

    # Thickness selectors
    for i, (tname, tval) in enumerate(zip(THICKNESS_NAMES, THICKNESS_VALUES)):
        z = zones[f"thick_{i}"]
        bx1, by1, bx2, by2 = z
        if i == state["thick_idx"]:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
        mid_y = (by1 + by2) // 2
        col   = (220, 220, 220) if i == state["thick_idx"] else (120, 120, 120)
        cv2.line(frame, (bx1+6, mid_y), (bx2-6, mid_y), col, tval)

    # Tool buttons
    btn_info = {
        "UNDO":   (200, 200, 200),
        "REDO":   (200, 200, 200),
        "ERASER": (60, 180, 255) if state["eraser_mode"] else (200, 200, 200),
        "CLEAR":  (80, 80, 255),
    }
    icon_fns = {
        "UNDO": _draw_undo_icon, "REDO": _draw_redo_icon,
        "ERASER": _draw_eraser_icon, "CLEAR": _draw_clear_icon,
    }
    for bname, col in btn_info.items():
        z = zones[bname]
        bx1, by1, bx2, by2 = z
        cx, cy = (bx1+bx2)//2, (by1+by2)//2
        bg = (60,60,60) if (bname == "ERASER" and state["eraser_mode"]) else (35,35,35)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), bg, -1)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (90,90,90), 1)
        icon_fns[bname](frame, cx, cy, col)


def check_dwell(fingertip, zones: dict, dwell_state: dict, dt: float):
    fx, fy = fingertip
    hit = None
    for name, (x1, y1, x2, y2) in zones.items():
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            hit = name
            break

    if hit is None:
        dwell_state.update(zone=None, elapsed=0.0)
        return None, dwell_state

    if dwell_state["zone"] != hit:
        dwell_state.update(zone=hit, elapsed=0.0)
        return None, dwell_state

    dwell_state["elapsed"] += dt
    if dwell_state["elapsed"] >= DWELL_TIME:
        dwell_state.update(zone=None, elapsed=0.0)
        return hit, dwell_state

    return None, dwell_state


def draw_dwell_arc(frame, cx, cy, elapsed):
    progress = min(elapsed / DWELL_TIME, 1.0)
    angle    = int(360 * progress)
    cv2.circle(frame, (cx, cy), 22, (60, 60, 60), 3, cv2.LINE_AA)
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (22, 22), -90, 0, angle, (255,255,255), 3, cv2.LINE_AA)


def put_shadow_text(img, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, (0,0,0),   thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, color, thick,   cv2.LINE_AA)


def draw_hud(frame, mode, color_name, thick_name, fps, eraser_mode):
    pad = 10; y = 28
    cv2.rectangle(frame, (pad-4, 6), (225, 122), (0,0,0), -1)
    cv2.rectangle(frame, (pad-4, 6), (225, 122), (50,50,50), 1)

    mode_col = {"DRAW": (57,255,20), "PAUSE": (0,200,255)}.get(mode, (120,120,120))
    put_shadow_text(frame, f"Mode : {mode}",         (pad, y), FONT_SM, mode_col);    y += 22
    put_shadow_text(frame, f"Color: {'ERASER' if eraser_mode else color_name}",
                            (pad, y), FONT_SM, (220,220,220));                         y += 22
    put_shadow_text(frame, f"Brush: {thick_name}",   (pad, y), FONT_SM, (220,220,220)); y += 22
    put_shadow_text(frame, f"FPS  : {fps:.0f}",      (pad, y), FONT_SM, (180,180,180))


def draw_connections(frame, lm_list, w, h):
    """Draw hand skeleton using connection pairs."""
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]
    for a, b in CONNECTIONS:
        pa = landmark_px(lm_list[a], w, h)
        pb = landmark_px(lm_list[b], w, h)
        cv2.line(frame, pa, pb, (60, 60, 60), 1, cv2.LINE_AA)
    for lm in lm_list:
        px, py = landmark_px(lm, w, h)
        cv2.circle(frame, (px, py), 3, (30, 30, 30), -1, cv2.LINE_AA)


def handle_action(action: str, state: dict, strokes: list, redo_stack: list):
    if action.startswith("color_"):
        state["color_idx"]   = int(action.split("_")[1])
        state["eraser_mode"] = False
        print(f"[Color] -> {COLOR_NAMES[state['color_idx']]}")
    elif action.startswith("thick_"):
        state["thick_idx"] = int(action.split("_")[1])
        print(f"[Thickness] -> {THICKNESS_NAMES[state['thick_idx']]}")
    elif action == "UNDO":
        if strokes:
            redo_stack.append(strokes.pop())
            print(f"[Undo] {len(strokes)} stroke(s) remain")
    elif action == "REDO":
        if redo_stack:
            strokes.append(redo_stack.pop())
            print(f"[Redo] {len(strokes)} stroke(s) now")
    elif action == "ERASER":
        state["eraser_mode"] = not state["eraser_mode"]
        print(f"[Eraser] {'ON' if state['eraser_mode'] else 'OFF'}")
    elif action == "CLEAR":
        strokes.clear()
        redo_stack.clear()
        print("[Clear] Canvas cleared")


# ── 5. MAIN LOOP ─────────────────────────────────────────────

def main():
    ensure_model()

    # ── Webcam init ──
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot open webcam.")
        return
    h, w = first_frame.shape[:2]

    # ── MediaPipe HandLandmarker (VIDEO mode = synchronous per-frame) ──
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ── App state ──
    state       = {"color_idx": 0, "thick_idx": 1, "eraser_mode": False}
    strokes     = []
    redo_stack  = []
    active_pts  = []
    canvas      = np.zeros((h, w, 3), dtype=np.uint8)
    prev_gesture = "NONE"
    prev_time    = time.time()
    fps          = 0.0
    dwell_state  = {"zone": None, "elapsed": 0.0}
    zones        = build_toolbar_zones(h, w)
    fingertip_pos = None
    frame_idx    = 0        # for VIDEO mode timestamp

    print("\nAir Draw started.")
    print("  [1 finger] INDEX FINGER only  -> DRAW mode")
    print("  [5 fingers] ALL FINGERS open  -> PAUSE / toolbar mode")
    print("  Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        frame_idx += 1

        # FPS
        now = time.time()
        dt  = max(now - prev_time, 1e-6)
        fps = 0.9 * fps + 0.1 / dt
        prev_time = now

        # ── Hand detection ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        # timestamp_ms must be monotonically increasing
        timestamp_ms = int(frame_idx * (1000 / 60))
        result: HandLandmarkerResult = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture       = "NONE"
        fingertip_pos = None
        lm_list       = None

        if result.hand_landmarks:
            lm_list = result.hand_landmarks[0]   # list of NormalizedLandmark
            draw_connections(frame, lm_list, w, h)
            gesture = get_gesture(lm_list)
            fx, fy  = landmark_px(lm_list[8], w, h)
            fingertip_pos = (fx, fy)

        # ── Current tool ──
        if state["eraser_mode"]:
            cur_color = ERASER_COLOR
            cur_thick = ERASER_THICKNESS
        else:
            cur_color = COLOR_VALUES[state["color_idx"]]
            cur_thick = THICKNESS_VALUES[state["thick_idx"]]
        cur_thick_name = THICKNESS_NAMES[state["thick_idx"]]

        # ── Gesture FSM ──
        if gesture == "DRAW" and fingertip_pos:
            if prev_gesture != "DRAW":
                active_pts = []
            active_pts.append(fingertip_pos)

        elif gesture == "PAUSE":
            # Finalize stroke
            if prev_gesture == "DRAW" and active_pts:
                strokes.append({"points": list(active_pts),
                                 "color": cur_color, "thickness": cur_thick})
                redo_stack.clear()
                canvas     = redraw_canvas(strokes, (h, w, 3))
                active_pts = []

            # Dwell toolbar
            if fingertip_pos:
                action, dwell_state = check_dwell(fingertip_pos, zones, dwell_state, dt)
                if action:
                    handle_action(action, state, strokes, redo_stack)
                    canvas     = redraw_canvas(strokes, (h, w, 3))
                    active_pts = []

        else:  # NONE
            if prev_gesture == "DRAW" and active_pts:
                strokes.append({"points": list(active_pts),
                                 "color": cur_color, "thickness": cur_thick})
                redo_stack.clear()
                canvas     = redraw_canvas(strokes, (h, w, 3))
                active_pts = []
            dwell_state.update(zone=None, elapsed=0.0)

        prev_gesture = gesture

        # ── Build preview canvas ──
        preview = canvas.copy()
        if len(active_pts) >= 2:
            for i in range(1, len(active_pts)):
                cv2.line(preview, active_pts[i-1], active_pts[i],
                         cur_color, cur_thick, cv2.LINE_AA)
        elif len(active_pts) == 1:
            cv2.circle(preview, active_pts[0], max(1, cur_thick//2), cur_color, -1)

        # ── Blend canvas onto frame ──
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask3   = cv2.merge([mask, mask, mask])
        blended = cv2.addWeighted(frame, 0.55, preview, 0.95, 0)
        frame   = np.where(mask3 > 0, blended, frame)

        # ── Toolbar ──
        zones = build_toolbar_zones(h, w)
        draw_toolbar(frame, state, zones)

        # ── Cursor dot ──
        if fingertip_pos:
            cv2.circle(frame, fingertip_pos, 8, cur_color, -1, cv2.LINE_AA)
            cv2.circle(frame, fingertip_pos, 8, (255, 255, 255), 1, cv2.LINE_AA)
            if gesture == "PAUSE" and dwell_state["zone"]:
                draw_dwell_arc(frame, fingertip_pos[0], fingertip_pos[1],
                               dwell_state["elapsed"])

        # ── HUD ──
        draw_hud(frame, gesture, COLOR_NAMES[state["color_idx"]],
                 cur_thick_name, fps, state["eraser_mode"])

        # ── Mode badge ──
        if gesture in ("DRAW", "PAUSE"):
            badge  = "DRAW MODE" if gesture == "DRAW" else "PAUSE/SELECT"
            bcol   = (57,255,20) if gesture == "DRAW" else (0,200,255)
            badge_x = w - TB_WIDTH - 10
            (tw, _), _ = cv2.getTextSize(badge, FONT, FONT_SM, 1)
            bx1, by1, bx2, by2 = badge_x-tw-14, 8, badge_x, 26
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,0,0), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcol,    1)
            put_shadow_text(frame, badge, (bx1+6, by2-4), FONT_SM, bcol)

        cv2.imshow("Air Draw  |  q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Air Draw closed.")


if __name__ == "__main__":
    main()

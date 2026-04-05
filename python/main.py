import sys
from pathlib import Path
from urllib.request import urlretrieve

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError:
    print("Missing dependency. Install with: pip install opencv-python numpy mediapipe")
    sys.exit(1)


WINDOW_NAME = "Cursed Technique: Sky Manipulation"

# camera and mesh params
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
MIRROR_INPUT = True
MESH_ROWS = 14
MESH_COLS = 18
POINT_GRAB_RADIUS = 18
POINT_DRAW_RADIUS = 3
MAX_NODE_DISPLACEMENT = 2200
INTERPOLATION_METHOD = cv2.INTER_CUBIC
LINE_THICKNESS = 1
SHOW_PINNED_POINTS = False
POINT_COLOR = (30, 200, 255)
ACTIVE_POINT_COLOR = (20, 80, 255)
PINNED_POINT_COLOR = (160, 160, 160)
MESH_COLOR = (80, 255, 80)
PIN_EDGES = False
SHOW_INTERACTION_OVERLAY = True
FULLSCREEN_WINDOW = True

# Cloth simulation parameters
PHYSICS_SUBSTEPS = 3
PHYSICS_TIMESTEP = 1.0
VELOCITY_DAMPING = 0.985
SPRING_STIFFNESS_STRUCT = 0.14
SPRING_STIFFNESS_DIAG = 0.11
ANCHOR_STIFFNESS = 0.0
DRAG_STIFFNESS = 0.52
REST_LENGTH_ADAPTATION = 0.08
DRAG_FOLLOW = 0.74
MIN_COMPRESSION_RATIO = 0.72
ENABLE_HAND_INPUT = True
HAND_MAX_NUM = 1
HAND_DETECT_CONFIDENCE = 0.55
HAND_TRACK_CONFIDENCE = 0.55
HAND_GRAB_RADIUS_SCALE = 1.4
HAND_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def build_base_grid(width: int, height: int, rows: int, cols: int) -> np.ndarray:
    xs = np.linspace(0, width - 1, cols, dtype=np.float32)
    ys = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.dstack((grid_x, grid_y))


def is_border_node(row: int, col: int, rows: int, cols: int) -> bool:
    return row == 0 or col == 0 or row == rows - 1 or col == cols - 1


def clamp_displacement(dx: float, dy: float) -> tuple[float, float]:
    if MAX_NODE_DISPLACEMENT <= 0:
        return dx, dy
    mag = float(np.hypot(dx, dy))
    if mag == 0.0 or mag <= MAX_NODE_DISPLACEMENT:
        return dx, dy
    scale = MAX_NODE_DISPLACEMENT / mag
    return dx * scale, dy * scale


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    clamped_x = float(max(0, min(width - 1, x)))
    clamped_y = float(max(0, min(height - 1, y)))
    return clamped_x, clamped_y


def reset_displacement(state: dict) -> None:
    base_grid = state["base_grid"]
    if base_grid is None:
        return

    state["pos_grid"] = state["base_grid"].copy()
    state["vel_grid"] = np.zeros_like(state["pos_grid"], dtype=np.float32)
    state["rest_h"] = np.linalg.norm(base_grid[:, 1:, :] - base_grid[:, :-1, :], axis=2, keepdims=True)
    state["rest_v"] = np.linalg.norm(base_grid[1:, :, :] - base_grid[:-1, :, :], axis=2, keepdims=True)
    state["rest_d1"] = np.linalg.norm(base_grid[1:, 1:, :] - base_grid[:-1, :-1, :], axis=2, keepdims=True)
    state["rest_d2"] = np.linalg.norm(base_grid[1:, :-1, :] - base_grid[:-1, 1:, :], axis=2, keepdims=True)
    state["drag_index"] = -1
    state["drag_target"] = None
    state["hand_grabbing"] = False
    state["hand_closed"] = False
    state["hand_cursor"] = None


def on_mouse(event, x, y, flags, userdata):
    del flags

    state = userdata
    base_grid = state["base_grid"]
    pos_grid = state["pos_grid"]
    if base_grid is None or pos_grid is None:
        return

    rows = state["rows"]
    cols = state["cols"]
    node_positions = pos_grid

    if event == cv2.EVENT_LBUTTONDOWN:
        best_idx = None
        best_dist = float("inf")

        for row in range(rows):
            for col in range(cols):
                px, py = node_positions[row, col]
                dist = float(np.hypot(x - px, y - py))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = (row, col)

        if best_idx is not None and best_dist <= POINT_GRAB_RADIUS:
            state["drag_index"] = best_idx
            width = state["width"]
            height = state["height"]
            px, py = clamp_point(x, y, width, height)
            state["drag_target"] = np.array([px, py], dtype=np.float32)

    elif event == cv2.EVENT_MOUSEMOVE and state["drag_index"] != -1:
        width = state["width"]
        height = state["height"]
        px, py = clamp_point(x, y, width, height)
        state["drag_target"] = np.array([px, py], dtype=np.float32)

    elif event == cv2.EVENT_LBUTTONUP:
        state["drag_index"] = -1
        state["drag_target"] = None


def try_start_drag(state: dict, x: float, y: float, radius: float) -> bool:
    pos_grid = state["pos_grid"]
    if pos_grid is None:
        return False

    rows = state["rows"]
    cols = state["cols"]
    best_idx = None
    best_dist = float("inf")

    for row in range(rows):
        for col in range(cols):
            px, py = pos_grid[row, col]
            dist = float(np.hypot(x - px, y - py))
            if dist < best_dist:
                best_dist = dist
                best_idx = (row, col)

    if best_idx is None or best_dist > radius:
        return False

    state["drag_index"] = best_idx
    width = state["width"]
    height = state["height"]
    tx, ty = clamp_point(int(x), int(y), width, height)
    state["drag_target"] = np.array([tx, ty], dtype=np.float32)
    return True


def set_drag_target(state: dict, x: float, y: float) -> None:
    width = state["width"]
    height = state["height"]
    tx, ty = clamp_point(int(x), int(y), width, height)
    state["drag_target"] = np.array([tx, ty], dtype=np.float32)


def _get_landmark_xy(hand_landmarks, index: int) -> np.ndarray:
    if hasattr(hand_landmarks, "landmark"):
        lm = hand_landmarks.landmark[index]
    else:
        lm = hand_landmarks[index]
    return np.array([lm.x, lm.y], dtype=np.float32)


def is_hand_closed(hand_landmarks) -> bool:
    wrist = _get_landmark_xy(hand_landmarks, 0)
    mcp = _get_landmark_xy(hand_landmarks, 9)
    palm_size = float(np.linalg.norm(mcp - wrist))
    if palm_size < 1e-6:
        return False

    tip_ids = [4, 8, 12, 16, 20]
    mcp_ids = [2, 5, 9, 13, 17]
    normalized = []
    for tip_id, mcp_id in zip(tip_ids, mcp_ids):
        tip = _get_landmark_xy(hand_landmarks, tip_id)
        base = _get_landmark_xy(hand_landmarks, mcp_id)
        normalized.append(float(np.linalg.norm(tip - base) / palm_size))

    mean_fold = float(np.mean(normalized))
    return mean_fold < 0.58


def hand_cursor_point(hand_landmarks, width: int, height: int) -> tuple[float, float]:
    # Cursor at palm center gives stable dragging when fist is closed.
    p0 = _get_landmark_xy(hand_landmarks, 0)
    p9 = _get_landmark_xy(hand_landmarks, 9)
    x = (p0[0] + p9[0]) * 0.5 * width
    y = (p0[1] + p9[1]) * 0.5 * height
    return x, y


def ensure_hand_model() -> bool:
    if HAND_MODEL_PATH.exists():
        return True
    try:
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(HAND_MODEL_URL, str(HAND_MODEL_PATH))
        return True
    except Exception as exc:
        print(f"Failed to download hand model: {exc}")
        return False


def create_hand_tracker():
    if not ENABLE_HAND_INPUT:
        return None, None

    if hasattr(mp, "solutions"):
        tracker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=HAND_MAX_NUM,
            min_detection_confidence=HAND_DETECT_CONFIDENCE,
            min_tracking_confidence=HAND_TRACK_CONFIDENCE,
        )
        return tracker, "solutions"

    if not ensure_hand_model():
        return None, None

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except Exception as exc:
        print(f"MediaPipe tasks API unavailable: {exc}")
        return None, None

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=HAND_MAX_NUM,
        min_hand_detection_confidence=HAND_DETECT_CONFIDENCE,
        min_tracking_confidence=HAND_TRACK_CONFIDENCE,
    )
    tracker = vision.HandLandmarker.create_from_options(options)
    return tracker, "tasks"


def update_hand_input(state: dict, frame: np.ndarray) -> None:
    if not ENABLE_HAND_INPUT or state["hand_tracker"] is None:
        return

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result_landmarks = None
    if state["hand_tracker_type"] == "solutions":
        result = state["hand_tracker"].process(rgb)
        result_landmarks = result.multi_hand_landmarks
    elif state["hand_tracker_type"] == "tasks":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = state["hand_tracker"].detect(mp_image)
        result_landmarks = result.hand_landmarks

    state["hand_cursor"] = None
    state["hand_closed"] = False

    if not result_landmarks:
        if state["hand_grabbing"]:
            state["hand_grabbing"] = False
            state["drag_index"] = -1
            state["drag_target"] = None
        return

    hand_landmarks = result_landmarks[0]
    hx, hy = hand_cursor_point(hand_landmarks, width, height)
    state["hand_cursor"] = (hx, hy)
    closed = is_hand_closed(hand_landmarks)
    state["hand_closed"] = closed

    if closed:
        if not state["hand_grabbing"]:
            grabbed = try_start_drag(state, hx, hy, POINT_GRAB_RADIUS * HAND_GRAB_RADIUS_SCALE)
            state["hand_grabbing"] = grabbed
        if state["hand_grabbing"] and state["drag_index"] != -1:
            set_drag_target(state, hx, hy)
    else:
        if state["hand_grabbing"]:
            state["hand_grabbing"] = False
            state["drag_index"] = -1
            state["drag_target"] = None


def apply_spring(forces: np.ndarray, pos_grid: np.ndarray, p1, p2, rest_length, stiffness: float) -> None:
    delta = p2 - p1
    dist = np.linalg.norm(delta, axis=2, keepdims=True)
    dist_safe = np.maximum(dist, 1e-6)
    direction = delta / dist_safe
    extension = dist - rest_length
    spring_force = stiffness * extension * direction
    forces_slice_1, forces_slice_2 = forces
    forces_slice_1 += spring_force
    forces_slice_2 -= spring_force


def enforce_pinned_edges(state: dict) -> None:
    pos_grid = state["pos_grid"]
    vel_grid = state["vel_grid"]
    base_grid = state["base_grid"]
    pos_grid[0, :, :] = base_grid[0, :, :]
    pos_grid[-1, :, :] = base_grid[-1, :, :]
    pos_grid[:, 0, :] = base_grid[:, 0, :]
    pos_grid[:, -1, :] = base_grid[:, -1, :]
    vel_grid[0, :, :] = 0
    vel_grid[-1, :, :] = 0
    vel_grid[:, 0, :] = 0
    vel_grid[:, -1, :] = 0


def clamp_displacements_from_base(state: dict) -> None:
    if MAX_NODE_DISPLACEMENT <= 0:
        return
    pos_grid = state["pos_grid"]
    base_grid = state["base_grid"]
    disp = pos_grid - base_grid
    mag = np.linalg.norm(disp, axis=2)
    mask = mag > MAX_NODE_DISPLACEMENT
    if not np.any(mask):
        return
    scale = (MAX_NODE_DISPLACEMENT / np.maximum(mag[mask], 1e-6)).reshape(-1, 1)
    disp[mask] *= scale
    pos_grid[mask] = base_grid[mask] + disp[mask]


def enforce_min_neighbor_distance(state: dict) -> None:
    if MIN_COMPRESSION_RATIO <= 0:
        return

    pos_grid = state["pos_grid"]
    min_h = state["rest_h"] * MIN_COMPRESSION_RATIO
    min_v = state["rest_v"] * MIN_COMPRESSION_RATIO
    min_d1 = state["rest_d1"] * MIN_COMPRESSION_RATIO
    min_d2 = state["rest_d2"] * MIN_COMPRESSION_RATIO

    def solve_pair(p1: np.ndarray, p2: np.ndarray, min_len: np.ndarray) -> None:
        delta = p2 - p1
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        need = dist < min_len
        if not np.any(need):
            return
        direction = delta / np.maximum(dist, 1e-6)
        correction = 0.5 * (min_len - dist) * direction
        correction *= need.astype(np.float32)
        p1 -= correction
        p2 += correction

    solve_pair(pos_grid[:, :-1, :], pos_grid[:, 1:, :], min_h)
    solve_pair(pos_grid[:-1, :, :], pos_grid[1:, :, :], min_v)
    solve_pair(pos_grid[:-1, :-1, :], pos_grid[1:, 1:, :], min_d1)
    solve_pair(pos_grid[:-1, 1:, :], pos_grid[1:, :-1, :], min_d2)


def step_cloth_simulation(state: dict) -> None:
    pos_grid = state["pos_grid"]
    vel_grid = state["vel_grid"]
    base_grid = state["base_grid"]

    dt = PHYSICS_TIMESTEP / max(PHYSICS_SUBSTEPS, 1)

    for _ in range(PHYSICS_SUBSTEPS):
        forces = np.zeros_like(pos_grid, dtype=np.float32)

        left = pos_grid[:, :-1, :]
        right = pos_grid[:, 1:, :]
        rest_h = state["rest_h"]
        delta = right - left
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        direction = delta / np.maximum(dist, 1e-6)
        ext = dist - rest_h
        f = SPRING_STIFFNESS_STRUCT * ext * direction
        forces[:, :-1, :] += f
        forces[:, 1:, :] -= f

        top = pos_grid[:-1, :, :]
        bottom = pos_grid[1:, :, :]
        rest_v = state["rest_v"]
        delta = bottom - top
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        direction = delta / np.maximum(dist, 1e-6)
        ext = dist - rest_v
        f = SPRING_STIFFNESS_STRUCT * ext * direction
        forces[:-1, :, :] += f
        forces[1:, :, :] -= f

        diag_tl = pos_grid[:-1, :-1, :]
        diag_br = pos_grid[1:, 1:, :]
        rest_d1 = state["rest_d1"]
        delta = diag_br - diag_tl
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        direction = delta / np.maximum(dist, 1e-6)
        ext = dist - rest_d1
        f = SPRING_STIFFNESS_DIAG * ext * direction
        forces[:-1, :-1, :] += f
        forces[1:, 1:, :] -= f

        diag_tr = pos_grid[:-1, 1:, :]
        diag_bl = pos_grid[1:, :-1, :]
        rest_d2 = state["rest_d2"]
        delta = diag_bl - diag_tr
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        direction = delta / np.maximum(dist, 1e-6)
        ext = dist - rest_d2
        f = SPRING_STIFFNESS_DIAG * ext * direction
        forces[:-1, 1:, :] += f
        forces[1:, :-1, :] -= f

        forces += ANCHOR_STIFFNESS * (base_grid - pos_grid)

        if state["drag_index"] != -1 and state["drag_target"] is not None:
            row, col = state["drag_index"]
            target = state["drag_target"]
            forces[row, col] += DRAG_STIFFNESS * (target - pos_grid[row, col])

        vel_grid += forces * dt
        vel_grid *= VELOCITY_DAMPING
        pos_grid += vel_grid * dt

        if state["drag_index"] != -1 and state["drag_target"] is not None:
            row, col = state["drag_index"]
            target = state["drag_target"]
            pos_grid[row, col] = (1.0 - DRAG_FOLLOW) * pos_grid[row, col] + DRAG_FOLLOW * target
            vel_grid[row, col] *= 0.35

        if REST_LENGTH_ADAPTATION > 0:
            current_h = np.linalg.norm(pos_grid[:, 1:, :] - pos_grid[:, :-1, :], axis=2, keepdims=True)
            current_v = np.linalg.norm(pos_grid[1:, :, :] - pos_grid[:-1, :, :], axis=2, keepdims=True)
            current_d1 = np.linalg.norm(pos_grid[1:, 1:, :] - pos_grid[:-1, :-1, :], axis=2, keepdims=True)
            current_d2 = np.linalg.norm(pos_grid[1:, :-1, :] - pos_grid[:-1, 1:, :], axis=2, keepdims=True)
            a = np.clip(REST_LENGTH_ADAPTATION * dt, 0.0, 1.0)
            state["rest_h"] = (1.0 - a) * state["rest_h"] + a * current_h
            state["rest_v"] = (1.0 - a) * state["rest_v"] + a * current_v
            state["rest_d1"] = (1.0 - a) * state["rest_d1"] + a * current_d1
            state["rest_d2"] = (1.0 - a) * state["rest_d2"] + a * current_d2

        enforce_min_neighbor_distance(state)
        clamp_displacements_from_base(state)
        if PIN_EDGES:
            enforce_pinned_edges(state)


def draw_controls(frame: np.ndarray, node_positions: np.ndarray, active_index) -> None:
    rows, cols = node_positions.shape[:2]

    for row in range(rows):
        for col in range(cols - 1):
            p1 = node_positions[row, col].astype(np.int32)
            p2 = node_positions[row, col + 1].astype(np.int32)
            cv2.line(frame, tuple(p1), tuple(p2), MESH_COLOR, LINE_THICKNESS)

    for row in range(rows - 1):
        for col in range(cols):
            p1 = node_positions[row, col].astype(np.int32)
            p2 = node_positions[row + 1, col].astype(np.int32)
            cv2.line(frame, tuple(p1), tuple(p2), MESH_COLOR, LINE_THICKNESS)

    for row in range(rows):
        for col in range(cols):
            border = is_border_node(row, col, rows, cols)
            if border and not SHOW_PINNED_POINTS:
                continue

            px, py = node_positions[row, col].astype(np.int32)
            is_active = active_index != -1 and active_index == (row, col)
            if is_active:
                color = ACTIVE_POINT_COLOR
            elif border:
                color = PINNED_POINT_COLOR
            else:
                color = POINT_COLOR
            cv2.circle(frame, (int(px), int(py)), POINT_DRAW_RADIUS, color, -1)


def draw_hand_status(frame: np.ndarray, state: dict) -> None:
    cursor = state.get("hand_cursor")
    if cursor is None:
        return

    x, y = int(cursor[0]), int(cursor[1])
    if state.get("hand_closed", False):
        color = (0, 80, 255)
    else:
        color = (0, 220, 120)
    cv2.circle(frame, (x, y), 9, color, 2)


def ensure_mesh_initialized(state: dict, width: int, height: int) -> None:
    if state["base_grid"] is not None and state["width"] == width and state["height"] == height:
        return

    state["base_grid"] = build_base_grid(width, height, state["rows"], state["cols"])
    state["x_coords"] = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    state["y_coords"] = np.tile(np.arange(height, dtype=np.float32).reshape(-1, 1), (1, width))
    base_grid = state["base_grid"]
    state["rest_h"] = np.linalg.norm(base_grid[:, 1:, :] - base_grid[:, :-1, :], axis=2, keepdims=True)
    state["rest_v"] = np.linalg.norm(base_grid[1:, :, :] - base_grid[:-1, :, :], axis=2, keepdims=True)
    state["rest_d1"] = np.linalg.norm(base_grid[1:, 1:, :] - base_grid[:-1, :-1, :], axis=2, keepdims=True)
    state["rest_d2"] = np.linalg.norm(base_grid[1:, :-1, :] - base_grid[:-1, 1:, :], axis=2, keepdims=True)
    state["width"] = width
    state["height"] = height
    reset_displacement(state)
    state["drag_index"] = -1
    state["drag_target"] = None


def build_warped_frame(frame: np.ndarray, state: dict) -> tuple[np.ndarray, np.ndarray]:
    step_cloth_simulation(state)
    disp_grid = state["pos_grid"] - state["base_grid"]

    disp_x = cv2.resize(disp_grid[:, :, 0], (state["width"], state["height"]), interpolation=INTERPOLATION_METHOD)
    disp_y = cv2.resize(disp_grid[:, :, 1], (state["width"], state["height"]), interpolation=INTERPOLATION_METHOD)

    map_x = state["x_coords"] - disp_x
    map_y = state["y_coords"] - disp_y
    np.clip(map_x, 0, state["width"] - 1, out=map_x)
    np.clip(map_y, 0, state["height"] - 1, out=map_y)

    warped = cv2.remap(
        frame,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    node_positions = state["pos_grid"]
    return warped, node_positions


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    hand_tracker, hand_tracker_type = create_hand_tracker()

    state = {
        "width": 0,
        "height": 0,
        "rows": MESH_ROWS,
        "cols": MESH_COLS,
        "base_grid": None,
        "pos_grid": None,
        "vel_grid": None,
        "x_coords": None,
        "y_coords": None,
        "rest_h": None,
        "rest_v": None,
        "rest_d1": None,
        "rest_d2": None,
        "drag_index": -1,
        "drag_target": None,
        "hand_tracker": hand_tracker,
        "hand_tracker_type": hand_tracker_type,
        "hand_grabbing": False,
        "hand_closed": False,
        "hand_cursor": None,
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN_WINDOW:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)
    show_overlay = SHOW_INTERACTION_OVERLAY

    print("Camera started. Drag mesh points with mouse or closed hand.")
    if ENABLE_HAND_INPUT and hand_tracker is None:
        print("Hand tracking unavailable. Mouse drag still works.")
    if PIN_EDGES:
        print("Edges are pinned.")
    else:
        print("Weightless mode.")
    print("Keybinds: t toggle overlay | r reset cloth | q/esc quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        if MIRROR_INPUT:
            frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        ensure_mesh_initialized(state, width, height)
        update_hand_input(state, frame)
        warped, node_positions = build_warped_frame(frame, state)

        if show_overlay:
            draw_controls(warped, node_positions, state["drag_index"])
        if ENABLE_HAND_INPUT and show_overlay:
            draw_hand_status(warped, state)

        cv2.imshow(WINDOW_NAME, warped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("t"):
            show_overlay = not show_overlay
            print(f"Overlay: {'on' if show_overlay else 'off'}")
        if key == ord("r"):
            reset_displacement(state)

    cap.release()
    if state["hand_tracker"] is not None:
        state["hand_tracker"].close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

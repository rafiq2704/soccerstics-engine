"""Soccerstics — Instep Drive analysis engine.

Pipeline
--------
1. Read the video frame-by-frame (OpenCV).
2. Player kinematics: YOLO11 pose -> hip/knee/ankle/shoulder keypoints.
3. Ball: YOLO11 detection with SAHI slicing (small, fast object), then a
   constant-velocity Kalman filter to fill occluded frames.
4. Scale: homography (4 points) or a 2-point known distance -> pixels/metre.
5. Metrics: contact frame, ball speed, launch angle, knee angular velocity,
   foot speed, plant-foot distance, hip extension.
6. Emit JSON matching  soccerstics.instep_drive.v1  (the dashboard contract).

The engine is written to be host-agnostic: predict.py wraps it for Replicate,
but it can be run anywhere Python + a GPU are available.
"""
import math
import numpy as np
import cv2

# COCO-17 keypoint indices (ultralytics pose output order)
KP = {
    "nose": 0, "l_shoulder": 5, "r_shoulder": 6, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _angle(a, b, c):
    """Interior angle at point b (degrees), formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _extrapolate_foot(knee, ankle, factor=0.35):
    """COCO has no toe keypoint; estimate the foot tip by extending the
    shank a little past the ankle. Purely for a natural-looking overlay."""
    kx, ky = knee
    ax, ay = ankle
    return [ax + (ax - kx) * factor, ay + (ay - ky) * factor]


# --------------------------------------------------------------------------
# Calibration -> pixels per metre
# --------------------------------------------------------------------------
def build_scale(calibration, frame_w, frame_h):
    """Return (px_per_m, homography_or_None, method, confidence).

    calibration accepts either:
      - {"points_px": [[x,y]*4], "points_m": [[X,Y]*4]}  full homography
      - {"points_px": [[x,y],[x,y]], "distance_m": 5.0}   2-point scale
    Falls back to a rough default if nothing usable is supplied.
    """
    if calibration and "points_m" in calibration and len(calibration.get("points_px", [])) >= 4:
        src = np.array(calibration["points_px"][:4], dtype=np.float32)
        dst = np.array(calibration["points_m"][:4], dtype=np.float32)
        Hmat, _ = cv2.findHomography(src, dst)
        # px_per_m near image centre, for display only
        p0 = _to_world(Hmat, [frame_w / 2, frame_h / 2])
        p1 = _to_world(Hmat, [frame_w / 2 + 100, frame_h / 2])
        ppm = 100.0 / (abs(p1[0] - p0[0]) + 1e-9)
        return ppm, Hmat, "homography-4pt", 0.9

    if calibration and "distance_m" in calibration and len(calibration.get("points_px", [])) >= 2:
        a, b = calibration["points_px"][0], calibration["points_px"][1]
        ppm = _dist(a, b) / max(calibration["distance_m"], 1e-6)
        return ppm, None, "cones-%gm" % calibration["distance_m"], 0.7

    # No calibration: assume a player ~1.6 m tall fills ~55% of frame height.
    ppm = (frame_h * 0.55) / 1.6
    return ppm, None, "uncalibrated-estimate", 0.3


def _scale_calibration(calibration, scale):
    """Calibration pixel points arrive in native resolution; scale them to
    the processing resolution so they line up with the detected coords."""
    if not calibration or scale == 1.0:
        return calibration
    cal = dict(calibration)
    if "points_px" in cal:
        cal["points_px"] = [[p[0] * scale, p[1] * scale] for p in cal["points_px"]]
    return cal


def _to_world(Hmat, pt):
    v = np.array([pt[0], pt[1], 1.0])
    w = Hmat @ v
    return [w[0] / w[2], w[1] / w[2]]


def _to_metres(pt, ppm, Hmat):
    if Hmat is not None:
        return _to_world(Hmat, pt)
    return [pt[0] / ppm, pt[1] / ppm]


# --------------------------------------------------------------------------
# Detection passes
# --------------------------------------------------------------------------
def run_pose(pose_model, frame, foot):
    """Return dict of pixel keypoints for the kicking side, or None."""
    res = pose_model(frame, verbose=False)
    if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
        return None
    r = res[0]
    # pick the largest detected person (nearest to camera)
    boxes = r.boxes.xywh.cpu().numpy() if r.boxes is not None else None
    idx = 0
    if boxes is not None and len(boxes) > 1:
        idx = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
    kp = r.keypoints.xy.cpu().numpy()[idx]  # (17,2)

    side = "r" if foot == "right" else "l"
    def g(name):
        i = KP[name]
        x, y = kp[i]
        if x == 0 and y == 0:
            return None
        return [float(x), float(y)]

    hip = g(f"{side}_hip"); knee = g(f"{side}_knee")
    ankle = g(f"{side}_ankle"); shoulder = g(f"{side}_shoulder")
    if not (hip and knee and ankle):
        return None
    foot_pt = _extrapolate_foot(knee, ankle)
    plant = g("l_ankle" if side == "r" else "r_ankle")
    return {"shoulder": shoulder or [hip[0], hip[1] - 60],
            "hip": hip, "knee": knee, "ankle": ankle, "foot": foot_pt,
            "plant_ankle": plant, "knee_angle": _angle(hip, knee, ankle)}


def run_ball(models, frame):
    """Fast path first: full-frame YOLO. Only fall back to slow SAHI slicing
    on frames where the ball is missed. This is the main speed win — most
    frames resolve on the fast path and never touch slicing.

    `models` is a tuple: (fast_yolo_model, sahi_model).
    """
    fast_model, sahi_model = models

    # --- fast path: plain full-frame detection ---
    res = fast_model(frame, verbose=False, classes=[32])  # 32 = sports ball
    best, best_score = None, 0.0
    if res and res[0].boxes is not None and len(res[0].boxes) > 0:
        b = res[0].boxes
        xywh = b.xywh.cpu().numpy()
        conf = b.conf.cpu().numpy()
        for (cx, cy, w, h), sc in zip(xywh, conf):
            if sc > best_score:
                best, best_score = [float(cx), float(cy)], float(sc)
    if best is not None:
        return best

    # --- slow path: SAHI slicing, only when the ball was missed ---
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        frame, sahi_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    best, best_score = None, 0.0
    for o in result.object_prediction_list:
        name = (o.category.name or "").lower()
        if name in ("sports ball", "ball") and o.score.value > best_score:
            box = o.bbox
            cx = (box.minx + box.maxx) / 2.0
            cy = (box.miny + box.maxy) / 2.0
            best, best_score = [cx, cy], o.score.value
    return best


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------
def find_contact(frames):
    """Contact = frame where the kicking foot is closest to the ball."""
    best_i, best_d = None, 1e18
    for i, f in enumerate(frames):
        if f["ball"] is None or f["pose"] is None:
            continue
        d = _dist(f["pose"]["ankle"], f["ball"])
        if d < best_d:
            best_d, best_i = d, i
    return best_i if best_i is not None else len(frames) // 2


def compute_metrics(frames, fps, ppm, Hmat, contact_i):
    ms = []

    def add(key, label, val, unit, conf, sub):
        if val is None:
            return
        ms.append({"key": key, "label": label, "value": round(float(val), 1),
                   "unit": unit, "confidence": conf, "sub": sub})

    # ---- Ball speed: displacement over the 3 frames after contact ----
    ball_speed = None
    post = [f for f in frames[contact_i + 1: contact_i + 5] if f["ball"] and not f["ball_predicted"]]
    if len(post) >= 2:
        p0 = _to_metres(post[0]["ball"], ppm, Hmat)
        p1 = _to_metres(post[-1]["ball"], ppm, Hmat)
        dt = (post[-1]["t"] - post[0]["t"]) or (1.0 / fps)
        mps = _dist(p0, p1) / dt
        ball_speed = mps * 3.6
    add("ball_speed", "Ball speed", ball_speed, "km/h", "high", "peak post-contact")

    # ---- Launch angle: image-plane trajectory just after contact ----
    launch = None
    if len(post) >= 2:
        dx = post[-1]["ball"][0] - post[0]["ball"][0]
        dy = post[0]["ball"][1] - post[-1]["ball"][1]  # image y is down
        launch = abs(math.degrees(math.atan2(dy, abs(dx) + 1e-6)))
    add("launch_angle", "Launch angle", launch, "\u00b0", "high", "above ground plane")

    # ---- Knee angular velocity: peak d(angle)/dt through the swing ----
    peak_kav = None
    for i in range(1, len(frames)):
        a, b = frames[i - 1]["pose"], frames[i]["pose"]
        if not a or not b:
            continue
        dt = (frames[i]["t"] - frames[i - 1]["t"]) or (1.0 / fps)
        kav = abs(b["knee_angle"] - a["knee_angle"]) / dt
        if peak_kav is None or kav > peak_kav:
            peak_kav = kav
    add("knee_ang_vel", "Knee angular velocity", peak_kav, "\u00b0/s", "high", "peak during swing")

    # ---- Foot (ankle) speed around contact ----
    foot_speed = None
    lo, hi = max(0, contact_i - 1), min(len(frames) - 1, contact_i + 1)
    if frames[lo]["pose"] and frames[hi]["pose"] and hi > lo:
        p0 = _to_metres(frames[lo]["pose"]["ankle"], ppm, Hmat)
        p1 = _to_metres(frames[hi]["pose"]["ankle"], ppm, Hmat)
        dt = (frames[hi]["t"] - frames[lo]["t"]) or (1.0 / fps)
        foot_speed = _dist(p0, p1) / dt * 3.6
    add("foot_speed", "Foot speed at contact", foot_speed, "km/h", "medium", "ankle marker")

    # ---- Plant-foot distance from ball at contact ----
    plant_d = None
    cf = frames[contact_i]
    if cf["pose"] and cf["pose"].get("plant_ankle") and cf["ball"]:
        p = _to_metres(cf["pose"]["plant_ankle"], ppm, Hmat)
        b = _to_metres(cf["ball"], ppm, Hmat)
        plant_d = _dist(p, b) * 100.0  # cm
    add("plant_dist", "Plant foot distance", plant_d, "cm", "medium", "beside the ball")

    # ---- Hip extension (single-camera -> low confidence) ----
    hip_rom = None
    angs = []
    for f in frames:
        p = f["pose"]
        if p and p["shoulder"] and p["hip"] and p["knee"]:
            angs.append(_angle(p["shoulder"], p["hip"], p["knee"]))
    if angs:
        hip_rom = max(angs) - min(angs)
    add("hip_rom", "Hip extension", hip_rom, "\u00b0", "low", "single-camera estimate")

    return ms


def segment_phases(frames, fps, contact_i):
    """Simple heuristic phase split around the detected contact frame."""
    dur = frames[-1]["t"] if frames else 0.0
    ct = frames[contact_i]["t"] if frames else 0.0
    return [
        {"name": "Approach", "t0": 0.0, "t1": max(0.0, ct - 0.30)},
        {"name": "Plant", "t0": max(0.0, ct - 0.30), "t1": max(0.0, ct - 0.12)},
        {"name": "Swing", "t0": max(0.0, ct - 0.12), "t1": max(0.0, ct - 0.01)},
        {"name": "Contact", "t0": max(0.0, ct - 0.01), "t1": ct + 0.02},
        {"name": "Follow-through", "t0": ct + 0.02, "t1": dur},
    ]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def analyse(video_path, pose_model, ball_models, session, calibration=None,
            max_frames=240):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    foot = (session or {}).get("foot", "right")

    # Downscale huge frames (e.g. 4K/8K phone video) before detection.
    # Detectors shrink images to ~640px internally anyway, so processing at
    # native 8K is pure wasted work — and it makes SAHI slicing explode.
    # We process at PROC_W wide, then scale coordinates back to native W/H.
    PROC_W = 1280
    proc_scale = 1.0
    if W > PROC_W:
        proc_scale = PROC_W / float(W)
    procW = int(W * proc_scale)
    procH = int(H * proc_scale)

    ppm, Hmat, cal_method, cal_conf = build_scale(
        _scale_calibration(calibration, proc_scale), procW, procH)
    from kalman import BallKalman
    kf = BallKalman()

    frames, i = [], 0
    import time as _time
    _t0 = _time.time()
    print(f"[soccerstics] start: fps={fps} size={W}x{H} foot={foot} max_frames={max_frames}", flush=True)
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if proc_scale != 1.0:
            frame = cv2.resize(frame, (procW, procH), interpolation=cv2.INTER_AREA)
        t = i / fps
        pose = run_pose(pose_model, frame, foot)
        raw_ball = run_ball(ball_models, frame)
        ball_xy, predicted = kf.step(raw_ball)

        frames.append({
            "t": round(t, 4),
            "pose": pose,
            "ball": [round(ball_xy[0], 2), round(ball_xy[1], 2)] if ball_xy else None,
            "ball_predicted": bool(predicted),
            "knee_angle": round(pose["knee_angle"], 1) if pose else None,
        })
        if i % 5 == 0:
            print(f"[soccerstics] frame {i}/{max_frames}  "
                  f"pose={'y' if pose else 'n'} ball={'y' if raw_ball else 'n'}  "
                  f"elapsed={_time.time()-_t0:.1f}s", flush=True)
        i += 1
    cap.release()
    print(f"[soccerstics] processed {len(frames)} frames in {_time.time()-_t0:.1f}s", flush=True)

    contact_i = find_contact(frames)
    metrics = compute_metrics(frames, fps, ppm, Hmat, contact_i)
    phases = segment_phases(frames, fps, contact_i)

    # normalise coords 0..1 for the dashboard overlay.
    # Detection ran at proc size, so normalise by proc size (ratios are
    # identical to native, and the dashboard only needs 0..1).
    def norm(pt):
        return [round(pt[0] / procW, 4), round(pt[1] / procH, 4)] if pt else None
    out_frames = []
    for f in frames:
        p = f["pose"]
        pose_n = None
        if p:
            pose_n = {k: norm(p[k]) for k in ("shoulder", "hip", "knee", "ankle", "foot")}
        out_frames.append({
            "t": f["t"],
            "pose": pose_n,
            "ball": norm(f["ball"]),
            "ball_predicted": f["ball_predicted"],
            "knee_angle": f["knee_angle"] if f["knee_angle"] is not None else 0,
        })

    return {
        "schema": "soccerstics.instep_drive.v1",
        "session": {
            "player": (session or {}).get("player", "Player"),
            "foot": foot,
            "date": (session or {}).get("date", ""),
            "notes": (session or {}).get("notes", ""),
        },
        "video": {"width": W, "height": H, "fps": round(fps, 2), "url": ""},
        "calibration": {"px_per_m": round(ppm, 1), "method": cal_method, "confidence": cal_conf},
        "phases": phases,
        "contact": {"t": frames[contact_i]["t"] if frames else 0.0, "frame": contact_i},
        "metrics": metrics,
        "frames": out_frames,
    }

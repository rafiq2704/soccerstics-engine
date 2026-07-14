"""Replicate entry point (Cog Predictor).

setup()   — loads YOLO11 weights once when the container starts.
predict() — runs one clip through the engine and returns metrics.json.

The dashboard's "Load a metrics.json" button reads this file directly, and
your Cloudflare Pages Function will POST a video URL here and store the
returned JSON in Firestore/Supabase.
"""
import json
import tempfile
from cog import BasePredictor, Input, Path

from ultralytics import YOLO
from sahi import AutoDetectionModel
import engine


class Predictor(BasePredictor):
    def setup(self):
        # Pose model (kinematics) — nano for fast load + inference on T4.
        self.pose_model = YOLO("yolo11n-pose.pt")
        # Fast full-frame detector (the ball, fast path).
        self.fast_model = YOLO("yolo11n.pt")
        # SAHI sliced detector (the ball, slow fallback only when missed).
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolo11n.pt",
            confidence_threshold=0.25,
            device="cuda:0",
        )

    def predict(
        self,
        video: Path = Input(description="Drill clip (instep drive), filmed side-on."),
        player: str = Input(description="Player name", default="Player"),
        foot: str = Input(description="Kicking foot", choices=["right", "left"], default="right"),
        date: str = Input(description="Session date (ISO)", default=""),
        calibration_json: str = Input(
            description=(
                "Optional JSON. Either 4-point homography "
                '{"points_px":[[x,y]*4],"points_m":[[X,Y]*4]} or a 2-cone scale '
                '{"points_px":[[x,y],[x,y]],"distance_m":5}.'
            ),
            default="",
        ),
        max_frames: int = Input(description="Max frames to analyse", default=240, ge=30, le=600),
    ) -> Path:
        calibration = None
        if calibration_json.strip():
            try:
                calibration = json.loads(calibration_json)
            except json.JSONDecodeError:
                calibration = None

        result = engine.analyse(
            video_path=str(video),
            pose_model=self.pose_model,
            ball_models=(self.fast_model, self.sahi_model),
            session={"player": player, "foot": foot, "date": date},
            calibration=calibration,
            max_frames=max_frames,
        )

        out = Path(tempfile.mkdtemp()) / "metrics.json"
        with open(out, "w") as f:
            json.dump(result, f)
        return out

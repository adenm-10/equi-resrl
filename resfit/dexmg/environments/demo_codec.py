"""The demos' video codec, applied to one camera frame from the env.

The demo frames reach the agent through a lossy video codec and the env's frames do not. On
Can's front camera that leaves every env frame with about twice the demos' colour (chroma)
detail and an even colour offset of about 0.005: each is a perfect separator of demo from env
frames (EXPERIMENTS.md, 2026-10-08 FINDING). `encode_env_frames_like_demos` passes the env's
frames through the same codec, so that cue is gone from rollouts, evals and the online buffer.

The settings are LeRobot's `encode_video_frames` defaults, which wrote the robomimic image
datasets: libsvtav1 at crf 30, yuv420p, a keyframe every second frame (ffprobe on the dataset's
videos: AV1 Main, yuv420p, keyframes at frames 0, 2, 4, ..., each followed by one small P-frame).
Each frame is coded as a two-frame clip of itself and the keyframe is decoded: svt-av1 spends
more bits on a keyframe that a later frame refers to, so a one-frame clip codes it more coarsely
than the demos' keyframes are (measured on 40 Can resets, 2026-10-08). The decode is PyAV's,
which gives the dataset loader's demo frames bit for bit.
"""

import io
import os

import av
import numpy as np
from PIL import Image

# svt-av1 prints its configuration on every encoder it opens, which is two per frame here.
# Level 1 keeps its errors. It reads the variable when an encoder opens, so this covers the env
# workers, which import this module before they render.
os.environ.setdefault("SVT_LOG", "1")

VCODEC = "libsvtav1"
PIX_FMT = "yuv420p"
GOP = 2
CRF = 30
FPS = 20


def encode_like_demos(frame: np.ndarray) -> np.ndarray:
    """`frame` (H, W, 3) uint8, as the dataset's videos would hold it: same shape and dtype."""
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3, (frame.dtype, frame.shape)
    height, width = frame.shape[:2]
    clip = io.BytesIO()
    with av.open(clip, "w", format="mp4") as out:
        stream = out.add_stream(VCODEC, FPS, options={"g": str(GOP), "crf": str(CRF)})
        stream.pix_fmt, stream.width, stream.height = PIX_FMT, width, height
        picture = av.VideoFrame.from_image(Image.fromarray(frame, "RGB"))
        for _ in range(2):  # the keyframe and the P-frame that refers to it
            for packet in stream.encode(picture):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    clip.seek(0)
    with av.open(clip, "r") as inp:
        return next(inp.decode(video=0)).to_ndarray(format="rgb24")

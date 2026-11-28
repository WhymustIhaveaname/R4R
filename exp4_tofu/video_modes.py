"""Step 2: the modes of a real Jello cube, from Visual Vibration Tomography's 4000 fps videos.

Feng, Ogren, Daraio and Bouman (CVPR 2022) filmed a Jello cube, 4.9 x 4.7 x 4.5 cm on a fixed
base with a clay block inside, after plucking or twisting it (NOTES.md).  The videos are raw
BGR24, about 280 x 290, captured at 4000 fps (the file says 60).  Here:

  1. every 4 frames are averaged, to 1000 fps;
  2. Farneback's optical flow gives each frame's displacement from the clip's mean frame, read on
     a grid of points inside the cube (the polygon of the clicked corners, clicks.npy);
  3. the free decay starts where the motion is largest, at the release;
  4. the largest temporal components of the points' motion (SVD) go into a matrix pencil, whose
     poles z = exp((-s + i w_d) dt) give each mode's natural frequency w_n = |s + i w_d| and
     damping ratio zeta = s / w_n, as common.measure defines them.

  python video_modes.py /data/datasets/vvt_jello_cube
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 4000
AVERAGE = 4
CLIPS = ("top_front_pluck", "top_right_pluck", "left_side_twist")


def size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    w, h = out.strip().split(",")
    return int(w), int(h)


def frames(path):
    """The clip's frames in grey, averaged AVERAGE at a time (float32, T x H x W)."""
    W, H = size(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    n = len(v) // AVERAGE * AVERAGE
    return v[:n].reshape(-1, AVERAGE, H, W).mean(1, dtype=np.float32)


def motion(video, clicks, spacing=6):
    """Displacement (pixels) of a grid of points inside the cube: (2 * points, T)."""
    H, W = video.shape[1:]
    hull = cv2.convexHull(clicks.astype(np.float32))
    ys, xs = np.mgrid[spacing:H - spacing:spacing, spacing:W - spacing:spacing]
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) > spacing
                       for x, y in zip(xs.ravel(), ys.ravel())])
    px, py = xs.ravel()[inside], ys.ravel()[inside]
    ref = video.mean(0)
    ref8 = np.clip(ref, 0, 255).astype(np.uint8)
    out = np.empty((2 * len(px), len(video)), np.float32)
    for t, f in enumerate(video):
        flow = cv2.calcOpticalFlowFarneback(ref8, np.clip(f, 0, 255).astype(np.uint8), None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        out[:len(px), t] = flow[py, px, 0]
        out[len(px):, t] = flow[py, px, 1]
    return out, px, py


def matrix_pencil(y, dt, order, L=None):
    """Poles of the sum of damped exponentials in the rows of y (components x time)."""
    T = y.shape[1]
    L = L or T // 3
    blocks = [np.stack([row[i:i + L + 1] for i in range(T - L)]) for row in y]
    Y = np.concatenate(blocks, 0)                        # (components * (T - L)) x (L + 1)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    V = Vt[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    s = np.log(z) / dt
    return -s.real, s.imag, S


def modes(motion_xy, dt, start, components=8, order=24, fmax=60.0):
    """(f_n Hz, zeta, weight) of the free decay from `start`, the strongest first."""
    y = motion_xy[:, start:]
    y = y - y.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(y, full_matrices=False)
    comps = (S[:components, None] * Vt[:components])
    decay, wd, _ = matrix_pencil(comps[:, ::2], 2 * dt, order)   # 500 Hz is ample below 60 Hz
    keep = (wd > 0) & (wd / (2 * np.pi) < fmax) & (decay > 0)
    wn = np.hypot(decay[keep], wd[keep])
    zeta = decay[keep] / wn
    # the weight of each mode: least squares of the components on the fitted exponentials
    t = np.arange(y.shape[1]) * dt
    basis = np.exp(np.outer(t, -decay[keep] + 1j * wd[keep]))
    amp = np.linalg.lstsq(basis, comps.T.astype(complex), rcond=None)[0]
    weight = np.linalg.norm(amp, axis=1)
    order_ = np.argsort(-weight)
    return wn[order_] / (2 * np.pi), zeta[order_], weight[order_] / weight.max()


def main(root):
    root = Path(root)
    dt = AVERAGE / FPS
    for clip in CLIPS:
        clicks = np.load(root / clip / "clicks.npy", allow_pickle=False)
        video = frames(root / clip / f"{clip}-4000fps.avi")
        xy, px, py = motion(video, clicks)
        amplitude = np.sqrt((xy ** 2).mean(0))
        start = int(np.argmax(amplitude)) + 20                  # past the release
        f, z, wgt = modes(xy, dt, start)
        np.savez(root / clip / "motion.npz", xy=xy, px=px, py=py, dt=dt, start=start)
        print(f"{clip}: {len(video)} frames at {1 / dt:g} fps, {len(px)} points, release at {start * dt:.3f} s, "
              f"peak motion {amplitude.max():.2f} px")
        for fi, zi, wi in zip(f, z, wgt):
            if wi > 0.02:
                print(f"   f {fi:6.2f} Hz   zeta {zi:.4f}   weight {wi:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])

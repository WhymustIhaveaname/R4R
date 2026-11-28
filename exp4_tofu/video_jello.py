"""Step 2's animation: the real Jello cube, and MuJoCo's with and without the theory, over it.

Top: the video of the top-front pluck (Visual Vibration Tomography, 4000 fps), its motion magnified
MAG times: the clip's mean frame, the cube cut out of it and moved by the video's own optical flow
(video_modes.py), band-passed to 4-20 Hz, over the background with the cube painted out.  Over it,
the edges of MuJoCo's cube, their displacement magnified alike, drawn through the camera that the
seven clicked corners give (a DLT, 0.6 px rms).  Blue: MuJoCo given E*, beta* of the theory;
orange: given the nominal E, beta (jello_sim.py).
Bottom: the sway of the cube's top, in the video and in both runs.

The video's motion is read as MuJoCo's two sways (the lowest P1 modes, along each side): the
tracked points are put on the cube's faces through the camera, and at each instant the two sways'
image displacements are fitted to the points'.  Each run starts in those two sways; it takes its
frequencies and dampings from MuJoCo and only its amplitudes and phases from the video, as in
plot_jello.py.

  python video_jello.py /data/datasets/vvt_jello_cube
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.signal import butter, filtfilt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import jello_model as jm  # noqa: E402
import jello_sim as js  # noqa: E402
import p1  # noqa: E402
from figures import gif_font  # noqa: E402
from plot_jello import matched  # noqa: E402

CLIP = "top_front_pluck"
MAG = 20                     # the displacement, magnified in both the video and the runs
SPEED = 0.1
FPS = 25
DURATION = 2.5               # s after the release
SCALE = 2                    # pixels on screen per pixel of the video
STRIP_H = 190
H = 1e-3
INK, INK2, GRID = (34, 34, 33), (110, 109, 105), (214, 212, 207)
COLORS = {"video": (34, 34, 33), "renormalized": (42, 120, 214), "nominal": (235, 104, 52)}


# ----------------------------------------------------------------------------------- the camera
def corners():
    """The clicked corners in the cube's frame: front at the origin, left along x, right along y.
    Clicked: bottom-left, bottom-front, bottom-right, top-left, top-front, top-back, top-right."""
    Lx, Ly, Lz = jm.SIDES
    return np.array([[Lx, 0, 0], [0, 0, 0], [0, Ly, 0], [Lx, 0, Lz], [0, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]])


def calibrate(clicks):
    A = []
    for X, (u, v) in zip(corners(), clicks):
        p = [*X, 1.0]
        A.append([*p, 0, 0, 0, 0, *[-u * q for q in p]])
        A.append([0, 0, 0, 0, *p, *[-v * q for q in p]])
    return np.linalg.svd(np.array(A))[2][-1].reshape(3, 4)


def project(P, X):
    y = np.c_[X, np.ones(len(X))] @ P.T
    return y[:, :2] / y[:, 2:]


def on_faces(P, clicks, pts):
    """The 3D points of the cube's visible faces seen at image points pts (top, left, right)."""
    Lx, Ly, Lz = jm.SIDES
    faces = (([3, 4, 6, 5], 2, Lz), ([0, 1, 4, 3], 1, 0.0), ([1, 2, 6, 4], 0, 0.0))   # polygon, axis, value
    out = np.full((len(pts), 3), np.nan)
    for poly, axis, value in faces:
        inside = Polygon(clicks[poly]).contains_points(pts)
        free = [k for k in range(3) if k != axis]
        for i in np.flatnonzero(inside & np.isnan(out[:, 0])):
            u, v = pts[i]
            # P [X; 1] = lam [u; v; 1], with X[axis] = value: three equations in two coordinates and lam
            M = np.c_[P[:, free], -np.array([u, v, 1.0])]
            X = np.linalg.solve(M, -(P[:, axis] * value + P[:, 3]))
            out[i, free] = X[:2]
            out[i, axis] = value
    return out


# ------------------------------------------------------------------------------------- the video
def mean_frame(path, every=8):
    W, Hh = 278, 288
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, Hh = int(out[0]), int(out[1])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=not(mod(n\\,{every}))",
                             "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    total, n = np.zeros((Hh, W, 3)), 0
    while True:
        buf = proc.stdout.read(W * Hh * 3)
        if len(buf) < W * Hh * 3:
            break
        total += np.frombuffer(buf, np.uint8).reshape(Hh, W, 3)
        n += 1
    proc.wait()
    return (total / n).astype(np.float32)


def video_motion(root):
    """Band-passed displacement of the tracked points from the release: (x, y) each N x T, at 1 kHz."""
    z = np.load(Path(root) / CLIP / "motion.npz", allow_pickle=False)
    xy, dt, start = z["xy"], float(z["dt"]), int(z["start"])
    b, a = butter(4, [4.0, 20.0], btype="band", fs=1 / dt)
    y = filtfilt(b, a, xy - xy.mean(1, keepdims=True), axis=1)[:, start:]
    n = len(z["px"])
    return np.c_[z["px"], z["py"]].astype(float), y[:n], y[n:], dt


def dense(pts, dx, dy, shape, spacing=6):
    """A displacement field over the frame from the tracked points' (a 6 px grid inside the cube):
    the points' own values, the nearest point's beyond them, smoothed."""
    gh, gw = shape[0] // spacing + 1, shape[1] // spacing + 1
    ix, iy = (pts[:, 0] // spacing).astype(int), (pts[:, 1] // spacing).astype(int)
    known = np.zeros((gh, gw), bool)
    known[iy, ix] = True
    _, (ny, nx) = distance_transform_edt(~known, return_indices=True)
    out = []
    for d in (dx, dy):
        g = np.zeros((gh, gw), np.float32)
        g[iy, ix] = d
        g = g[ny, nx]
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        out.append(cv2.resize(g, (gw * spacing, gh * spacing), interpolation=cv2.INTER_CUBIC)[:shape[0], :shape[1]])
    return out


# ------------------------------------------------------------------------------------ MuJoCo
def sways():
    """MuJoCo's mesh and its two lowest P1 modes, the sways along x and y, each scaled to 1 m at
    its largest."""
    X, tets = p1.mesh_from_mujoco(None, js.xml(1.0, 1e-3, 1e-3))
    X = X - X.min(0)
    K, _ = p1.assemble(X, tets, 1.0, js.NU)
    m = np.full(len(X), np.prod(jm.SIDES) / len(X))
    free = np.flatnonzero(np.repeat(X[:, 2] > 1e-9, 3))
    w2, V = eigsh(K[free][:, free], k=4, M=diags(np.repeat(m, 3)[free]), sigma=0.0, which="LM")
    shapes = []
    for k in np.argsort(w2)[:2]:
        s = np.zeros((len(X), 3))
        s.ravel()[free] = V[:, k]
        s /= s[np.argmax(np.abs(s).max(1))][np.argmax(np.abs(s).max(0))]
        shapes.append(s)
    return X, shapes


def interpolator(X, s):
    axes = [np.unique(np.round(X[:, k], 9)) for k in range(3)]
    idx = [np.searchsorted(a, np.round(X[:, k], 9)) for k, a in enumerate(axes)]
    grid = np.zeros((*map(len, axes), 3))
    grid[idx[0], idx[1], idx[2]] = s
    return RegularGridInterpolator(axes, grid, bounds_error=False, fill_value=None)


def edges(X):
    """The cube's nine visible edges, each as its vertices in order."""
    Lx, Ly, Lz = jm.SIDES
    e = 1e-7

    def line(cond, key):
        i = np.flatnonzero(cond)
        return i[np.argsort(X[i, key])]
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return [line((abs(z - Lz) < e) & (abs(y) < e), 0), line((abs(z - Lz) < e) & (abs(x) < e), 1),
            line((abs(z - Lz) < e) & (abs(y - Ly) < e), 0), line((abs(z - Lz) < e) & (abs(x - Lx) < e), 1),
            line((abs(x) < e) & (abs(y) < e), 2), line((abs(x - Lx) < e) & (abs(y) < e), 2),
            line((abs(x) < e) & (abs(y - Ly) < e), 2),
            line((abs(z) < e) & (abs(y) < e), 0), line((abs(z) < e) & (abs(x) < e), 1)]


# ----------------------------------------------------------------------------------- the frames
def main(root):
    root = Path(root)
    clicks = np.load(root / CLIP / "clicks.npy", allow_pickle=False)
    P = calibrate(clicks)
    print("camera: corners reprojected to", np.sqrt(((project(P, corners()) - clicks) ** 2).sum(1).mean()), "px rms")
    pts, dx, dy, dt = video_motion(root)
    X, shapes = sways()
    X3 = on_faces(P, clicks, pts)
    ok = ~np.isnan(X3[:, 0])
    # each sway's image displacement per metre at the tracked points
    M = []
    for s in shapes:
        S3 = interpolator(X, s)(X3[ok])
        eps = 1e-6
        M.append(((project(P, X3[ok] + eps * S3) - project(P, X3[ok])) / eps).T.ravel())
    M = np.stack(M, 1)                                       # (2 n) x 2
    Y = np.r_[dx[ok], dy[ok]]                                # (2 n) x T
    q, *_ = np.linalg.lstsq(M, Y, rcond=None)                # 2 x T, metres at each sway's largest
    explained = 1 - ((Y - M @ q) ** 2).sum() / (Y ** 2).sum()
    print(f"{ok.sum()} of {len(pts)} points on the faces; the two sways hold {explained:.0%} of the motion;"
          f" amplitudes {np.abs(q).max(1) * 1e3} mm")

    n_steps = int(DURATION / dt)
    t = np.arange(n_steps) * dt
    q = q[:, :n_steps]
    # the runs: each sway's frequency and damping from MuJoCo, its amplitude and phase from the video
    C = js.continuum_constant()
    E = js.RHO * (js.F_VIDEO / C) ** 2
    C_mj, X0, _ = js.mujoco_sway_constant()
    S = (C / C_mj) ** 2
    w = 2 * np.pi * js.F_VIDEO
    T, beta_star = js.backward_euler(w, js.ZETA_VIDEO, H)
    params = {"renormalized": (E * S * T, beta_star), "nominal": (E, 2 * js.ZETA_VIDEO / w)}
    runs = {}
    for name, (Ek, bk) in params.items():
        series = []
        for k, s in enumerate(shapes):
            f, z = js.run(Ek, bk, H, X, s, duration=DURATION + 0.5)
            series.append(matched(f, z, t, q[k]))
            print(f"{name}: sway {k}: f {f:.3f} Hz, zeta {z:.4f}")
        runs[name] = np.stack(series)

    ref = mean_frame(root / CLIP / f"{CLIP}-4000fps.avi")
    Hh, W = ref.shape[:2]
    mask = np.zeros((Hh, W), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(clicks.astype(np.int32)), 255)
    background = cv2.inpaint(ref.astype(np.uint8), cv2.dilate(mask, np.ones((7, 7), np.uint8)), 9,
                             cv2.INPAINT_TELEA).astype(np.float32)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 0.8)[..., None]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    lines = edges(X)
    # the strip: the sway of the top along the stronger sway, in the video and both runs (mm)
    main_k = int(np.argmax(np.abs(q).max(1)))
    curves = {"video": q[main_k] * 1e3, **{k: v[main_k] * 1e3 for k, v in runs.items()}}
    lim = 1.15 * max(np.abs(v).max() for v in curves.values())

    font, small = gif_font(15), gif_font(13)
    Wd, Hd = W * SCALE, Hh * SCALE
    per_frame = SPEED / FPS                                    # s of the clip per frame shown
    n_frames = int(DURATION / per_frame)
    out = c.RESULTS / "jello.mp4"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{Wd}x{Hd + STRIP_H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p",
                           "-c:v", "libx264", "-crf", "18", str(out)], stdin=subprocess.PIPE)
    for j in range(n_frames):
        i = min(int(round(j * per_frame / dt)), n_steps - 1)
        Dx, Dy = dense(pts, MAG * dx[:, i], MAG * dy[:, i], (Hh, W))
        cube = cv2.remap(np.dstack([ref, alpha[..., 0] * 255]), gx - Dx, gy - Dy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        a = cube[..., 3:] / 255
        frame = background * (1 - a) + cube[..., :3] * a
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).resize((Wd, Hd), Image.LANCZOS)
        img = img.convert("RGB")
        canvas = Image.new("RGB", (Wd, Hd + STRIP_H), (255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for name in ("nominal", "renormalized"):
            u = sum(s * runs[name][k, i] for k, s in enumerate(shapes))
            uv = project(P, X + MAG * u) * SCALE
            for ln in lines:
                draw.line([tuple(p) for p in uv[ln]], fill=COLORS[name], width=3, joint="curve")
        draw.text((10, 8), f"{SPEED:g}x", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=INK)
        draw.text((10, 28), f"motion ×{MAG}", fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=INK)
        # the strip
        left, right, s_top, s_bot = 64, Wd - 190, Hd + 30, Hd + STRIP_H - 22

        def yv(v):
            return (s_top + s_bot) / 2 - v / lim * (s_bot - s_top) / 2
        draw.text((left, s_top - 10), "sway of the top (mm)", fill=INK2, font=small, anchor="ld")
        tick = float(f"{lim / 1.15:.1g}")
        for v in (tick, 0.0, -tick):
            draw.text((left - 8, yv(v)), f"{v:g}".replace("-", "−"), fill=INK2, font=small, anchor="rm")
        draw.line([(left, yv(0)), (right, yv(0))], fill=GRID, width=2)
        for sec in range(int(DURATION) + 1):
            xt = left + sec / ((n_steps - 1) * dt) * (right - left)
            draw.text((xt, s_bot + 2), f"{sec} s", fill=INK2, font=small, anchor="ma")
        xs = left + np.arange(i + 1) / (n_steps - 1) * (right - left)
        if i > 0:
            for name in ("nominal", "renormalized"):
                draw.line(list(zip(xs, yv(curves[name][:i + 1]))), fill=COLORS[name], width=2, joint="curve")
            for x_, y_ in list(zip(xs, yv(curves["video"][:i + 1])))[::5]:
                draw.ellipse([x_ - 1.3, y_ - 1.3, x_ + 1.3, y_ + 1.3], fill=COLORS["video"])
        for k, (name, label) in enumerate((("video", "video"), ("renormalized", "MuJoCo, renormalized"),
                                           ("nominal", "MuJoCo, nominal"))):
            draw.text((right + 16, s_top + 4 + 20 * k), label, fill=COLORS[name], font=small, anchor="lt")
        ff.stdin.write(np.asarray(canvas).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])

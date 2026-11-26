"""Preview of the ground truth: the pendulum striking the slider, a little past the fit window.

Written twice from the same frames: slider.mp4 at full rate, and slider.gif, every other frame,
for the README, where GitHub does not play a video kept in the repository.

Underneath, two strips trace the pendulum's angle and the block's displacement as the clip runs;
the identification window is shaded, and the frame says when the clip has left it.  The camera
is orthographic, so the spring drawn over the frame lines up with the scene.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
from figures import gif_font, save_gif  # noqa: E402

F_CTRL, DEC = 50, 60  # the ground truth sampled at 3 kHz
FPS, SPEED = 25, 0.2
GIF_EVERY = 2  # the GIF keeps every other frame: 80 ms each, a whole number of the GIF's 10 ms ticks
WIDTH = HEIGHT = 600
STRIP_H = 220
VIEW_H, VIEW_X, VIEW_Z = 0.5, -0.08, -0.17  # m: height of the view and its centre
WALL_X = c.X_FACE - 2 * c.BLOCK_HALF - 0.11  # m, the spring's anchor
INK, GREY, LIGHT = (11, 11, 11), (82, 81, 78), (204, 202, 197)
ROD, BLOCK = (217, 115, 38), (41, 120, 214)
PAST = (200, 60, 30)


def scene_xml():
    xml = c.model_xml()
    xml = xml.replace('<visual><global offwidth="720" offheight="720"/></visual>', f"""
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight ambient=".55 .55 .55" diffuse=".5 .5 .5" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
  </asset>""")
    hz = c.BLOCK_HALF
    rail_z = c.BLOCK_Z - hz - 0.006
    xml = xml.replace("<worldbody>", f"""<worldbody>
    <camera name="side" pos="{VIEW_X} -1 {VIEW_Z}" xyaxes="1 0 0 0 0 1" projection="orthographic"
            fovy="{VIEW_H}"/>
    <geom type="box" pos="{WALL_X - 0.01} 0 {c.BLOCK_Z}" size="0.01 0.05 {hz + 0.02}"
          rgba=".45 .45 .45 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="{(WALL_X + c.X_FACE) / 2} 0 {rail_z}" size="{(c.X_FACE - WALL_X) / 2 + 0.03} 0.01 0.003"
          rgba=".6 .6 .6 1" contype="0" conaffinity="0"/>""")
    return xml


def to_px(x, z):
    s = HEIGHT / VIEW_H
    return WIDTH / 2 + (x - VIEW_X) * s, HEIGHT / 2 - (z - VIEW_Z) * s


def spring(draw, x_block):
    """A zigzag from the anchor to the back of the block."""
    x0, x1 = WALL_X, c.X_FACE - 2 * c.BLOCK_HALF + x_block
    n, amp = 10, 0.009
    xs = np.linspace(x0, x1, 2 * n + 3)
    zs = np.r_[0, [amp * (-1) ** k for k in range(2 * n + 1)], 0] + c.BLOCK_Z
    draw.line([to_px(x, z) for x, z in zip(xs, zs)], fill=GREY, width=2)


def strip(draw, box, values, n, lim, label, unit, rgb, font):
    left, top, right, bottom = box
    lo, hi = lim
    t_px = left + c.T_FIT / c.T_SHOW * (right - left)
    draw.rectangle([left, top, t_px, bottom], fill=(240, 239, 236))

    def y(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    draw.line([(left, y(0)), (right, y(0))], fill=LIGHT, width=1)
    draw.text((left - 8, top), f"{hi:g}", fill=GREY, font=font, anchor="rt")
    draw.text((left - 8, bottom), f"{lo:g}", fill=GREY, font=font, anchor="rb")
    draw.text((left + 6, top + 2), f"{label} ({unit})", fill=rgb, font=font, anchor="lt")
    assert lo <= values.min() and values.max() <= hi, "the strip would clip"
    xs = left + np.arange(n + 1) / (len(values) - 1) * (right - left)
    if n > 0:
        draw.line(list(zip(xs, y(values[:n + 1]))), fill=rgb, width=2, joint="curve")


def main(out, gif):
    refs = c.ref_sequence(F_CTRL, t_end=c.T_SHOW)
    traj = np.vstack([np.zeros((1, 2)), c.ground_truth(refs, F_CTRL, DEC)])
    rate = F_CTRL * DEC
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    stride = round(rate * SPEED / FPS)
    font, small = gif_font(19), gif_font(15)
    left, right = 70, WIDTH - 24
    boxes = [(left, HEIGHT + 22, right, HEIGHT + 102), (left, HEIGHT + 122, right, HEIGHT + 202)]
    t_px = left + c.T_FIT / c.T_SHOW * (right - left)

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT + STRIP_H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
    frames = []
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for k, i in enumerate(range(0, len(traj), stride)):
            t = i / rate
            data.qpos[:] = traj[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="side")
            img = Image.new("RGB", (WIDTH, HEIGHT + STRIP_H), (255, 255, 255))
            img.paste(Image.fromarray(renderer.render()))
            draw = ImageDraw.Draw(img)
            spring(draw, traj[i, 1])
            strip(draw, boxes[0], traj[:, 0], i, (-0.5, 0.5), "pendulum θ", "rad", ROD, small)
            strip(draw, boxes[1], 100 * traj[:, 1], i, (-6, 6), "block x", "cm", BLOCK, small)
            draw.line([(t_px, boxes[0][1] - 4), (t_px, boxes[1][3] + 4)], fill=PAST, width=2)
            draw.text((t_px - 6, boxes[1][3] + 6), "end of identification window", fill=PAST,
                      font=small, anchor="rt")
            draw.text((16, 14), f"ground truth, command {c.A_REF:g} rad × sin {c.F_REF:g} Hz, "
                      f"block {c.F_S:.2f} Hz", fill=INK, font=font)
            draw.text((16, 42), f"t = {t:4.2f} s   ({SPEED:g}x speed)", fill=GREY, font=font)
            if t <= c.T_FIT:
                draw.text((WIDTH - 16, 42), "inside the identification window", fill=GREY,
                          font=small, anchor="ra")
            else:
                draw.text((WIDTH - 16, 42), "past the identification window", fill=PAST,
                          font=small, anchor="ra")
            ffmpeg.stdin.write(img.tobytes())
            if k % GIF_EVERY == 0:
                frames.append(np.asarray(img))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    save_gif(frames, gif, FPS / GIF_EVERY)
    print(f"wrote {out} and {gif}")


if __name__ == "__main__":
    here = Path(__file__).parent
    main(str(here / "slider.mp4"), here / "slider.gif")

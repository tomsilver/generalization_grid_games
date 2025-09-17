import os

import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg


def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)


matplotlib.use("Agg")
def fig2data(fig):
    """
    Converts a Matplotlib figure to a numpy RGB array, handling backend differences.
    Works on macOS and other platforms.
    """
    # Ensure the figure uses the Agg canvas
    if not isinstance(fig.canvas, FigureCanvasAgg):
        fig.set_canvas(FigureCanvasAgg(fig))
    fig.canvas.draw()
    canvas = fig.canvas
    width, height = canvas.get_width_height()
    expected_size = width * height * 3
    try:
        if hasattr(canvas, "tostring_rgb"):
            buf = canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8)
            if img.size != expected_size:
                # Try ARGB or RGBA buffer
                buf = canvas.tostring_argb() if hasattr(canvas, "tostring_argb") else canvas.buffer_rgba()
                img = np.frombuffer(buf, dtype=np.uint8)
                if img.size == width * height * 4:
                    img = img.reshape(height, width, 4)
                    img = img[..., 1:]  # Drop alpha channel
                else:
                    print("[fig2data] ERROR: ARGB/RGBA buffer size still mismatched.")
            else:
                img = img.reshape(height, width, 3)
        else:
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            if img.size == width * height * 4:
                img = img.reshape(height, width, 4)
                img = img[..., :3]
            else:
                print("[fig2data] ERROR: RGBA buffer size mismatch.")
        return img
    except Exception as e:
        print(f"[fig2data] Exception: {e}")
        raise


def run_random_agent_demo(env_cls, outdir=None, max_num_steps=10):
    if outdir is None:
        outdir = "/tmp/{}".format(env_cls.__name__)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    video_path = os.path.join(outdir, 'random_demo.mp4')
    env = env_cls(interactive=False, record_video=True, video_out_path=video_path)
    env.reset()
    
    for t in range(max_num_steps):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()


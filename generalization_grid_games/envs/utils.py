import numpy as np
import os

def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)

def fig2data(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

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


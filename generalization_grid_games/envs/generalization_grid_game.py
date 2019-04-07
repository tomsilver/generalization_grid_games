from .utils import fig2data

from gym import spaces
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import gym
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt


class InvalidState(Exception):
    pass


class GeneralizationGridGame(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second' : 2}
    fig_scale = 1.

    def __init__(self, layout, interactive=False, record_video=False, video_out_path='out.mp4'):
        layout = np.array(layout, dtype=object)

        self.initial_layout = layout.copy()
        self.current_layout = layout.copy()

        self.interactive = interactive

        if record_video:
            self.start_recording_video(video_out_path)
        else:
            self.record_video = False
        
        height, width = layout.shape
        self.width, self.height = width, height

        self.observation_space = spaces.MultiDiscrete(self.num_tokens * np.ones((height, width)))
        self.action_space = spaces.MultiDiscrete([self.height, self.width])

        if interactive:
            self.action_lock = False

            # Create the figure and axes
            self.fig, self.ax = self.initialize_figure(height, width)
            self.drawings = []
            self.render_onscreen()

            # Create event hook for mouse clicks
            self.fig.canvas.mpl_connect('button_press_event', self.button_press)

            plt.show()

    ### Main stateful methods
    def reset(self):
        self.current_layout = self.initial_layout.copy()

        self.last_action = None

        if self.record_video:
            self.recorded_video_frames.append(self.render())
        if self.interactive:
            self.render_onscreen()

        return self.current_layout.copy()

    def render(self):
        return self.get_image(self.current_layout, self.last_action)

    def step(self, action):
        self.last_action = action

        next_layout = self.transition(self.current_layout, action)
        reward = self.compute_reward(self.current_layout, action, next_layout)
        done = self.compute_done(next_layout)
        self.current_layout = next_layout
        
        if self.record_video:
            self.recorded_video_frames.append(self.render())
        if self.interactive:
            self.render_onscreen()
        
        return next_layout.copy(), reward, done, {}

    ### Main stateless methods
    @staticmethod
    def transition(layout, action):
        raise NotImplementedError()

    @staticmethod
    def compute_reward(layout0, action, layout1):
        raise NotImplementedError()

    @staticmethod
    def compute_done(layout):
        raise NotImplementedError()

    ### Helper stateful methods
    def start_recording_video(self, video_out_path):
        self.record_video = True
        self.recorded_video_frames = []
        self.video_out_path = video_out_path

    def close(self):
        if self.record_video:
            imageio.mimsave(self.video_out_path, self.recorded_video_frames, 
                fps=self.metadata['video.frames_per_second'])
            print("Wrote out video to {}.".format(self.video_out_path))

        return super(GeneralizationGridGame, self).close()

    def button_press(self, event):
        if self.action_lock:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        i, j = map(int, (event.xdata, event.ydata))
    
        if (i < 0 or j < 0 or i >= self.width or j >= self.height):
            return

        self.action_lock = True
        c, r = i, self.height - 1 - j
        if event.button == 1:
            self.step((r, c))
        self.fig.canvas.draw()
        self.action_lock = False

    def render_onscreen(self):
        for drawing in self.drawings:
            drawing.remove()
        self.drawings = []

        for r in range(self.height):
            for c in range(self.width):
                token = self.current_layout[r, c]
                drawing = self.draw_token(token, r, c, self.ax, self.height, self.width, token_scale=0.75)
                if drawing is not None:
                    self.drawings.append(drawing)

    ### Helper stateless methods
    @classmethod
    def get_image(cls, observation, action, mode='human', close=False):
        height, width = observation.shape

        fig, ax = cls.initialize_figure(height, width)

        for r in range(height):
            for c in range(width):
                token = observation[r, c]
                cls.draw_token(token, r, c, ax, height, width)

        if action is not None:
            cls.draw_action(action, ax, height, width)

        im = fig2data(fig)
        plt.close(fig)

        return im

    @classmethod
    def initialize_figure(cls, height, width):
        fig = plt.figure(figsize=((width + 2) * cls.fig_scale, (height + 2) * cls.fig_scale))
        ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, width + 0.05),
                                    ylim=(-0.05, height + 0.05))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())
        return fig, ax

    @classmethod
    def draw_action(cls, action, ax, height, width):
        r, c = action
        if not (isinstance(r, int) or isinstance(r, np.int8) or isinstance(r, np.int64)):
            r -= 0.5
            c -= 0.5
        oi = OffsetImage(cls.hand_icon, zoom = 0.3 * cls.fig_scale * (2.5 / max(height, width)**0.5))
        box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
        ax.add_artist(box)

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width):
        raise NotImplementedError()



class TwoPlayerGeneralizationGridGame(GeneralizationGridGame):

    def __init__(self, *args, **kwargs):
        self.current_player = 0
        GeneralizationGridGame.__init__(self, *args, **kwargs)

    def reset(self):
        observation = GeneralizationGridGame.reset(self)
        self.current_player = 0
        return (observation, self.current_player)

    def step(self, action):
        self.last_action = action

        state = (self.current_layout, self.current_player)
        next_state = self.transition(state, action)
        reward = self.compute_reward(state, action, next_state)
        done = self.compute_done(next_state)

        next_layout, next_player = next_state
        self.current_layout = next_layout
        self.current_player = next_player
        
        if self.record_video:
            self.recorded_video_frames.append(self.render())
        if self.interactive:
            self.render_onscreen()
        
        return (next_layout.copy(), next_player), reward, done, {}



def VersusComputerTwoPlayerGeneralizationGridGameFactory(name, two_player_cls, player1_policy, additional_methods=None):
    def __init__(self, *args, **kwargs):
        GeneralizationGridGame.__init__(self, *args, **kwargs)

    def step(self, action):
        self.last_action = action

        # For rendering
        previous_layout = self.current_layout.copy()
        self.current_layout, next_player = two_player_cls.transition((self.current_layout, 0), action)
        timeout = 0
        while next_player == 1 and not two_player_cls.compute_done((self.current_layout, next_player)):
            if self.record_video:
                self.recorded_video_frames.append(self.render())
            if self.interactive:
                self.render_onscreen()
                plt.pause(0.5)

            player1_move = player1_policy(self.current_layout)
            self.current_layout, next_player = two_player_cls.transition((self.current_layout, next_player), player1_move)

            timeout += 1
            if timeout > 100:
                import pdb; pdb.set_trace()

        reward = self.compute_reward(previous_layout, action, self.current_layout)
        done = self.compute_done(self.current_layout)
        
        if self.record_video:
            self.recorded_video_frames.append(self.render())
        if self.interactive:
            self.render_onscreen()
        
        return self.current_layout.copy(), reward, done, {}

    def transition(state, action):
        state, next_player = two_player_cls.transition((state, 0), action)
        timeout = 0
        while next_player == 1 and not two_player_cls.compute_done((state, next_player)):
            player1_move = player1_policy(state)
            state, next_player = two_player_cls.transition((state, next_player), player1_move)
            timeout += 1
            if timeout > 100:
                import pdb; pdb.set_trace()
        return state

    def compute_done(state):
        return two_player_cls.compute_done((state, 0))

    def compute_reward(state0, action, state1):
        return two_player_cls.compute_reward((state0, 0), action, (state1, 0))

    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        return two_player_cls.draw_token(token, r, c, ax, height, width, token_scale=token_scale)

    def initialize_figure(cls, height, width):
        return two_player_cls.initialize_figure(height, width)

    method_map = {
        "__init__": __init__, 
        "step": step,
        "transition": staticmethod(transition),
        "compute_done" : staticmethod(compute_done),
        "compute_reward" : staticmethod(compute_reward),
        "draw_token" : classmethod(draw_token),
        "num_tokens" : two_player_cls.num_tokens,
        "hand_icon" : two_player_cls.hand_icon,
        "initialize_figure" : classmethod(initialize_figure)
    }

    if additional_methods is not None:
        method_map.update(additional_methods)

    newclass = type(name, (GeneralizationGridGame,), method_map)
    return newclass

def GymEnvFactory(base_class, layout_id, layout):
    name = "{}GymEnv{}".format(base_class.__name__, layout_id)

    def __init__(self, *args, **kwargs):
        base_class.__init__(self, layout, *args, **kwargs)

    newclass = type(name, (base_class,), {'__init__' : __init__})

    return newclass

def create_gym_envs(base_class, layouts, global_context):
    for i, layout in enumerate(layouts):
        gym_env = GymEnvFactory(base_class, i, layout)
        global_context[gym_env.__name__] = gym_env



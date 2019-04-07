from .generalization_grid_game import GeneralizationGridGame, create_gym_envs
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
FALLING = 'falling_token'
RED = 'red_token'
STATIC = 'static_token'
ADVANCE = 'advance_token'
DRAWN = 'drawn_token'
ALL_TOKENS = [EMPTY, FALLING, RED, STATIC, ADVANCE, DRAWN]

TOKEN_IMAGES = {
    FALLING : plt.imread(get_asset_path('skydiver.png')),
    RED : plt.imread(get_asset_path('fire.png')),
    STATIC : plt.imread(get_asset_path('block.jpg')),
    ADVANCE : plt.imread(get_asset_path('green_button.png')),
    DRAWN : plt.imread(get_asset_path('blue_block.jpg')),
}

HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class StopTheFall(GeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.1

    @staticmethod
    def transition(layout, action):
        r, c = action
        token = layout[r, c]
        new_layout = layout.copy()

        if token == ADVANCE:
            new_layout[r, c] = STATIC
            StopTheFall.finish_simulation(new_layout)
            return new_layout

        if token != EMPTY:
            return new_layout

        new_layout[r, c] = DRAWN
        return new_layout

    @staticmethod
    def compute_done(layout):
        return StopTheFall.compute_reward(None, None, layout) > 0

    @staticmethod
    def compute_reward(state0, action, state1):
        if np.any(state1 == ADVANCE):
            return 0

        falling_positions = np.argwhere(state1 == FALLING)

        for falling_pos in falling_positions:
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_pos = np.add(falling_pos, direction)
                if not (neighbor_pos[0] >= 0 and neighbor_pos[0] < state1.shape[0] and \
                        neighbor_pos[1] >= 0 and neighbor_pos[1] < state1.shape[1]):
                    continue
                if state1[neighbor_pos[0], neighbor_pos[1]] == RED:
                    return 0

        return 1

    @staticmethod
    def finish_simulation(layout):
        height, width = layout.shape

        while True:
            something_moved = False

            for r in range(height-2, -1, -1):
                for c in range(width):
                    token = layout[r, c]

                    if (token == FALLING or token == DRAWN) and (layout[r+1, c] == EMPTY):
                        layout[r, c] = EMPTY
                        layout[r+1, c] = token
                        something_moved = True

            if not something_moved:
                break

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            return None

        im = TOKEN_IMAGES[token]
        oi = OffsetImage(im, zoom = cls.fig_scale * (token_scale / max(height, width)**0.5))
        box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

        ax.add_artist(box)

        return box

    @classmethod
    def initialize_figure(cls, height, width):
        fig, ax = GeneralizationGridGame.initialize_figure(height, width)

        # Draw a white grid in the background
        for r in range(height):
            for c in range(width):
                edge_color = '#888888'
                face_color = 'white'
                
                drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                             numVertices=4,
                                             radius=0.5 * np.sqrt(2),
                                             orientation=np.pi / 4,
                                             ec=edge_color,
                                             fc=face_color)
                ax.add_patch(drawing)

        return fig, ax


### Specific environments

E = EMPTY
R = RED
F = FALLING
S = STATIC
A = ADVANCE
D = DRAWN

layout0 = [
    [E, E, E, E, E, F, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, R, R, E, E],
    [E, E, E, R, R, E, E, R, R, E, E],
    [E, E, E, R, R, E, E, R, R, E, E],
    [S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, A, S, S]
]

layout1 = [
    [E, E, E, F, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [R, R, R, E, R, R, R, R, R, R, R, R],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A]
]

layout2 = [
    [E, E, E, E, E, E, E, E, F, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [E, E, E, E, E, E, R, R, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A]
]

layout3 = [
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, F, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, R, R, R, R, R, R],
    [E, E, E, E, E, E, R, R, R, R, R, R],
    [E, E, E, E, E, E, R, R, R, R, R, R],
    [E, E, E, E, E, E, R, R, R, R, R, R],
    [E, E, S, S, S, S, S, S, S, S, S, S],
    [E, E, S, S, S, S, S, S, S, S, S, S],
    [E, E, S, S, S, S, S, S, S, S, S, S],
    [E, E, S, S, S, S, S, S, S, S, S, S],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A]
]

layout4 = [
    [E, E, E, E, E, E, F, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [R, R, R, R, R, R, E, R, R, R, R, R, R, R, R, R, R],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, A]
]

layout5 = [
    [E, E, E, E, E, E, E, F, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [R, R, E, E, R, E, R, E, R, E, E, R],
    [R, R, E, R, R, R, R, E, R, R, R, R],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A]
]

layout6 = [
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, F, E, E],
    [E, E, E, E, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [E, R, E, R, E],
    [S, S, S, S, S],
    [A, S, S, S, S],
]

layout7 = [
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, S, S, S],
    [E, R, R, E, E, F, E, E, E, E, E, E],
    [E, S, S, E, R, E, R, E, E, E, E, E],
    [E, E, E, E, S, S, S, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, S, E, S, S, E, S, S, S],
    [E, E, S, S, S, S, S, S, S, S, S, S],
    [R, R, S, S, S, S, S, S, S, S, S, S],
    [R, R, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A]
]

layout8 = [
    [E, E, E, E, E, A, E, E, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    [E, E, E, E, E, E, E, E, F, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, R, R, R, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, R, R, R, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
]

layout9 = [
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, F, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, R, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, R, E, R, E, E, E, E],
    [E, E, E, E, E, E, E, R, E, R, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, A, S, S]
]

layout10 = [
    [E, E, E, F, E, E, E],
    [E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E],
    [S, E, E, E, E, E, S],
    [R, S, E, E, E, S, R],
    [S, R, S, E, S, R, S],
    [R, S, R, E, R, S, R],
    [R, R, S, E, S, R, R],
    [R, R, R, E, R, R, R],
    [S, S, S, S, S, S, S],
    [S, S, S, S, S, S, A]
]

layout11 = [
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, F, E, E, E, E, E, E, E, E, E],
    [E, S, E, S, E, E, E, E, E, E, E, E],
    [E, S, E, S, E, E, E, E, E, E, E, E],
    [E, S, E, S, E, E, E, E, E, E, E, E],
    [E, S, E, S, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E],
    [E, R, E, R, E, E, E, E, E, E, E, E],
    [E, R, E, R, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, A, S, S, S, S, S, S, S, S, S],
]

layout12 = [
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, S, S, S, E, E, E, E, E],
    [E, E, E, E, E, E, E, S, A, S, E, E, E, E, E],
    [E, E, E, E, E, E, E, S, S, S, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, F, E, E, E, E, E, E, E, E, E],
    [R, R, R, R, R, E, E, E, E, E, E, E, E, E, E],
    [R, R, R, R, R, E, E, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
]

layout13 = [
    [E, E, E, E, E, E, E, R, R, R, R],
    [E, E, E, E, E, F, E, R, R, R, R],
    [E, E, E, E, E, E, E, R, R, R, R],
    [E, E, E, E, E, E, E, R, R, R, R],
    [E, E, E, E, E, E, E, R, R, R, R],
    [R, R, R, R, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E],
    [R, R, R, R, R, E, R, R, R, R, R],
    [R, R, R, R, R, E, R, R, R, R, R],
    [S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, A, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S],
]

layout14 = [
    [E, E, E, E, F, E, E, E, E, E, S, A],
    [E, E, E, E, E, E, E, E, E, E, S, S],
    [S, S, S, S, E, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E, E],
    [S, S, S, S, E, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E, E],
    [S, S, S, S, E, E, E, E, E, E, E, E],
    [R, R, R, R, E, E, E, E, E, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S, S, S],
]

layout15 = [
    [E, E, E, E, E, E, E, F, E, E],
    [E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, R, E, R, E],
    [E, E, E, E, E, E, S, E, S, E],
    [E, E, E, E, E, E, R, E, R, E],
    [E, E, E, E, E, E, S, E, S, E],
    [E, E, E, E, E, E, R, E, R, E],
    [S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, S, S, S, S, S],
    [S, S, A, S, S, S, S, S, S, S],
]

layout16 = [
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, F, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, R, E, R, E, R, E, R, E, E, E],
    [E, E, E, R, E, R, E, R, E, R, E, E, E],
    [E, E, E, R, E, R, E, R, E, R, E, E, E],
    [S, S, S, S, S, S, S, S, S, S, S, S, S],
    [S, S, S, S, S, A, S, S, S, S, S, S, S],
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
    [E, E, E, E, E, E, E, E, E, E, E, E, E],
]

layout17 = [
   [S, E, S, E, E, F, S, E, S, E],
   [E, S, E, S, E, E, E, S, E, S],
   [S, E, S, E, E, E, S, E, S, E],
   [E, S, E, S, E, E, E, S, E, S],
   [S, E, S, E, E, E, S, E, S, E],
   [E, S, E, S, E, E, E, S, E, S],
   [S, E, S, E, R, E, S, E, S, E],
   [E, S, E, S, R, E, E, S, E, S],
   [S, E, S, R, R, E, S, E, S, E],
   [S, S, S, S, S, S, S, S, S, S],
   [S, S, A, S, S, S, S, S, S, S],   
]

layout18 = [
   [R, R, E, E, R, R, E, F, E, E, E, E],
   [R, R, E, E, R, R, E, E, E, E, E, E],
   [E, E, R, R, E, E, E, E, E, E, E, E],
   [E, E, R, R, E, E, E, E, R, E, E, E],
   [R, R, E, E, R, R, E, E, R, R, E, E],
   [R, R, E, E, R, R, E, E, R, R, R, E],
   [E, E, R, R, E, E, E, E, R, R, R, R],
   [E, E, R, R, E, E, E, E, R, R, R, R],
   [S, S, S, S, S, S, S, S, S, S, S, S],
   [S, S, A, S, S, S, S, S, S, S, S, S],
   [S, S, S, S, S, S, S, S, S, S, S, S],
]

layout19 = [
    [E, E, E, E, E, E, E],
    [E, E, F, E, E, E, E],
    [E, E, E, E, E, E, E],
    [R, R, E, E, E, E, E],
    [R, R, E, E, E, E, E],
    [S, S, S, E, S, S, S],
    [E, E, E, E, S, A, S],
    [E, E, E, E, S, S, S],
]

layouts = [layout0, layout1, layout2, layout3, layout4, layout5, layout6, 
           layout7, layout8, layout9, layout10, layout11, layout12, layout13,
           layout14, layout15, layout16, layout17, layout18, layout19]

create_gym_envs(StopTheFall, layouts, globals())

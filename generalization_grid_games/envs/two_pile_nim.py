from .generalization_grid_game import TwoPlayerGeneralizationGridGame, create_gym_envs, \
    VersusComputerTwoPlayerGeneralizationGridGameFactory
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
import numpy as np



EMPTY = 'empty'
TOKEN = 'token'
ALL_TOKENS = [EMPTY, TOKEN]

TOKEN_IMAGES = {
    TOKEN : plt.imread(get_asset_path('matchstick.png')),
}

HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class TwoPileNimTwoPlayer(TwoPlayerGeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 0.9

    @staticmethod
    def transition(state, action):
        layout, current_player = state
        r, c = action
        height, width = layout.shape
        new_layout = layout.copy()
        token = layout[r, c]

        if token != TOKEN:
            return (new_layout, current_player)

        TwoPileNimTwoPlayer.execute_player_move(new_layout, r, c)

        next_player = (current_player + 1) % 2
        
        return (new_layout, next_player)

    @staticmethod
    def compute_done(state):
        layout, player = state
        return not np.any(layout == TOKEN)

    @staticmethod
    def compute_reward(state0, action, state1):
        layout0, player0 = state0

        if not TwoPileNimTwoPlayer.compute_done(state1):
            return 0.
 
        if action[0] != layout0.shape[0] - 1:
            return 0.
 
        if action[1] == 0 and np.all(layout0[:, 1] == EMPTY):
            return 1. if player0 == 0 else -1
 
        if action[1] == 1 and np.all(layout0[:, 0] == EMPTY):
            return 1. if player0 == 0 else -1
 
        return 0.

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            edge_color = '#888888'
            face_color = 'white'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

            return drawing

        else:
            edge_color = '#888888'
            face_color = '#DDDDDD'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

            im = TOKEN_IMAGES[token]
            oi = OffsetImage(im, zoom = cls.fig_scale * (token_scale / max(height, width)**0.5))
            box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

            ax.add_artist(box)

            return box

    @classmethod
    def initialize_figure(cls, height, width):
        fig, ax = TwoPlayerGeneralizationGridGame.initialize_figure(height, width)

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

    @staticmethod
    def execute_player_move(layout, r, c):
        if layout[r, c] != TOKEN:
            raise Exception("Can only remove tokens.")

        # Remove this token and all those above it
        for r1 in range(r+1):
            token1 = layout[r1, c]

            if token1 == TOKEN:
                layout[r1, c] = EMPTY


def player1_policy(state):
    r1 = np.max(np.argwhere(state == EMPTY)[:, 0])
    if state[r1, 0] == TOKEN:
        c1 = 0
    elif state[r1, 1] == TOKEN:
        c1 = 1
    else:
        r1 += 1
        c1 = 0
    return (r1, c1)

TwoPileNim = VersusComputerTwoPlayerGeneralizationGridGameFactory("TwoPileNim", TwoPileNimTwoPlayer, player1_policy)

### Specific environments
rng = np.random.RandomState(0)
num_layouts = 20

def create_random_layout():
    height = rng.randint(2, 20)
    left_column_height = rng.randint(1, height)
    while True:
        right_column_height = rng.randint(1, height)
        if right_column_height != left_column_height:
            break
    layout = np.full((height, 2), TOKEN, dtype=object)
    layout[:left_column_height, 0] = EMPTY
    layout[:right_column_height, 1] = EMPTY
    return layout

layouts = [create_random_layout() for _ in range(num_layouts)]
create_gym_envs(TwoPileNim, layouts, globals())
create_gym_envs(TwoPileNimTwoPlayer, layouts, globals())

from .generalization_grid_game import TwoPlayerGeneralizationGridGame, create_gym_envs, \
    VersusComputerTwoPlayerGeneralizationGridGameFactory
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib.pyplot as plt
import numpy as np



EMPTY = 'empty'
TARGET = 'target'
AGENT = 'agent'
WALL = 'wall'
DRAWN = 'drawn'
LEFT_ARROW = 'left_arrow'
RIGHT_ARROW = 'right_arrow'
UP_ARROW = 'up_arrow'
DOWN_ARROW = 'down_arrow'

ALL_TOKENS = [EMPTY, TARGET, AGENT, DRAWN, WALL, LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW]

TOKEN_IMAGES = {
    AGENT : plt.imread(get_asset_path('stickfigure.png')),
    TARGET : plt.imread(get_asset_path('rabbit.png')),
    WALL : plt.imread(get_asset_path('bush.png')),
    DRAWN : plt.imread(get_asset_path('blue_block.jpg')),
}

HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class ChaseTwoPlayer(TwoPlayerGeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.2

    @staticmethod
    def transition(state, action):
        layout, current_player = state
        new_layout = layout.copy()
        r, c = action
        token = layout[r, c]

        if current_player == 0:
            moving_obj_type = AGENT
        else:
            moving_obj_type = TARGET

        if token == EMPTY:
            new_layout[r, c] = DRAWN

        elif token == UP_ARROW:
            ChaseTwoPlayer.step_move_in_direction(new_layout, (-1, 0), moving_obj_type)

        elif token == DOWN_ARROW:
            ChaseTwoPlayer.step_move_in_direction(new_layout, (1, 0), moving_obj_type)

        elif token == LEFT_ARROW:
            ChaseTwoPlayer.step_move_in_direction(new_layout, (0, -1), moving_obj_type)

        elif token == RIGHT_ARROW:
            ChaseTwoPlayer.step_move_in_direction(new_layout, (0, 1), moving_obj_type)

        next_player = (current_player + 1) % 2

        return (new_layout, next_player)

    @staticmethod
    def compute_reward(state0, action, state1):
        return float(ChaseTwoPlayer.compute_done(state1))

    @staticmethod
    def compute_done(state):
        return not np.any(state[0] == TARGET)

    @staticmethod
    def step_move_in_direction(layout, direction, moving_obj_type):
        r, c = np.argwhere(layout == moving_obj_type)[0]
        neighbor_cell = layout[r + direction[0], c + direction[1]]

        if neighbor_cell in [EMPTY, TARGET]:
            next_r, next_c = r + direction[0], c + direction[1]

        else:
            return

        layout[r, c] = EMPTY
        layout[next_r, next_c] = moving_obj_type

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            return None

        if 'arrow' in token:
            edge_color = '#888888'
            face_color = 'white'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

        if token == LEFT_ARROW:
            arrow_drawing = FancyArrow(c + 0.75, height - 1 - r + 0.5, -0.25, 
                0.0, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == RIGHT_ARROW:
            arrow_drawing = FancyArrow(c + 0.25, height - 1 - r + 0.5, 0.25, 
                0.0, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == UP_ARROW:
            arrow_drawing = FancyArrow(c + 0.5, height - 1 - r + 0.25, 0.0, 
                0.25, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == DOWN_ARROW:
            arrow_drawing = FancyArrow(c + 0.5, height - 1 - r + 0.75, 0.0, 
                -0.25, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        else:
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


def player1_policy(layout):
    r, c = np.argwhere(layout == TARGET)[0]
    ra, rc = np.argwhere(layout == AGENT)[0]

    current_distance_to_agent = abs(ra - r) + abs(rc - c)

    possible_moves = []
    for direction in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        new_r, new_c = r + direction[0], c + direction[1]
        neighbor_cell = layout[new_r, new_c]

        if neighbor_cell not in [EMPTY, TARGET]:
            continue

        next_distance_to_agent = abs(ra - new_r) + abs(rc - new_c)
        if next_distance_to_agent > current_distance_to_agent:
            possible_moves.append(direction)

    if len(possible_moves) == 0:
        return (0, 0)

    idx = 0
    dr, dc = possible_moves[idx]

    if (dr, dc) == (0, 1):
        return np.argwhere(layout == RIGHT_ARROW)[0]

    if (dr, dc) == (0, -1):
        return np.argwhere(layout == LEFT_ARROW)[0]

    if (dr, dc) == (-1, 0):
        return np.argwhere(layout == UP_ARROW)[0]

    if (dr, dc) == (1, 0):
        return np.argwhere(layout == DOWN_ARROW)[0]

    raise Exception("Not possible")

Chase = VersusComputerTwoPlayerGeneralizationGridGameFactory("Chase", ChaseTwoPlayer, player1_policy)

### Specific environments

E = EMPTY
T = TARGET
A = AGENT
W = WALL
U = UP_ARROW
D = DOWN_ARROW
L = LEFT_ARROW
R = RIGHT_ARROW

layout0 = [
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, W, W, W, W, W, W, W, E, E, W],
    [W, E, W, E, E, T, E, E, W, E, E, W],
    [W, E, W, E, E, E, E, E, W, E, E, W],
    [W, E, W, E, E, E, E, E, W, E, E, W],
    [W, E, W, E, A, E, E, E, W, E, E, W],
    [W, E, W, E, E, E, E, E, W, E, E, W],
    [W, E, W, E, E, E, E, E, W, E, E, W],
    [W, E, W, E, E, E, E, E, W, E, E, W],
    [W, E, W, W, W, W, W, W, W, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, R, L, U, D],
    [W, W, W, W, W, W, W, W, W, W, W, W],
]

layout1 = [
    [W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, T, E, A, E, W],
    [W, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, U, D, L, R],
]

layout2 = [
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, W, W, W, W, W],
    [W, E, E, E, T, E, W, W, W, W, W],
    [W, E, E, E, E, E, W, W, W, W, W],
    [W, E, E, E, E, E, W, W, W, W, W],
    [W, E, E, E, E, E, W, W, W, W, W],
    [W, E, A, E, E, E, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, R, W, L, W, D, W, U, W, W],
]

layout3 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, A, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, T, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, R, L, D, U],
]

layout4 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, W, W, W, W, W, W, W, W, W, W, E, E, W],
    [W, E, E, W, E, E, E, E, E, E, E, E, W, E, E, W],
    [W, E, E, W, E, E, E, E, E, E, E, E, W, E, E, W],
    [W, E, E, W, E, E, T, E, E, E, E, E, W, E, E, W],
    [W, E, E, W, E, E, E, A, E, E, E, E, W, E, E, W],
    [W, E, E, W, E, E, E, E, E, E, E, E, W, E, E, W],
    [W, E, E, W, W, W, W, W, W, W, W, W, W, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, U, D, L, R],
]

layout5 = [
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, E, E, E, T, E, W],
    [W, W, E, E, E, E, E, W],
    [W, W, E, A, E, E, E, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, D, U, R, L],
]

layout6 = [
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, R, L, D, U, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, A, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, E, T, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
]

layout7 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, A, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, T, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, D, W, L, W, W, W, W, W, W, W, W, W, W, W, W],
    [R, W, U, W, W, W, W, W, W, W, W, W, W, W, W, W],
]

layout8 = [
    [W, W, W, W, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, T, E, W],
    [W, E, A, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, E, E, E, W],
    [W, W, W, W, W],
    [W, U, D, L, R],
]

layout9 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, A, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, T, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, E, E, E, E, E, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, U, D, L, R],
]

layout10 = [
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, W, W, E, E],
    [W, E, E, E, E, E, E, W, W, E, E],
    [W, E, E, E, E, E, E, W, W, E, E],
    [W, E, E, A, E, E, E, W, W, W, W],
    [W, E, E, E, E, E, E, W, W, W, W],
    [W, E, E, E, E, T, E, W, W, W, W],
    [W, E, E, E, E, E, E, W, W, W, W],
    [W, E, E, E, E, E, E, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, E, E, E, W, W, W, W, W, W],
    [W, W, E, E, E, W, W, W, W, W, W],
    [W, W, E, E, E, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, U, D, L, R],
]

layout11 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, T, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, A, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, U, D, L, R, W, W, W, W, W, W, W, W, W],
]

layout12 = [
    [W, W, W, W, W, W, W, W, W, W, W],
    [U, W, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, W],
    [D, W, E, E, E, T, E, E, E, E, W],
    [W, W, E, E, E, E, E, E, E, E, W],
    [L, W, E, E, E, E, E, E, E, E, W],
    [W, W, E, E, A, E, E, E, E, E, W],
    [R, W, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]

layout13 = [
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, E, W, E, E, E, E, W, E, W],
    [W, E, W, W, W, W, W, W, W, W, W],
    [W, W, W, E, E, E, E, E, W, E, W],
    [W, W, W, E, E, E, E, E, W, W, W],
    [W, E, W, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, E, E, W, E, W],
    [W, W, W, T, E, E, E, E, W, E, W],
    [W, W, W, E, E, E, E, E, W, W, W],
    [W, W, W, E, E, E, A, E, W, E, W],
    [W, E, W, E, E, E, E, E, W, W, W],
    [W, W, W, W, W, W, W, W, W, E, W],
    [W, W, W, E, W, W, E, W, W, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, U, D, L, R],
]

layout14 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [U, L, D, R, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, W, W, E, E, E, E, E, E, E, E, E, W, E, W, E, W],
    [W, W, W, W, W, E, E, E, E, E, E, E, E, E, W, W, W, W, W],
    [W, E, E, W, W, E, E, E, E, A, E, E, E, E, W, E, W, E, W],
    [W, W, W, W, W, E, E, E, E, E, E, E, E, E, W, W, W, W, W],
    [W, E, E, W, W, E, T, E, E, E, E, E, E, E, W, E, E, E, W],
    [W, E, E, W, W, E, E, E, E, E, E, E, E, E, W, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
]

layout15 = [
    [W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, T, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, A, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, D, U, L, R, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
]


layout16 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, W, W, W, W, W, W, W, W, W, W, W, E, W],
    [W, E, W, E, E, E, E, E, E, E, E, E, W, E, W],
    [W, E, W, E, W, W, W, W, W, W, W, E, W, E, W],
    [W, E, W, E, W, E, E, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, E, A, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, E, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, E, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, T, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, E, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, E, E, E, E, E, W, E, W, E, W],
    [W, E, W, E, W, W, W, W, W, W, W, E, W, E, W],
    [W, E, W, E, E, E, E, E, E, E, E, E, W, E, W],
    [W, E, W, W, W, W, W, W, W, W, W, W, W, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, U, D, L, R, W, W, W, W, W],
]

layout17 = [
    [W, W, L, R, U, D, W],
    [W, W, W, W, W, W, W],
    [W, E, E, E, E, E, W],
    [W, E, E, E, E, E, W],
    [W, E, E, E, E, E, W],
    [W, E, E, E, E, E, W],
    [W, E, E, E, E, E, W],
    [W, E, E, A, E, E, W],
    [W, E, E, E, T, E, W],
    [W, E, E, E, E, E, W],
    [W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W],
]

layout18 = [
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, A, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, T, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, D, U, L, R, W, W, W, W, W, W],
]

layout19 = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, A, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, T, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, W, D, U, L, R, W, W, W, W, W, W, W, W],
]


layouts = [layout0, layout1, layout2, layout3, layout4, layout5, layout6, 
           layout7, layout8, layout9, layout10, layout11, layout12, layout13,
           layout14, layout15, layout16, layout17, layout18, layout19]

create_gym_envs(Chase, layouts, globals())
create_gym_envs(ChaseTwoPlayer, layouts, globals())

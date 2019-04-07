from .generalization_grid_game import GeneralizationGridGame, create_gym_envs
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
AGENT = 'agent'
STAR = 'star'
DRAWN = 'drawn'
LEFT_ARROW = 'left_arrow'
RIGHT_ARROW = 'right_arrow'
ALL_TOKENS = [EMPTY, AGENT, DRAWN, STAR, LEFT_ARROW, RIGHT_ARROW]


TOKEN_IMAGES = {
    AGENT : plt.imread(get_asset_path('robot.png')),
    DRAWN : plt.imread(get_asset_path('brown_block.jpg')),
    STAR : plt.imread(get_asset_path('star.png')),
}


HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class ReachForTheStar(GeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.2

    @staticmethod
    def transition(layout, action):
        r, c = action
        token = layout[r, c]
        new_layout = layout.copy()

        if token == EMPTY:
            new_layout[r, c] = DRAWN

        elif token == LEFT_ARROW:
            ReachForTheStar.step_move_in_direction(new_layout, -1)

        elif token == RIGHT_ARROW:
            ReachForTheStar.step_move_in_direction(new_layout, 1)

        else:
            return new_layout

        ReachForTheStar.finish_simulation(new_layout)

        return new_layout

    @staticmethod
    def compute_reward(layout0, action, layout1):
        return float(ReachForTheStar.compute_done(layout1))

    @staticmethod
    def compute_done(layout):
        return not np.any(layout == STAR)

    @staticmethod
    def step_move_in_direction(layout, direction):
        height, width = layout.shape

        r, c = np.argwhere(layout == AGENT)[0]

        if c + direction < 0 or c + direction >= width:
            return

        neighbor_cell = layout[r, c + direction]

        if neighbor_cell in [EMPTY, STAR]:
            next_r, next_c = r, c + direction

        elif neighbor_cell == DRAWN and layout[r - 1, c + direction] in [EMPTY, STAR]:
            next_r, next_c = r - 1, c + direction

        else:
            return

        layout[r, c] = EMPTY
        layout[next_r, next_c] = AGENT

    @staticmethod
    def finish_simulation(layout):
        height, width = layout.shape

        while True:
            something_moved = False

            for r in range(height-2, -1, -1):
                for c in range(width):
                    token = layout[r, c]

                    if (token == AGENT or token == DRAWN) and (layout[r+1, c] == EMPTY):
                        layout[r, c] = EMPTY
                        layout[r+1, c] = token
                        something_moved = True

            if not something_moved:
                break

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            return None

        if 'arrow' in token:
            edge_color = '#888888'
            face_color = '#AAAAAA'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

        if token == LEFT_ARROW:
            arrow_drawing = FancyArrow(c + 0.75, height - 1 - r + 0.5, -0.25, 
                0.0, width=0.1, fc='green', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == RIGHT_ARROW:
            arrow_drawing = FancyArrow(c + 0.25, height - 1 - r + 0.5, 0.25, 
                0.0, width=0.1, fc='green', head_length=0.2)
            ax.add_patch(arrow_drawing)

        else:
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
rng = np.random.RandomState(0)
num_layouts = 20

def create_random_layout():
    stairs_height = rng.randint(2, 11)
    stairs_dist_from_right = rng.randint(0, 5)
    stairs_dist_from_top = rng.randint(0, 5)
    agent_dist_from_stairs = rng.randint(0, 5)
    agent_dist_from_left = rng.randint(0, 5)

    height = 2 + stairs_height + stairs_dist_from_top
    width = stairs_dist_from_right + 2*stairs_height + agent_dist_from_stairs + agent_dist_from_left
    layout = np.full((height, width), EMPTY, dtype=object)

    star_r = stairs_dist_from_top
    star_c = agent_dist_from_left + agent_dist_from_stairs + stairs_height
    agent_r = height - 3
    agent_c = agent_dist_from_left

    layout[star_r, star_c] = STAR
    layout[agent_r, agent_c] = AGENT

    if rng.uniform() > 0.5:
        layout = np.fliplr(layout)

    layout[-2:, :] = DRAWN
    layout[-1, -1] = RIGHT_ARROW
    layout[-1, -2] = LEFT_ARROW

    return layout

layouts = [create_random_layout() for _ in range(num_layouts)]
create_gym_envs(ReachForTheStar, layouts, globals())

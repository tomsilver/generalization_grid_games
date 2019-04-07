from .generalization_grid_game import TwoPlayerGeneralizationGridGame, create_gym_envs, InvalidState, \
    VersusComputerTwoPlayerGeneralizationGridGameFactory
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
BLACK_KING = 'black_king'
WHITE_KING = 'white_king'
WHITE_QUEEN = 'white_queen'
HIGHLIGHTED_WHITE_QUEEN = 'highlighted_white_queen'
HIGHLIGHTED_WHITE_KING = 'highlighted_white_king'
HIGHLIGHTED_BLACK_KING = 'highlighted_black_king'
ALL_TOKENS = [EMPTY, BLACK_KING, WHITE_KING, WHITE_QUEEN, HIGHLIGHTED_WHITE_QUEEN, HIGHLIGHTED_WHITE_KING, HIGHLIGHTED_BLACK_KING]
COLOR_TO_PIECES = {
    'white' : [WHITE_KING, WHITE_QUEEN, HIGHLIGHTED_WHITE_QUEEN, HIGHLIGHTED_WHITE_KING],
    'black' : [BLACK_KING, HIGHLIGHTED_BLACK_KING],
}
opposite_color = lambda c : 'black' if c == 'white' else 'white'
HIGHlIGHTED_ANALOGS = {
    WHITE_QUEEN : HIGHLIGHTED_WHITE_QUEEN,
    WHITE_KING : HIGHLIGHTED_WHITE_KING,
    BLACK_KING : HIGHLIGHTED_BLACK_KING,
}
UNHIGHLIGHTED_ANALOGS = {v : k for k, v in HIGHlIGHTED_ANALOGS.items()}

TOKEN_IMAGES = {
    BLACK_KING : plt.imread(get_asset_path('black_king.png')),
    WHITE_KING : plt.imread(get_asset_path('white_king.png')),
    WHITE_QUEEN : plt.imread(get_asset_path('white_queen.png')),
    HIGHLIGHTED_WHITE_QUEEN : plt.imread(get_asset_path('white_queen.png')),
    HIGHLIGHTED_WHITE_KING : plt.imread(get_asset_path('white_king.png')),
    HIGHLIGHTED_BLACK_KING : plt.imread(get_asset_path('black_king.png')),
}

PIECE_VALID_MOVES = {
    BLACK_KING : lambda pos, layout : king_valid_moves(pos, layout, 'black'),
    WHITE_KING : lambda pos, layout : king_valid_moves(pos, layout, 'white'),
    WHITE_QUEEN : lambda pos, layout : queen_valid_moves(pos, layout, 'white'),
    HIGHLIGHTED_WHITE_QUEEN : lambda pos, layout : queen_valid_moves(pos, layout, 'white'),
    HIGHLIGHTED_WHITE_KING : lambda pos, layout : king_valid_moves(pos, layout, 'white'),
    HIGHLIGHTED_BLACK_KING : lambda pos, layout : king_valid_moves(pos, layout, 'black'),
}

HAND_ICON_IMAGE = plt.imread(get_asset_path('blue_hand_icon.png'))


def king_valid_moves(pos, layout, color):
    valid_moves = []

    attacking_mask = get_attacking_mask(layout, opposite_color(color))

    for direction in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        running_pos = np.add(pos, direction)
        if np.any(running_pos < 0):
            continue
        try:
            next_cell = layout[running_pos[0], running_pos[1]]
        except IndexError:
            continue

        if next_cell in COLOR_TO_PIECES[color]:
            continue

        if next_cell in COLOR_TO_PIECES[opposite_color(color)]:
            excluded_layout = layout.copy()
            excluded_layout[running_pos[0], running_pos[1]] = EMPTY
            excluded_attacking_mask = get_attacking_mask(excluded_layout, opposite_color(color))
            if not excluded_attacking_mask[running_pos[0], running_pos[1]]:
                valid_moves.append(running_pos)
            continue

        if not attacking_mask[running_pos[0], running_pos[1]]:
            valid_moves.append(running_pos)

    return valid_moves

def get_king_attacking_spaces(pos, layout):
    attacking_spaces = []
    for direction in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        running_pos = np.add(pos, direction)
        if np.any(running_pos < 0):
            continue
        try:
            next_cell = layout[running_pos[0], running_pos[1]]
        except IndexError:
            continue
        attacking_spaces.append(running_pos)
    return attacking_spaces

def queen_valid_moves(pos, layout, color):
    valid_moves = []

    for direction in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        running_pos = np.array(pos)
        while True:
            running_pos += direction
            if np.any(running_pos < 0):
                break
            try:
                next_cell = layout[running_pos[0], running_pos[1]]
            except IndexError:
                break

            if next_cell in COLOR_TO_PIECES[color]:
                break
            elif next_cell == EMPTY or 'king' in next_cell:
                valid_moves.append(running_pos.copy())
            else:
                assert next_cell in COLOR_TO_PIECES[opposite_color(color)]
                valid_moves.append(running_pos.copy())
                break

    return valid_moves

def get_attacking_mask(layout, color):
    attacking_mask = np.zeros(layout.shape, dtype=bool)

    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            piece = layout[r, c]
            if piece not in COLOR_TO_PIECES[color]:
                continue

            if 'king' in piece:
                attacked_spaces = get_king_attacking_spaces((r, c), layout)
            else:
                attacked_spaces = PIECE_VALID_MOVES[piece]((r, c), layout)

            for space in attacked_spaces + [(r, c)]:
                attacking_mask[space[0], space[1]] = True

    return attacking_mask

class CheckmateTacticTwoPlayer(TwoPlayerGeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.1

    @staticmethod
    def transition(state, action):
        layout, current_player = state
        new_layout = layout.copy()

        selected_piece_rcs = np.argwhere((layout == HIGHLIGHTED_WHITE_QUEEN) | (layout == HIGHLIGHTED_WHITE_KING) | (layout == HIGHLIGHTED_BLACK_KING))
        if len(selected_piece_rcs) >= 1:
            if len(selected_piece_rcs) > 1:
                raise InvalidState()
            selected_piece_pos = selected_piece_rcs[0]

            if not CheckmateTacticTwoPlayer.action_is_valid(new_layout, action, selected_piece_pos):
                return (new_layout, current_player)

            CheckmateTacticTwoPlayer.execute_player_move(new_layout, action, selected_piece_pos)
            next_player = (current_player + 1) % 2

            return (new_layout, next_player)

        r, c = action
        token = new_layout[r, c]
        current_color = 'white' if current_player == 0 else 'black'
        if token not in COLOR_TO_PIECES[current_color]:
            return (new_layout, current_player)

        try:
            new_layout[r, c] = HIGHlIGHTED_ANALOGS[token]
        except KeyError:
            pass

        return (new_layout, current_player)

    @staticmethod
    def action_is_valid(layout, action, piece_pos):
        for valid_action in CheckmateTacticTwoPlayer.get_valid_moves(piece_pos, layout):
            if action[0] == valid_action[0] and action[1] == valid_action[1]:
                return True
        return False

    @staticmethod
    def get_valid_moves(piece_pos, layout):
        r, c = piece_pos
        piece = layout[r, c]
        return PIECE_VALID_MOVES[piece](piece_pos, layout)

    @staticmethod
    def execute_player_move(layout, new_piece_pos, old_piece_pos):
        old_r, old_c = old_piece_pos
        new_r, new_c = new_piece_pos

        piece = layout[old_r, old_c]
        if piece in UNHIGHLIGHTED_ANALOGS:
            piece = UNHIGHLIGHTED_ANALOGS[piece]

        layout[new_r, new_c] = piece
        layout[old_r, old_c] = EMPTY

    @staticmethod
    def checkmate(layout):
        for king, color in zip([BLACK_KING, WHITE_KING], ['black', 'white']):
            try:
                king_pos = np.argwhere(layout == king)[0]
            except IndexError:
                return False
            valid_moves = CheckmateTacticTwoPlayer.get_valid_moves(king_pos, layout)
            if len(valid_moves) > 0:
                continue

            attacking_mask = get_attacking_mask(layout, opposite_color(color))
            if attacking_mask[king_pos[0], king_pos[1]]:
                return True

        return False

    @staticmethod
    def stalemate(state):
        layout, player = state

        black_king_pos = np.argwhere((layout == BLACK_KING) | (layout == HIGHLIGHTED_BLACK_KING))[0]

        if player == 1 and len(CheckmateTacticTwoPlayer.get_valid_moves(black_king_pos, layout)) == 0:
            return True

        return False

    @staticmethod
    def compute_done(state):
        layout, player = state
        return CheckmateTacticTwoPlayer.checkmate(layout) or CheckmateTacticTwoPlayer.stalemate(state)

    @staticmethod
    def compute_reward(state0, action, state1):
        if CheckmateTacticTwoPlayer.checkmate(state1[0]):
            return 1.
        return 0.

    @classmethod
    def initialize_figure(cls, height, width):
        fig, ax = TwoPlayerGeneralizationGridGame.initialize_figure(height, width)

        # Draw a b&w grid in the background
        for r in range(height):
            for c in range(width):
                edge_color = '#888888'
                face_color = '#111111' if ((r + c) % 2) else 'white'
                
                drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                             numVertices=4,
                                             radius=0.5 * np.sqrt(2),
                                             orientation=np.pi / 4,
                                             ec=edge_color,
                                             fc=face_color)
                ax.add_patch(drawing)

        return fig, ax

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.1):
        if token == EMPTY:
            return None

        im = TOKEN_IMAGES[token]
        oi = OffsetImage(im, zoom = cls.fig_scale * (token_scale / max(height, width)**0.5))
        box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

        ax.add_artist(box)

        return box


def player1_policy(layout):
    if np.any(layout == BLACK_KING):
        return np.argwhere(layout == BLACK_KING)[0]
    black_king_pos = np.argwhere(layout == HIGHLIGHTED_BLACK_KING)[0]
    action = CheckmateTacticTwoPlayer.get_valid_moves(black_king_pos, layout)[0]
    return action

CheckmateTactic = VersusComputerTwoPlayerGeneralizationGridGameFactory("CheckmateTactic", CheckmateTacticTwoPlayer, player1_policy)

### Specific environments
rng = np.random.RandomState(0)
num_layouts = 20

def create_random_layout():
    height = rng.randint(5, 20)
    width = rng.randint(5, 20)

    layout = np.full((height, width), EMPTY, dtype=object)

    rank = rng.randint(2, width-2)
    layout[0, rank] = BLACK_KING
    layout[2, rank] = WHITE_KING

    attack_direction = rng.randint(4)

    if attack_direction == 0:
        r = 1
        c = rng.randint(rank+2, width)
    elif attack_direction == 1:
        r = 1
        c = rng.randint(0, rank-1)
    elif attack_direction == 2:
        r = 1
        c = rank
        delta = rng.randint(2, min(height, width-rank))
        r += delta
        c += delta
    elif attack_direction == 3:
        r = 1
        c = rank
        delta = rng.randint(2, min(height, rank+1))
        r += delta
        c -= delta

    layout[r, c] = WHITE_QUEEN

    k = rng.randint(4)
    layout = np.rot90(layout, k=k)
    
    return layout

layouts = [create_random_layout() for _ in range(num_layouts)]
create_gym_envs(CheckmateTactic, layouts, globals())
create_gym_envs(CheckmateTacticTwoPlayer, layouts, globals())

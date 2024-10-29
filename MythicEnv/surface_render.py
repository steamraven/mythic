from typing import Callable
import pygame.freetype
from pygame import Surface
from MythicEnv.env import MythicMischiefEnv
from MythicEnv.game import (
    BOOKS,
    BOOKS_MASK,
    CLUTTER,
    DEST,
    DEST_MASK,
    HORZ_WALL,
    KEEPER,
    PLAYER,
    PLAYER_MASK,
    PLAYER_SPECIAL,
    PLAYER_SPECIAL_MASK,
    RESPAWN,
    VERT_WALL,
    ActionType,
    Player,
    board_to_action,
)
from MythicEnv.game.teams.monsters import Monster
from MythicEnv.game.teams.vampires import Vampire
from MythicEnv.renderable import *
import numpy as np


def find_font(size: int):
    font_size = size - 1
    line_size = 0
    font = None
    while line_size < size:
        font_size += 1
        font = pygame.freetype.SysFont("unifont", font_size)
        line_size = font.get_sized_height(font_size)
    assert font
    assert font.fixed_width
    return font
    # print("debug: ", font_size, self.font.get_rect("Testy").width/5, self.font.get_sized_height(0))


# Icons
# TODO: use sprites
PLAYER_ICON = "웃"
KEEPER_ICON = "T"
BOOK_ICON = "▯"
CLUTTER_ICON = "X"
RESPAWN_ICON = "↺"
PLAYER_SPECIAL_ICON = "*"


class MythicMischiefRenderable(Renderable):
    def __init__(self, game: MythicMischiefEnv, font: pygame.freetype.Font):
        self.font = font
        self.game = game
        self.flex = Flex(
            horz=False,
            items=[
                Box(
                    Flex(
                        horz=True,
                        items=[
                            Box(
                                StaticText("Mythic Mischief   ", font),
                                color=BLACK,
                            ),
                            Box(
                                self.attr_box(
                                    0, RED, lambda player: "(0) pass", action=0
                                ),
                                color=BLACK,
                            ),
                            Box(
                                self.attr_box(
                                    1, BLUE, lambda player: "(0) pass", action=0
                                ),
                                color=BLACK,
                            ),
                            Box(
                                self.game_attr_box(
                                    WHITE,
                                    lambda: (
                                        "After Lunch"
                                        if game.game_state
                                        and game.game_state.after_lunch
                                        else "Before Lunch"
                                    ),
                                ),
                                color=BLACK,
                            ),
                        ],
                    )
                ),
                Flex(
                    horz=True,
                    items=[
                        self.player_board(0),
                        self.game_board(),
                        self.player_board(1),
                    ],
                ),
            ],
        )

    def game_board(self) -> Renderable:
        """Create the main game board GUI"""
        game = self.game
        game_state = game.game_state
        assert game_state
        board = game_state.board

        # Size grid according to size of Monospaced characters
        char_size = (
            self.font.get_rect(PLAYER_ICON).right,
            self.font.get_sized_height(0),
        )

        # useed for highlighting grid: Avaliable Action, and Action to Confirm
        # Only for board actions

        cell_action_types = set(
            [
                ActionType.SELECT_DEST,
                ActionType.SELECT_MYTHIC,
                ActionType.SELECT_SELF,
                ActionType.SELECT_OPP,
            ]
        )

        class GameGrid(Grid):
            "Override Grid for specific logic"

            def cell_bg(self, x: int, y: int):
                assert game.available_action_type is not None
                if game.available_action_type in cell_action_types:
                    if (
                        game.confirming_action
                        and board_to_action(x, y) == game.confirming_action
                    ):
                        return GREEN
                    if board_to_action(x, y) in game.available_actions:
                        return GRAY
                return BLACK

            def cell_border(self, x: int, y: int) -> BorderDef:
                # Render walls as borders
                data: np.uint16 = board[x, y]
                assert game.available_action_type is not None
                action_type = game.available_action_type

                available_color = BLUE if get_blink() else GRAY

                if (
                    action_type == ActionType.SELECT_HORZ_WALL
                    and game.confirming_action == board_to_action(x, y)
                ):
                    horz_border = BorderProps(GREEN, 3)
                elif (
                    action_type == ActionType.SELECT_HORZ_WALL
                    and board_to_action(x, y) in game.available_actions
                ):
                    horz_border = BorderProps(available_color, 5)
                elif data & HORZ_WALL:
                    horz_border = BorderProps(YELLOW, 3)
                else:
                    horz_border = None
                if (
                    action_type == ActionType.SELECT_VERT_WALL
                    and game.confirming_action == board_to_action(x, y)
                ):
                    vert_border = BorderProps(GREEN, 3)
                elif (
                    action_type == ActionType.SELECT_VERT_WALL
                    and board_to_action(x, y) in game.available_actions
                ):
                    vert_border = BorderProps(available_color, 5)
                elif data & VERT_WALL:
                    vert_border = BorderProps(YELLOW, 3)
                else:
                    vert_border = None

                return (
                    horz_border,
                    vert_border,
                )

            def render_cell(
                self, x: int, y: int, surface: Surface, bg: tuple[int, int, int]
            ):
                data: np.uint16 = board[x, y]

                # Rect is moved around
                char_rect = pygame.Rect(0, 0, char_size[0], char_size[1])

                # Mythic/Keeper
                if data & KEEPER:
                    self.font.render_to(surface, char_rect, KEEPER_ICON, YELLOW, bg)
                elif data & PLAYER_MASK:
                    if not (data & PLAYER[0]):
                        color = BLUE
                    elif not (data & PLAYER[1]):
                        color = RED
                    else:
                        color = RED if get_blink() else BLUE
                    self.font.render_to(surface, char_rect, PLAYER_ICON, color, bg)

                # Books
                if data & BOOKS_MASK:
                    if not (data & BOOKS[0]):
                        color = BLUE
                    elif not (data & BOOKS[1]):
                        color = RED
                    else:
                        color = RED if get_blink() else BLUE
                    self.font.render_to(
                        surface,
                        char_rect.move((char_size[0] * 2, 0)),
                        BOOK_ICON,
                        color,
                        bg,
                    )

                # Next Line
                char_rect.move_ip(0, char_size[1])
                # Clutter
                if data & CLUTTER:
                    self.font.render_to(surface, char_rect, CLUTTER_ICON, WHITE, bg)

                # Respawn
                if data & RESPAWN:
                    self.font.render_to(
                        surface,
                        char_rect.move((char_size[0] * 2, 0)),
                        RESPAWN_ICON,
                        WHITE,
                        bg,
                    )

                # Next Line
                char_rect.move_ip(0, char_size[1])

                # Keeper Dest
                if data & DEST_MASK:
                    for dest_no, dest in enumerate(DEST):
                        if data & dest:
                            self.font.render_to(
                                surface, char_rect, str(dest_no + 1), YELLOW, bg
                            )

                # Player special
                if data & PLAYER_SPECIAL_MASK:

                    if not (data & PLAYER_SPECIAL[0]):
                        color = BLUE
                    elif not (data & PLAYER_SPECIAL[1]):
                        color = RED
                    else:
                        color = RED if get_blink() else BLUE
                    self.font.render_to(
                        surface,
                        char_rect.move((char_size[0] * 2, 0)),
                        PLAYER_SPECIAL_ICON,
                        color,
                        bg,
                    )
                # self.font.render_to(surface, char_rect.move((self.char_size[0]*2, 0)), str(self.game.dest_costs[x,y]), WHITE, bg)

        return GameGrid(self.font, (5, 5), (char_size[0] * 3, char_size[1] * 3))

    def player_board(self, player_id: int) -> "Box":
        """Create a GUI for a player"""

        if player_id == 0:
            color = RED
        else:
            color = BLUE

        return Box(
            Flex(
                horz=False,
                items=[
                    self.attr_box(
                        player_id, color, lambda player: f"{player.team.data.name}"
                    ),
                    self.attr_box(
                        player_id, color, lambda player: f"Score: {player.score}"
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(1) Move: {BOOK_ICON*player.move_tomes}{player.move}",
                        action=1,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(2) Move Other ({player.team.data.move_other_attr}): {BOOK_ICON*player.move_other_tomes}{player.move_other}",
                        action=2,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(3) Move Horz Shelf ({player.team.data.move_shelf_attr}): {BOOK_ICON*player.move_shelf_tomes}{player.move_shelf}",
                        action=3,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(4) Move Vert Shelf ({player.team.data.move_shelf_attr}): {BOOK_ICON*player.move_shelf_tomes}{player.move_shelf}",
                        action=4,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(5) Distract: {BOOK_ICON*player.distract_tomes}{player.distract}",
                        action=5,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(6) Legendary ({player.team.data.legendary_attr}): {BOOK_ICON if player.legendary else ''}",
                        action=6,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"(7) After Lunch({player.team.data.after_lunch_attr}): {player.after_lunch}",
                        action=7,
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"Tomes : {BOOK_ICON * player.tomes}",
                    ),
                    self.attr_box(
                        player_id,
                        color,
                        lambda player: f"Unplaced : {PLAYER_ICON * (3-len(player.mythics))}",
                    ),
                ],
            ),
            color=color,
        )

    # hack to calc maxinum sizes
    MAX_PLAYERS = [
        Player(0, Vampire(None)),
        Player(1, Monster(None)),
    ]

    def attr_box(
        self,
        player_id: int,
        color: tuple[int, int, int],
        getter: Callable[[Player], str],
        action: Optional[int] = None,
    ):
        """Define a GUI for a single player attribute. Also handles background highlight for actions"""
        max_size = max(len(getter(player)) for player in self.MAX_PLAYERS)
        game = self.game
        game_state = game.game_state
        assert game_state

        class SubText(Text):
            def text(self) -> str:
                player = game_state.players[player_id]
                return getter(player)

            def bg(self) -> tuple[int, int, int]:
                assert game.available_action_type is not None
                if (
                    action is not None
                    and game.to_play == player_id
                    and (
                        game.available_action_type
                        in [ActionType.SELECT_SKILL, ActionType.PASS]
                    )
                ):
                    if action == game.confirming_action:
                        return GREEN
                    elif action in game.available_actions:
                        return GRAY
                return BLACK

        return SubText(self.font, max_size, color=color)

    def game_attr_box(
        self, color: tuple[int, int, int], getter: Callable[[], str]
    ) -> "Renderable":
        """Define a GUI for a game level attribute"""

        class SubText(Text):
            def text(self) -> str:
                return getter()

        return SubText(self.font, len(getter()), color=color)

    def render(self, surface: Surface):
        return self.flex.render(surface)

    def size(self) -> tuple[int, int]:
        return self.flex.size()

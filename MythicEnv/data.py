from dataclasses import dataclass
from . import *


@dataclass
class DestCard:
    """Before Lunch and after lunch cards"""

    dests: list[Coordinate]
    keeper_moves: int
    books: list[Coordinate]
    respawn: list[Coordinate]


# TODO: Add other cards
BEFORE_LUNCH = DestCard(
    keeper_moves=4,
    dests=[
        (0, 0),
        (1, 4),
        (4, 1),
    ],
    books=[
        (1, 0),
        (1, 3),
        (2, 4),
        (3, 2),
    ],
    respawn=[
        (0, 2),
        (0, 4),
        (2, 1),
        (3, 0),
        (4, 2),
        (4, 3),
    ],
)


@dataclass
class StartLayout:
    """Starting Layouts from book or cards"""

    clutter: list[Coordinate]
    horz_walls: list[Coordinate]
    vert_walls: list[Coordinate]


# TODO: Add other layouts
START_LAYOUTS = [
    StartLayout(
        clutter=[
            (1, 2),
            (2, 1),
            (2, 4),
            (4, 2),
        ],
        horz_walls=[
            (0, 0),
            (1, 3),
            (2, 2),
            (3, 0),
            (4, 1),
        ],
        vert_walls=[
            (0, 1),
            (1, 2),
            (2, 1),
            (2, 4),
            (3, 3),
        ],
    )
]




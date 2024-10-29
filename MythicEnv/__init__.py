from typing import NamedTuple


Coordinate = tuple[int, int]
Size = tuple[int,int]


class Wall(NamedTuple):
    wall_type: int
    pos: Coordinate

# Colors
# TODO: Refactor name to how they are used rather than color
import abc
from dataclasses import dataclass
from typing import Optional, cast

from pygame import Surface
import pygame.freetype
import pygame
from . import *
from ding.envs.env.base_env import BaseEnv

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (3, 173, 255)
GRAY = (100, 100, 100)
GREEN = (50, 255, 50)


@dataclass
class BorderProps:
    color: tuple[int, int, int]
    width: int


BorderDef = tuple[Optional[BorderProps], Optional[BorderProps]]


class Renderable(abc.ABC):
    """An (Abstract) renderable GUI Element"""

    @abc.abstractmethod
    def render(self, surface: Surface):
        """Render GUI Element"""
        ...

    @abc.abstractmethod
    def size(self) -> Coordinate:
        """Calculate size of GUI Element"""


class Grid(Renderable):
    """Grid of items"""

    def __init__(
        self,
        font: pygame.freetype.Font,
        size: tuple[int, int],
        cell_size: tuple[int, int],  # Cell size without padding
    ):
        self.font = font
        self.cell_size = cell_size
        self._size = size
        self.pad = 5

    def cell_bg(self, x: int, y: int) -> tuple[int, int, int]:
        """Background of cell"""
        return BLACK

    def cell_border(self, x: int, y: int) -> BorderDef:
        """Calculate dynamic border of cell"""
        return (None, None)

    @abc.abstractmethod
    def render_cell(self, x: int, y: int, surface: Surface, bg: tuple[int, int, int]):
        """Render a cell on given surface (without padding)"""
        ...

    # @override
    def render(self, surface: Surface):
        """Render grid and each cell in its own subsurface"""
        size = self.size()
        pad = self.pad
        cell_width = self.cell_size[0] + pad + pad
        cell_height = self.cell_size[1] + pad + pad

        x_start = pad
        y_start = pad

        # Grid backgrounds
        # Draw background highlight first
        for x in range(self._size[0]):
            x_pos = x_start + cell_width * x
            for y in range(self._size[1]):
                y_pos = y_start + cell_height * y
                bg = self.cell_bg(x, y)
                if bg != BLACK:
                    pygame.draw.rect(
                        surface, bg, pygame.Rect(x_pos, y_pos, cell_width, cell_height)
                    )
        # Grid lines
        x_end = size[0] - pad
        y_end = size[1] - pad

        x_pos = x_start
        for _ in range(self._size[0] + 1):
            pygame.draw.line(surface, WHITE, (x_pos, y_start), (x_pos, y_end))
            x_pos += cell_width

        y_pos = y_start
        for _ in range(self._size[1] + 1):
            pygame.draw.line(surface, WHITE, (x_start, y_pos), (x_end, y_pos))
            y_pos += cell_height

        # grid contents

        for x in range(self._size[0]):
            x_pos = x_start + cell_width * x
            for y in range(self._size[1]):
                y_pos = y_start + cell_height * y

                # Borders
                border_def = self.cell_border(x, y)

                start_border = (x_pos, y_pos + cell_height)

                for prop in border_def:
                    if prop:
                        pygame.draw.line(
                            surface,
                            prop.color,
                            start_border,
                            (x_pos + cell_width, y_pos + cell_height),
                            width=prop.width,
                        )

                    start_border = (x_pos + cell_width, y_pos)

                # Render contents
                rect = pygame.Rect(
                    x_pos + pad, y_pos + pad, self.cell_size[0], self.cell_size[1]
                )
                bg = self.cell_bg(x, y)

                self.render_cell(x, y, surface.subsurface(rect), bg)

    # @override
    def size(self) -> tuple[int, int]:
        """Size of grid with padding"""
        pad2 = self.pad * 2
        grid_size = (
            (self.cell_size[0] + pad2) * self._size[0] + pad2,
            (self.cell_size[1] + pad2) * self._size[1] + pad2,
        )
        return grid_size


class Flex(Renderable):
    """A grouping of GUI Elements. Horizontal or Vertical"""

    def __init__(self, horz: bool, items: Optional[list[Renderable]] = None):
        self.horz = horz
        if items is None:
            items = []
        self.items = items

    # @override
    def size(self) -> Coordinate:
        """Size of the entire flex"""
        sizes = [i.size() for i in self.items]
        if self.horz:
            height = max(s[1] for s in sizes)
            self.chunks = [s[0] for s in sizes]
            return (sum(self.chunks), height)
        else:
            width = max(s[0] for s in sizes)
            self.chunks = [s[1] for s in sizes]
            return (width, sum(self.chunks))

    # @override
    def render(self, surface: Surface):
        """Render each part within a subsurface"""
        if self.horz:
            flex_size = surface.get_size()[1]
        else:
            flex_size = surface.get_size()[0]
        chunk_pos = 0

        for i, i_size in zip(self.items, self.chunks):
            if self.horz:
                sub = surface.subsurface(pygame.Rect(chunk_pos, 0, i_size, flex_size))
            else:
                sub = surface.subsurface(pygame.Rect(0, chunk_pos, flex_size, i_size))
            i.render(sub)
            chunk_pos += i_size


class Text(Renderable):
    """A text GUI Element.  Subclass to make dynamic"""

    def __init__(
        self,
        font: pygame.freetype.Font,
        max_size: int,
        color: tuple[int, int, int] = WHITE,
    ):
        self.font = font
        self.max_size = max_size
        self.color = color

    # @override
    def size(self) -> Coordinate:
        width = self.font.get_rect("X" * self.max_size).width
        return (width, self.font.get_sized_height(0))

    # @override
    def render(self, surface: Surface):
        text = self.text()
        bg = self.bg()
        self.font.render_to(surface, (0, 0), text, self.color, bg)

    def text(self) -> str:
        return ""

    def bg(self) -> tuple[int, int, int]:
        return BLACK


class StaticText(Text):
    """Static text"""

    def __init__(
        self, text: str, font: pygame.freetype.Font, color: tuple[int, int, int] = WHITE
    ):
        self.static_text = text
        super().__init__(font, len(text), color)

    # @override
    def text(self) -> str:
        return self.static_text


class Box(Renderable):
    """Create a box around an GUI item. Fills parent area"""

    def __init__(
        self,
        inside: Renderable,
        pad: int = 10,
        line_width: int = 1,
        color: tuple[int, int, int] = WHITE,
    ):
        self.inside = inside
        self.pad = pad
        self.line_width = line_width
        self.color = color

    # @override
    def render(self, surface: Surface):
        rect = pygame.Rect(0, 0, *surface.get_size())

        rect.inflate_ip((-self.pad, -self.pad))
        pygame.draw.rect(surface, self.color, rect, width=self.line_width)
        rect.inflate_ip((-self.pad, -self.pad))
        self.inside.render(surface.subsurface(rect))

    # @override
    def size(self) -> tuple[int, int]:
        inside_size = self.inside.size()
        offset = self.pad * 2
        return (inside_size[0] + offset, inside_size[1] + offset)


class RenderableEnv(Renderable):
    """Wrap a Env to be usable with Renderable framewosk"""

    def __init__(self, env: BaseEnv):
        self.env = env

    # @override
    def render(self, surface: Surface):
        rendered = cast(pygame.Surface, self.env.render())
        surface.blit(rendered, (0, 0))

    # @override
    def size(self) -> tuple[int, int]:
        rendered = cast(pygame.Surface, self.env.render())
        return rendered.get_size()


g_blink = False
def toggle_blink():
    global g_blink
    g_blink = not g_blink

def get_blink() -> bool:
    global g_blink
    return g_blink

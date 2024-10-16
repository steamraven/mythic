
import abc
import copy
import random
from typing import Any, Callable, Generator, NamedTuple, Optional, cast

import numpy as np
import pygame
from pygame import Surface

from easydict import EasyDict

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
import pygame.freetype
import pygame.locals
from dataclasses import dataclass
from enum import IntEnum

Coordinate = tuple[int,int]


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
    dests= [
        (0,0),
        (1,4),
        (4,1),
    ],
    books= [
        (1,0),
        (1,3),
        (2,4),
        (3,2),
    ],
    respawn= [
        (0,2),
        (0,4),
        (2,1),
        (3,0),
        (4,2),
        (4,3),
    ]

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
        clutter = [
            (1,2),
            (2,1),
            (2,4),
            (4,2),
        ],
        horz_walls = [
            (0,0),
            (1,3),
            (2,2),
            (3,0),
            (4,1),
        ],
        vert_walls = [
            (0,1),
            (1,2),
            (2,1),
            (2,4),
            (3,3),
        ]
    )
]

@dataclass
class Team():
    """Team specific info"""
    id_: int
    name: str
    move_other_attr: str
    move_shelf_attr: str
    legendary_attr:str
    after_lunch_attr: str 

#TODO: Add more teams
VAMPIRE = Team(
    id_ = 0,
    name = "Vampire",
    move_other_attr = "Lure",
    move_shelf_attr = "Conceal",
    legendary_attr =  "Trance",
    after_lunch_attr = "Blood",
)

MONSTER = Team(
    id_ = 1,
    name = "Monster",
    move_other_attr = "Throw",
    move_shelf_attr = "Barge",
    legendary_attr =  "Intemidate",
    after_lunch_attr = "Trap",
)


@dataclass
class Player:
    """All the data relevent to a player"""
    id_: int # maps to index in game.players and PLAYERS and To_Play
    team: Team

    mythics: set[Coordinate]

    move: int
    move_tomes: int

    move_other: int
    move_other_tomes: int

    move_shelf: int
    move_shelf_tomes: int

    distract: int
    distract_tomes: int

    legendary: int
    after_lunch: int

    tomes: int
    score: int

    overlap: Coordinate | None


# Colors
# TODO: Refactor name to how they are used rather than color
WHITE = (255,255,255)
BLACK = (0,0,0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (3, 173, 255)
GRAY = (100,100,100)
GREEN = (50, 255, 50)

# Bit flags for board
# TODO: Use Enum
DEST = [1<<0, 1<<1, 1<<2]
DEST_MASK = 1<<0 | 1<<1 | 1<<2
DEST_SHIFT = 0
KEEPER = 1<<3
BOOKS = [1<<4, 1<<5]
BOOKS_MASK = 1<<4 | 1<<5
RESPAWN = 1 << 6
CLUTTER = 1 << 7
HORZ_WALL = 1 << 8
VERT_WALL = 1 << 9
PLAYER = [1<<10, 1 << 11]
PLAYER_MASK = 1<<10 | 1<<11
PLAYER_SPECIAL = [1<<12, 1 << 13]
PLAYER_SPECIAL_MASK = 1<<12 | 1<<13

class Action(IntEnum):
    """Specific Actions that a player can take"""
    PASS = 0
    MOVE = 1
    MOVE_OTHER = 2
    MOVE_HORZ_SHELF = 3
    MOVE_VERT_SHELF = 4
    DISTRACT = 5
    LEGENDARY = 6
    AFTER_LUNCH = 7

class ActionPhase(IntEnum):
    """Describes what multistep action is being fulfilled"""
    PLACE_MYTHIC = 0
    USE_SKILL = 1
    MOVE = 2
    MOVE_OPP = 3
    MOVE_HORZ_SHELF = 4
    MOVE_VERT_SHELF = 5
    DISTRACT = 6
    LEGENDARY = 7
    AFTER_LUNCH = 8
    MOVE_KEEPER = 9
    SPEND_TOME = 10
    BOOST = 11
    DECAY_MOVE = 12
    END_GAME = 13

class ActionType(IntEnum):
    SELECT_SELF = 0
    SELECT_OPP = 1
    SELECT_MYTHIC = 2
    SELECT_DEST = 3
    SELECT_HORZ_WALL = 4
    SELECT_VERT_WALL = 5
    SELECT_SKILL = 6
    PASS = 7




# Global blink variable. Is toggled by a pygame timer
g_blink = False

class PlayYield(NamedTuple):
    """Result of a Play yield"""
    to_play: int
    action_phase: ActionPhase
    actions_left: int
    action_type: ActionType
    available_actions: list[int]

# Type alias for Play sub-functions
# Does not end game
PlayGenerator = Generator[
            PlayYield,
            int,
            None]
# Can end game
PlayDoneGenerator = Generator[
            PlayYield,
            int,
            tuple[bool, int]]

class PlayableEnv(BaseEnv):
    """An env that can be played"""
    action_names: list[str]
    @abc.abstractmethod
    def confirm_action(self, action: int) -> bool:
        ...

@ENV_REGISTRY.register('mythic_mischief_v0') 
class MythicMischiefEnv(PlayableEnv):
    """Env for Mythic Mischief"""
    players: tuple[Player, Player]
    render_mode: str
    battle_mode: str

    config: dict[str, Any] = dict(
         # (str) The name of the environment registered in the environment registry.
        env_id="MythicMischief_v0",
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (str) The mode of the environment when doing the MCTS.
        battle_mode_in_simulation_env='self_play_mode',
        # (str) The render mode. Options are 'None', 'human', 'rgb_array'
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        # replay_path=None,
        # (str) The type of the bot of the environment.
        # bot_action_type='rule',
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        # agent_vs_human=False,
        # (float) The probability that a random agent is used instead of the learning agent.
        # prob_random_agent=0,
        # (float) The probability that an expert agent(the bot) is used instead of the learning agent.
        # prob_expert_agent=0,
        # (float) The probability that a random action will be taken when calling the bot.
        # prob_random_action_in_bot=0.,
        # (float) The scale of the render screen.
        # screen_scaling=9,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        # channel_last=False,
        # (bool) Whether to scale the observation.
        # scale=False,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        # stop_value=2,
    )
    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'  # type: ignore
        return cfg
    

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        # Load the config.
        self.cfg = cfg
        assert cfg

        self.render_mode = cfg['render_mode']

        self.battle_mode = cfg['battle_mode']
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'

        # play all keeper moves, even when there is only one option
        self.all_keeper_moves = False

        # ensure safe defaults. Most set in reset
        self.board:np.ndarray[Coordinate, np.dtype[np.uint16]] = np.ndarray((5, 5), np.uint16)
        self.available_actions = list[int]()
        self.confirming_action = None
        self.after_lunch = False

        # Setup pygame GUI
        if self.render_mode == 'surface':
            
            if not pygame.get_init():
                pygame.init()
            if not pygame.freetype.get_init():
                pygame.freetype.init()

            # Choose a font size
            # GNU unifont has a LOT of unicode characters
            font_size = 20 -1
            line_size = 0
            font = None
            while line_size < 20:    
                font_size += 1
                font = pygame.freetype.SysFont('unifont', font_size)
                line_size = font.get_sized_height(font_size)
            assert font
            assert font.fixed_width
            self.font = font
            #print("debug: ", font_size, self.font.get_rect("Testy").width/5, self.font.get_sized_height(0))

            self.renderable = Flex(horz=False, items=[
                Box(
                    Flex(horz=True, items =[
                        Box(StaticText("Mythic Mischief   ", self.font),color=BLACK),
                        Box(self.attr_box(0, RED, lambda player: f"(0) pass", action=0), color=BLACK),
                        Box(self.attr_box(1, BLUE, lambda player: f"(0) pass", action=0), color=BLACK),
                        Box(self.game_attr_box( WHITE, lambda: "After Lunch" if self.after_lunch else "Before Lunch"), color=BLACK),
                    ])
                ),
                Flex(horz=True, items=[
                    self.player_board(0),
                    self.game_board(),
                    self.player_board(1),
                ]),
            ])
            self.surface_size = self.renderable.size()

        # TODO: Define action_space and observation space

        # Crate easier names for actions
        self.action_names = ([str(action) for action in Action] 
                            + [f"{x},{y}" for x,y in (action_to_board(i+len(Action)) for i in range(5*5))])


    def game_board(self) -> "Renderable":
        """Create the main game board GUI"""
        game = self
        board = game.board

        # Size grid according to size of Monospaced characters
        char_size = (
            self.font.get_rect(PLAYER_ICON).right,
            self.font.get_sized_height(0),
        )

        # useed for highlighting grid: Avaliable Action, and Action to Confirm
        # Only for board actions
        available_actions = [action_to_board(a) for a in game.available_actions if a>=len(Action)]
        if game.confirming_action and game.confirming_action >= len(Action):
            confirming_action = action_to_board(game.confirming_action)
        else:
            confirming_action = None

        class GameGrid(Grid):
            "Override Grid for specific logic"
            
            def bg(self, x: int, y: int):
                if (x, y) == confirming_action:
                    return GREEN
                if (x, y) in available_actions:
                    return GRAY
                return BLACK

            def border(self, x: int, y: int) -> BorderDef:
                # Render walls as borders
                data: np.uint16 = board[x, y]
                return (
                    BorderProps(YELLOW, 3) if data & HORZ_WALL else None,
                    BorderProps(YELLOW, 3) if data & VERT_WALL else None,
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
                        color = RED if g_blink else BLUE
                    self.font.render_to(surface, char_rect, PLAYER_ICON, color, bg)

                # Books
                if data & BOOKS_MASK:
                    if not (data & BOOKS[0]):
                        color = BLUE
                    elif not (data & BOOKS[1]):
                        color = RED
                    else:
                        color = RED if g_blink else BLUE
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
                        color = RED if g_blink else BLUE
                    self.font.render_to(
                        surface,
                        char_rect.move((char_size[0] * 2, 0)),
                        PLAYER_SPECIAL_ICON,
                        color,
                        bg,
                    )
                # self.font.render_to(surface, char_rect.move((self.char_size[0]*2, 0)), str(self.game.dest_costs[x,y]), WHITE, bg)

        return GameGrid(self.font, (5, 5), (char_size[0] * 3, char_size[1] * 3))

        """Create a GUI for a player"""

        if player_id == 0:
            color = RED
        else:
            color = BLUE

        return Box(Flex(horz=False, items=[
           self.attr_box(player_id, color, lambda player: f"{player.team.name}"),
           self.attr_box(player_id, color, lambda player: f"Score: {player.score}"),
           self.attr_box(player_id, color, lambda player: f"(1) Move: {BOOK_ICON*player.move_tomes}{player.move}", action=1),
           self.attr_box(player_id, color, lambda player: f"(2) Move Other ({player.team.move_other_attr}): {BOOK_ICON*player.move_other_tomes}{player.move_other}", action=2),
           self.attr_box(player_id, color, lambda player: f"(3) Move Horz Shelf ({player.team.move_shelf_attr}): {BOOK_ICON*player.move_shelf_tomes}{player.move_shelf}", action=3), 
           self.attr_box(player_id, color, lambda player: f"(4) Move Vert Shelf ({player.team.move_shelf_attr}): {BOOK_ICON*player.move_shelf_tomes}{player.move_shelf}", action=4), 
           self.attr_box(player_id, color, lambda player: f"(5) Distract: {BOOK_ICON*player.distract_tomes}{player.distract}", action=5), 
           self.attr_box(player_id, color, lambda player: f"(6) Legendary ({player.team.legendary_attr}): {BOOK_ICON if player.legendary else ''}", action=6),
           self.attr_box(player_id, color, lambda player: f"(7) After Lunch({player.team.after_lunch_attr}): {player.after_lunch}", action=7),
           self.attr_box(player_id, color, lambda player: f"Tomes : {BOOK_ICON * player.tomes}"),
           self.attr_box(player_id, color, lambda player: f"Unplaced : {PLAYER_ICON * (3-len(player.mythics))}"),

        ]), color = color)

    # hack to calc maxinum sizes
    MAX_PLAYERS = [
        Player(1, VAMPIRE, set(), 9, 3, 3, 4, 3, 4, 3, 4, 1, 1, 4, 9, None),
        Player(1, MONSTER, set(), 9, 3, 3, 4, 3, 4, 3, 4, 1, 1, 4, 9, None),
    ]
    def attr_box(self, player_id: int, color: tuple[int,int,int], getter: Callable[[Player], str], action: Optional[int] = None):
        """Define a GUI for a single player attribute. Also handles background highlight for actions"""
        max_size = max(len(getter(player)) for player in self.MAX_PLAYERS)
        game = self
        class SubText( Text):
            def text(self) -> str:
                player = game.players[player_id]
                return getter(player)
            def bg(self) ->  tuple[int,int,int]:
                if action is not None and game.to_play == player_id and action == game.confirming_action:
                    return GREEN
                elif action is not None and game.to_play == player_id and action in game.available_actions:
                    return GRAY
                else:
                    return BLACK
        return SubText(self.font, max_size, color=color)
    def game_attr_box(self, color: tuple[int,int,int], getter: Callable[[], str]) -> "Renderable":
        """ Define a GUI for a game level attribute"""
        class SubText( Text):
            def text(self) -> str:
                return getter()
        return SubText(self.font, len(getter()), color=color)

    def reset(self, start_player_index: int = 0, init_state: Optional[Any] = None,
              replay_name_suffix: Optional[str] = None) -> dict[str,Any]:
        """Reset the game state"""
        self.after_lunch = False
        self.confirming_action = None

        #TODO: Randomize cards
        self.dest_card = BEFORE_LUNCH
        self.start_layout = START_LAYOUTS[0]

        # initialize board
        self.board.fill(0)
        self.dests = copy.deepcopy(self.dest_card.dests)
        for i, (x,y) in enumerate(self.dest_card.dests):
            self.board[x,y] |= DEST[i]

        assert self.dest_card.keeper_moves <=4
        x,y = self.keeper = (2,2)
        self.board[x,y] |= KEEPER

        assert len(self.dest_card.books) <= 4
        for x,y in self.dest_card.books:
            self.board[x,y] |= BOOKS[0] | BOOKS[1]

        for x,y in self.dest_card.respawn:
            self.board[x,y] |=RESPAWN

        for x,y in self.start_layout.clutter:
            self.board[x,y] |= CLUTTER

        for x,y in self.start_layout.horz_walls:
            self.board[x,y] |= HORZ_WALL

        for x,y in self.start_layout.vert_walls:
            self.board[x,y] |= VERT_WALL

        # Player initialization
        # TODO: Randomize player teams
        self.players = (
            Player(0, VAMPIRE, set(), 3, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, None),
            Player(1, MONSTER, set(), 3, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, None),
        )
        # second player gets an extra tome to start
        self.players[1].tomes += 1

        # test data
        if False:
            # Random starting positions
            mask = PLAYER_MASK | KEEPER | BOOKS_MASK

            available = [(i,j) for i in range(5) for j in range(5) if not (self.board[i][j] & mask)]

            starts = random.sample(available, 6)

            for (x,y) in starts[:3]:
                self.board[x,y] |= PLAYER[0]
                self.players[0].mythics.add((x,y))
            for (x,y) in starts[3:]:
                self.board[x,y] |= PLAYER[1]
                self.players[1].mythics.add((x,y))

        # create play generator/co-routine
        self.play = self.start_play()
        
        # Start the play sequence
        self.to_play, _, _, _, self.available_actions = next(self.play)


        return {
            "obs": None,
            "available_actions": self.available_actions,
            "to_play": self.to_play
        }
    
    def confirm_action(self, action:int):
        """Confirm if action is valid and highlight in gui"""
        if action in self.available_actions:
            self.confirming_action = action
            return True
        self.confirming_action = None
        return False

    def step(self, action: int) -> BaseEnvTimestep:
        """Step through play"""
        assert action in self.available_actions, f"invalid action: {action}. {action_to_board(action) if action >= len(Action) else ''}  "
        # We are taking the action, so confirmation logic is reset
        self.confirming_action = None
        action_phase = actions_left = action_type = None
        try:
            # Send action to Play co-routine and get availible next steps
            # This results are what the co-routine yields
            self.to_play, action_phase, actions_left, action_type, self.available_actions = self.play.send(action)
            assert action_phase is not None and actions_left and action_type is not None

            meta: dict[str, Any] = {
                "player_name": self.players[self.to_play].team.name.title(),
                "action_phase": action_phase.name.replace("_", " ").title() ,
                "actions_left": actions_left,
                "action_type": action_type.name.replace("_", " ").title(),
            }

            return BaseEnvTimestep({
                "obs": None,
                "available_actions": self.available_actions,
                "to_play": self.to_play
            }, 0, False, meta)
        except StopIteration as e:
            # Play co-routine returned  
            done, reward = e.value
        assert done


        
        # Done.  Ensure to_play for calling logic
        meta = {
            "player_name": self.players[self.to_play].team.name.title(),
        }
        return BaseEnvTimestep({"to_play": self.to_play}, reward, True, meta)
    
    def start_play(self) -> PlayDoneGenerator:
        """Play generator. Takes action, yield next available actions, and returns reward when done"""
        players = self.players

        # Setup
        yield from self.place_mythics(players[0], True)

        # Player 2 gets a tome, but does not affect start boost
        start_boost = players[1].move_shelf
        yield from self.spend_tomes(players[1])
        players[1].move_shelf = start_boost

        yield from self.place_mythics(players[1], True)

        # Main loop
        while True:
            for player in players:
                done, reward = yield from self.mythic_phase(player)
                if done:
                    return done, reward
                done, reward = yield from self.keeper_phase(player)
                if done:
                    return done, reward
                yield from self.cleanup_phase(player)

    def place_mythics(self, player: Player, anywhere: bool) -> PlayGenerator:
        """Place available mythics in spot in an available spot"""
        any_mask = PLAYER_MASK | KEEPER | BOOKS_MASK
        respawn_mask = PLAYER_MASK | KEEPER

        available = [] # to make typechecker happy

        while(len(player.mythics) < 3):
            if not anywhere:
                available = [respawn for respawn in self.dest_card.respawn if not (self.board[respawn] & respawn_mask)]
                if len(available) == 0:
                    #if we can't place now, we wont be able to anytime this call. Start searching anywhere
                    anywhere = True
            if anywhere:
                available = [(x,y) for x in range(5) for y in range(5) if not (self.board[x,y] & any_mask)]
    
            action = yield PlayYield(
                player.id_, 
                ActionPhase.PLACE_MYTHIC, 
                3-len(player.mythics), 
                ActionType.SELECT_DEST,
                [board_to_action(x,y) for x,y in available]
            )
            pos = action_to_board(action)
            self.board[pos] |= PLAYER[player.id_]
            player.mythics.add(pos)

    def spend_tomes(self, player: Player) -> PlayGenerator:
        """Spend all collected tomes on skills"""
        
        # Online game forces spend

        self.reset_skills(player) # This resets starting boost. See start_play
        while player.tomes:
            available = list[int]()
            if player.move_tomes < 3:
                available.append(Action.MOVE)
            if player.move_other_tomes < 4:
                available.append(Action.MOVE_OTHER)
            if player.move_shelf_tomes < 4:
                available.append(Action.MOVE_HORZ_SHELF)
                available.append(Action.MOVE_VERT_SHELF)
            if player.distract_tomes < 4:
                available.append(Action.DISTRACT)
            if player.legendary < 1:
                available.append(Action.LEGENDARY)
            action = yield PlayYield(
                player.id_, 
                ActionPhase.SPEND_TOME, 
                player.tomes, 
                ActionType.SELECT_SKILL,
                available
            )
            player.tomes -= 1
            if action == Action.MOVE:
                player.move_tomes += 1
            elif action == Action.MOVE_OTHER:
                player.move_other_tomes += 1
            elif action == Action.MOVE_HORZ_SHELF or action == Action.MOVE_VERT_SHELF:
                player.move_shelf_tomes += 1
            elif action == Action.DISTRACT:
                player.distract_tomes += 1
            elif action == Action.LEGENDARY:
                player.legendary = 1
            self.reset_skills(player)

    def reset_skills(self, player: Player):
        """Reset all skills based on spent tomes"""
        player.move = [3,4,5,6,7,8,9][player.move_tomes*2]
        player.move_other = [1,1,2,2,3][player.move_other_tomes]
        player.move_shelf = [0,1,2,2,3][player.move_shelf_tomes]
        player.distract = [0,1,1,2,3][player.distract_tomes]

    def mythic_phase(self, player: Player) -> PlayDoneGenerator:
        """Phase where actions are performed by the mythics"""
        yield from self.place_mythics(player, False)

        # TODO: all skills/abilities
        while True:

            available: list[int] = [Action.PASS]

            if player.overlap:
                # Cannot stop movement on another players space.  Or activate other actins
                yield from self.do_move(player)
                continue

            do_move = self.do_move(player)
            available.extend(next(do_move).available_actions)
            # if self.can_move_other(player):
            #    available.append(Action.MOVE_OTHER)
            # if self.can_move_shelf(player, horz = True):
            #    available.append(Action.MOVE_HORZ_SHELF)
            # if self.can_move_shelf(player, horz = False):
            #    available.append(Action.MOVE_VERT_SHELF)
            do_distract = self.do_distract(player)
            available.extend(next(do_distract).available_actions)

            action = yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, available
            )

            if action == Action.MOVE:
                yield from do_move
            elif action == Action.DISTRACT:
                done, reward = yield from do_distract
                if done:
                    return done, reward
            elif action == Action.PASS:
                return False, 0

    def do_move(self, player: Player) -> PlayGenerator:

        player_mask = PLAYER[player.id_]
        other_player_mask = PLAYER[player.id_ ^ 1]
        other_player = self.players[player.id_ ^ 1]
        not_available_mask = KEEPER | player_mask

        def find_available_moves(
            x: int, y: int, remaining: int
        ) -> list[tuple[Coordinate, int]]:
            available = list[tuple[Coordinate, int]]()

            def inner(x_: int, y_: int):
                if self.board[x_, y_] & not_available_mask:
                    return
                cost = 2 if self.board[x_, y_] & CLUTTER else 1
                if self.board[x_, y_] & other_player_mask:
                    cost += min(
                        a[1] for a in find_available_moves(x_, y_, remaining - cost)
                    )
                if remaining >= cost:
                    available.append(((x_, y_), cost))

            if y < 4 and not (self.board[x, y] & HORZ_WALL):
                inner(x, y + 1)
            if x < 4 and not (self.board[x, y] & VERT_WALL):
                inner(x + 1, y)
            if y > 0 and not (self.board[x, y - 1] & HORZ_WALL):
                inner(x, y - 1)
            if x > 0 and not (self.board[x - 1, y] & VERT_WALL):
                inner(x - 1, y)
            return available

        available_moves = dict[Coordinate, list[tuple[Coordinate, int]]]()

        if player.overlap:
            mythics = [player.overlap]
            assert player.move
        else:
            mythics = player.mythics

        if player.move:
            for mythic in mythics:
                self.board[mythic] ^= player_mask
                available_moves[mythic] = find_available_moves(*mythic, player.move)
                self.board[mythic] |= player_mask

        # overlap = player.mythics & other_player.mythics

        # def check_move(move: Coordinate, cost: int):
        #     """Simulate a move and Test if a is valid. Mainly that ALL overlaps still have moves"""
        #     # Second have of mythic move: Add to new position
        #     self.board[move] |= player_mask
        #     remaining = player.move - cost
        #     try:
        #         for o_mythic in overlap:
        #             # Don't test against self
        #             if mythic == o_mythic:
        #                 continue

        #             # Remove the overlap mythic to find available moves
        #             self.board[o_mythic] ^= player_mask
        #             try:
        #                 # if no available moves for the overlap, this move is invalid
        #                 if not find_available_moves(*o_mythic, remaining):
        #                     return False
        #             finally:
        #                 # Reverse removal of overlap mythic
        #                 self.board[o_mythic] |= player_mask
        #     finally:
        #         # Make sure to reverse reverse move
        #         self.board[move] ^= player_mask
        #     return True

        # # Test if all overlaps are still valid after each available move
        # if overlap:
        #     for mythic,available in available_moves.items():
        #         # Move the mythic. Start by removing from current position
        #         self.board[mythic] ^= player_mask

        #         available_moves[mythic] = [move for move in available if check_move(*move)]

        #         # Make sure to reverse move
        #         self.board[mythic] |= player_mask

        if any(v for v in available_moves.values()):
            yield PlayYield(
                player.id_, ActionPhase.MOVE, 1, ActionType.SELECT_SKILL, [Action.MOVE]
            )
        else:
            yield PlayYield(
                player.id_, ActionPhase.MOVE, 1, ActionType.SELECT_SKILL, []
            )

        # Select mythic to move
        # available = [board_to_action(x,y) for x,y in player.mythics]

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE,
            2,
            ActionType.SELECT_SELF,
            [board_to_action(*k) for k, v in available_moves.items() if v],
        )

        # Mythic Selected to move

        mythic = action_to_board(action)

        available = available_moves[mythic]

        if not available:
            # TODO: Should not reach; change can_move to handle this
            return

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE,
            1,
            ActionType.SELECT_DEST,
            [board_to_action(*a[0]) for a in available],
        )

        # Destination chosen

        # Actually Move mythic
        self.board[mythic] ^= PLAYER[player.id_]
        player.mythics.remove(mythic)
        mythic = action_to_board(action)

        if self.board[mythic] & CLUTTER:
            cost = 2
        else:
            cost = 1

        player.move -= cost
        self.board[mythic] |= PLAYER[player.id_]
        player.mythics.add(mythic)

        # check for book
        if self.board[mythic] & BOOKS[player.id_]:
            self.board[mythic] ^= BOOKS[player.id_]
            player.tomes += 1

        player.overlap = mythic if self.board[mythic] & other_player else None

        return

    def do_distract(self, player: Player) -> PlayOrDoneGenerator:
        yield PlayYield(
            player.id_,
            ActionPhase.MOVE,
            1,
            ActionType.SELECT_SKILL,
            [Action.MOVE] if player.distract > 0 else [],
        )

        # Select mythic to perform distract
        available = [board_to_action(x,y) for x,y in player.mythics]

        action = yield PlayYield(
            player.id_,
            ActionPhase.DISTRACT,
            2,
            ActionType.SELECT_SELF,
            available
        )
        dest = action_to_board(action)
        x,y = self.keeper

        # find shortest path and options
        dest_costs = self.calc_dest_costs(dest)
        
        available = list[int]()
        best_cost = 255

        def inner(x:int, y:int, best_cost: int):
            """Inner function to reuse finicky code"""
            cost = dest_costs[x,y]
            if cost < best_cost:
                available.clear()
                best_cost = cost
            if cost == best_cost:
                available.append(board_to_action(x,y))  
            return best_cost 

        if y < 4 and not (self.board[x,y] & HORZ_WALL):
            best_cost = inner(x,y+1, best_cost)
        if x < 4 and not (self.board[x,y] & VERT_WALL):
            best_cost = inner(x+1,y, best_cost)
        if y > 0 and not (self.board[x,y-1] & HORZ_WALL):
            best_cost = inner(x,y-1, best_cost) 
        if x>0 and not (self.board[x-1,y] & VERT_WALL):
            best_cost = inner(x-1,y, best_cost) 

        action = yield PlayYield(
            player.id_,
            ActionPhase.DISTRACT,
            1,
            ActionType.SELECT_DEST,
            available
        )

        dest = action_to_board(action)
        done, _ = self.move_keeper(dest)
        if done:
            return (yield from self.end_game())

        return False, 0
        yield

    def move_keeper(self,dest: Coordinate) -> tuple[bool, bool]:
        # Move Keeper
        popped_dest = False
        x,y = self.keeper
        self.board[x,y] ^= KEEPER

        self.board[dest] |= KEEPER
        self.keeper = dest

        # check if keeper reached dest
        if self.dests and self.keeper == self.dests[0]:
            self.board[self.dests[0]] ^= DEST[3-len(self.dests)]
            _ = self.dests.pop(0)
            popped_dest = True

        # Check if keeper caught any mythics
        if self.board[dest] & PLAYER_MASK:
            for player_id, mask in enumerate(PLAYER):
                if self.board[dest] & mask:
                    self.board[dest] ^= mask
                    self.players[player_id].mythics.remove(self.keeper)
                    self.players[player_id ^ 1].score += 1

        # check score for endgame
        # TODO: Refactor??
        if any(p.score > 9 for p in self.players):
            # end game
            return True, False
        return False, popped_dest

    def keeper_phase(self, player: Player) -> PlayDoneGenerator:
        """Move Keeper"""
        keeper_moves = self.dest_card.keeper_moves

        if self.dests:
                dest = self.dests[0]
        else:
            dest = (2,2)
        dest_costs = self.calc_dest_costs(dest)

            
        def inner(x:int, y:int, best_cost: int):
            """Inner function to reuse finicky code"""
            if keeper_moves >= 2 or not (self.board[x,y] & CLUTTER):
                cost = dest_costs[x,y]
                if cost < best_cost:
                    available.clear()
                    best_cost = cost
                if cost == best_cost:
                    available.append(board_to_action(x,y))  
            return best_cost  

        while keeper_moves:


            available = list[int]()
            best_cost = 255
            x,y = self.keeper
            if y < 4 and not (self.board[x,y] & HORZ_WALL):
                best_cost = inner(x,y+1, best_cost)
            if x < 4 and not (self.board[x,y] & VERT_WALL):
                best_cost = inner(x+1,y, best_cost)
            if y > 0 and not (self.board[x,y-1] & HORZ_WALL):
                best_cost = inner(x,y-1, best_cost) 
            if x>0 and not (self.board[x-1,y] & VERT_WALL):
                best_cost = inner(x-1,y, best_cost) 

            assert available
            if True or self.all_keeper_moves or len(available) > 1:
                action = yield PlayYield(
                    player.id_,
                    ActionPhase.MOVE_KEEPER,
                    keeper_moves,
                    ActionType.SELECT_DEST,
                    available
                )
            else:
                # If only one choice, no need to ask player
                action = available[0]

            x,y = action_to_board(action)
            done, popped_dest = self.move_keeper((x,y))
            if done:
                return (yield from self.end_game())
            
            if popped_dest:
                if self.dests:
                        dest = self.dests[0]
                else:
                    dest = (2,2)
                dest_costs = self.calc_dest_costs(dest)
            if self.board[x,y] & CLUTTER:
                keeper_moves -= 2
            else:
                keeper_moves -= 1

        # switch lunch or end game
        # TODO: Switch lunches
        # TODO: Proper end
        if not self.dests:
            return (yield from self.end_game())

        return False, 0

    def end_game(self) -> PlayDoneGenerator:
            yield PlayYield(
                0,
                ActionPhase.END_GAME,
                0,
                ActionType.PASS,
                [Action.PASS]
            )
            return True, 1

    def calc_dest_costs(self, dest: Coordinate):
        """Calculate costs for every cell to destination. Used for pathfinding"""
        # TODO: Review Algorithm
        
        dest_costs: np.ndarray[Coordinate, np.dtype[np.uint8]] = np.ndarray((5,5), np.uint8)
        dest_costs.fill(255)
        queue = list[tuple[Coordinate,int]]()

        def inner(x: int,y: int ,cost: int):
            if self.board[x,y] & CLUTTER:
                cost += 2
            else:
                cost += 1
            if dest_costs[x,y] > cost:
                dest_costs[x,y] = cost
                queue.append(((x,y), cost))

        inner(*dest, 0)

        while queue:
            node, cost = queue.pop(0)
            x,y = node
            if y < 4 and not (self.board[x,y] & HORZ_WALL):
                inner(x, y+1, cost)
            if x < 4 and not (self.board[x,y] & VERT_WALL):
                inner(x+1, y, cost)
            if y > 0 and not (self.board[x,y-1] & HORZ_WALL):
                inner(x, y-1, cost)
            if x>0 and not (self.board[x-1,y] & VERT_WALL):
                inner(x-1, y, cost)    
        return dest_costs

    def cleanup_phase(self, player: Player) -> PlayGenerator:
        """Cleanup and reset skills, spend tomes and boosts"""
        yield from self.spend_tomes(player)

        # boosts
        boosts = [0,0,0,0]
        for i in [2,1]:

            available = list[int]()
            if player.move < 9:
                available.append(Action.MOVE)
            if player.move_other < 3:
                available.append(Action.MOVE_OTHER)
            if player.move_shelf < 3:
                available.append(Action.MOVE_HORZ_SHELF)   
                available.append(Action.MOVE_VERT_SHELF)            
            if player.distract < 3:
                available.append(Action.DISTRACT)
            action = yield PlayYield(
                player.id_,
                ActionPhase.BOOST,
                i,
                ActionType.SELECT_SKILL,
                available
            )
            if action == Action.MOVE:
                boosts[0] += 1
                player.move = [3,4,5,6,7,8,9][player.move_tomes*2 + boosts[0]]
            elif action == Action.MOVE_OTHER:
                boosts[1] += 1
                player.move_other = [1,1,2,2,3][player.move_other_tomes + boosts[1]]
            elif action == Action.MOVE_HORZ_SHELF or action == Action.MOVE_VERT_SHELF:
                boosts[2] += 1
                player.move_shelf = [0,1,2,2,3][player.move_shelf_tomes + boosts[2]]
            elif action == Action.DISTRACT:
                boosts[3] += 1
                player.distract = [0,1,1,2,3][player.distract_tomes + boosts[3]]

    def render(self, mode: Optional[str]=None)  -> None | np.ndarray[tuple[int,int,int], Any] | pygame.Surface:
        """Render current state for human review"""
        render_mode = self.render_mode
        assert mode is None or mode == render_mode
        if render_mode in ("rgb_array", 'surface'):
            # for rgb_array, we still use pygame, then generate a ndarray
            screen = pygame.Surface(self.surface_size)
            screen.fill(BLACK)
            
            self.renderable.render(screen)

            if mode == 'rgb_array':
                return pygame.surfarray.pixels3d(screen) # type: ignore
            else:
                return screen
        return None

    def __repr__(self) -> str:
        return "LightZero MythicMischief v0 Env"
    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """Seed the random number gnerators"""
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

def board_to_action(x: int, y: int):
    """Convert board coordinate to an action int"""
    return y*5 + x+ len(Action)
def action_to_board(action: int):
    """Convert an action int to a board coordinate"""
    action -= len(Action)
    return (action % 5, action//5 )

class Renderable(abc.ABC):
    """An (Abstract) renderable GUI Element"""
    @abc.abstractmethod
    def render(self, surface: Surface):
        """Render GUI Element"""
        ...
    @abc.abstractmethod
    def size(self) -> Coordinate:
        """Calculate size of GUI Element"""
        ...

# Icons
# TODO: use sprites
PLAYER_ICON = "웃"
KEEPER_ICON = "T"
BOOK_ICON = "▯"
CLUTTER_ICON = "X"
RESPAWN_ICON = "↺"
PLAYER_SPECIAL_ICON = "*"


@dataclass
class BorderProps:
    color: tuple[int, int, int]
    width: int


BorderDef = tuple[Optional[BorderProps], Optional[BorderProps]]


class Grid(Renderable):
    """Grid of items"""

    def __init__(
        self,
        font: pygame.freetype.Font,
        size: tuple[int, int],
        cell_size: tuple[int, int],   # Cell size without padding
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

    def render_cell(self, x: int, y: int, surface: Surface, bg: tuple[int, int, int]):
        """Render a cell on given surface (without padding)"""
        pass

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
                y_pos = x_start + cell_width * y

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
    def size(self) -> Coordinate:
        sizes = [i.size() for i in self.items]
        if self.horz:
            height = max(s[1] for s in sizes)
            self.chunks = [s[0] for s in sizes]
            return (sum(self.chunks), height)
        else:
            width = max(s[0] for s in sizes)
            self.chunks = [s[1] for s in sizes]
            return (width, sum(self.chunks))
    def render(self, surface: Surface):
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
    def __init__(self, font: pygame.freetype.Font, max_size: int, color: tuple[int,int,int] =WHITE):
        self.font = font
        self.max_size = max_size  
        self.color = color
    def size(self) -> Coordinate:
        width = self.font.get_rect("X"* self.max_size).width
        return (width, self.font.get_sized_height(0))
    def render(self, surface: Surface):
        text = self.text()
        bg = self.bg()
        self.font.render_to(surface, (0,0), text, self.color, bg)
    def text(self) -> str:
        return ''
    def bg(self) -> tuple[int,int,int]:
        return BLACK
    
class StaticText(Text):
    """Static text"""
    def __init__(self, text: str, font: pygame.freetype.Font, color: tuple[int,int,int] =WHITE):
        self.static_text = text
        super().__init__(font, len(text), color)
    def text(self) -> str:
        return self.static_text

class Box(Renderable):
    """Create a box around an GUI item. Fills parent area"""
    def __init__(self, inside: Renderable, pad: int = 10, line_width: int=1, color:tuple[int,int,int]=WHITE):
        self.inside = inside
        self.pad = pad
        self.line_width = line_width
        self.color = color
    def render(self, surface: Surface):
        rect = pygame.Rect(0,0, *surface.get_size())


        rect.inflate_ip((-self.pad, -self.pad))
        pygame.draw.rect(surface, self.color, rect, width=self.line_width)
        rect.inflate_ip((-self.pad,-self.pad))
        self.inside.render(surface.subsurface(rect))

    def size(self) -> tuple[int, int]:
        inside_size = self.inside.size()
        offset = self.pad*2
        return (inside_size[0] + offset, inside_size[1] + offset )

       
class RenderableEnv(Renderable):
    """Wrap a Env to be usable with Renderable framewosk"""
    def __init__(self, env: BaseEnv):
        self.env = env
    def render(self, surface: Surface):
        rendered = cast(pygame.Surface, self.env.render())
        surface.blit(rendered, (0,0))

    def size(self) -> tuple[int, int]:
        rendered =  cast(pygame.Surface, self.env.render())
        return rendered.get_size()

def prompt(to_play: int, buffer: str, confirmed: bool, meta: dict[str, Any]):
    """Dynamic prompt"""
    # Own function to allow easy use of format strings
    player_name = f" <{meta['player_name']}>" if 'player_name' in meta else ''
    actions_left = f" ({meta['actions_left']})" if 'actions_left' in meta else ''
    action_phase = f"{meta['action_phase']}{actions_left}: " if 'action_phase' in meta else ''
    action_type = meta['action_type'] if 'action_type' in meta else 'Specify Action'
    

    return f"{action_phase}{action_type} for Player {to_play}{player_name}: {buffer}{'   (Enter to play)' if confirmed else ''}"

def play(env: PlayableEnv, seed: Optional[int] = None):
    """Play a game"""
    if seed is not None:
        env.seed(seed)
    env.reset()

    # Reversk the map so we can generate actions
    name_to_action = {k:v for v,k in enumerate(env.action_names)}

    running = True
    done = True  # Causes a first reset
    obs: Optional[dict[str,Any]] = None
    action: Optional[int] = None
    meta = dict[str, Any]

    

    # Pygame, yay!
    pygame.init()
    pygame.freetype.init()
    font = pygame.freetype.SysFont('unifont', 20)

    # Play GUI consists of the ENV and a dynbamic prompt
    max_prompt = max(len(prompt(1, action, True, {})) for action in env.action_names)
    curr_prompt = ""
    class ActionBar(Text):
        """Renderable Text Subcalss for dynamic prompt"""
        def text(self) -> str:
            return curr_prompt
 
    display = Flex(horz=False, items=[
        RenderableEnv(env),
        Box(ActionBar(font, max_prompt))
    ])

    # Initialize pygame window
    video_size = display.size()
    screen = pygame.display.set_mode(video_size)

    # timing utilities
    clock = pygame.time.Clock()
    pygame.time.set_timer(pygame.USEREVENT, 1000)

    to_play = 0
    confirmed = False
    buffer = ""

    # initial script to run
    # TODO: Refactor out
    script = []
    if True:
        script = [
            "2,0", # place Mythic 1
            "1,1", # place mythic 2
            "3,1", # place mythic 3
            "1",   # start tome in move
            "2,1", # place mythic 1
            "3,0", # place mythic 2
            "0,0", # place mythic 3   
        ]
        x = [
            "0",   # pass
            "2,1", # move keeper
            "2,0",  # move keeper
            "1,0",  # move keeper
            "1",   # boost move
            "1",   # boost move

        ]

    while running:
        curr_prompt = prompt(to_play, buffer, confirmed, meta)
        if done:
            # Restart
            done=False
            if seed is not None:
                env.seed(seed)
            obs = env.reset()
            assert obs
            to_play = cast(int, obs['to_play'])
            buffer=""
            confirmed=False
            action = None
        else:
            # Execute next step in script
            if script:
                action = name_to_action[script.pop(0)]

            # Execute pending actions
            if action is not None:
                prev_obs = obs
                obs, rew, done, meta = env.step(action)
                assert done or obs is not None
                to_play = cast(int, obs['to_play'])
                action = None

            # Render data
            if obs is not None:
                #rendered = env.render()
                #rendered = pygame.transform.scale(rendered, video_size)
                screen.fill(BLACK)
                display.render(screen)
            
            # Process all pending events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False                
                elif event.type == pygame.QUIT:
                    running = False

                #elif event.type == pygame.locals.VIDEORESIZE:
                #    video_size = event.size
                #    game = pygame.display.set_mode(video_size)

                # Handle Prompt Input
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    if confirmed:
                        action = name_to_action[buffer]
                        buffer = ""
                        confirmed = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                    if buffer:
                        buffer = buffer[:-1] 
                        if buffer in name_to_action:
                            confirmed = env.confirm_action(name_to_action[buffer])
                        else:
                            confirmed =  env.confirm_action(-1)
                elif event.type == pygame.KEYDOWN:
                    buffer = buffer + event.unicode
                    if buffer in name_to_action:
                        confirmed = env.confirm_action(name_to_action[buffer])
                    else:
                        confirmed = env.confirm_action(-1)

                # Timer event for blinking
                elif event.type == pygame.USEREVENT:
                    global g_blink
                    g_blink = not g_blink

            pygame.display.flip()
            clock.tick(10)
    pygame.quit()


if __name__ == '__main__':
    cfg = MythicMischiefEnv.default_config()
    cfg['render_mode']='surface'
    play(MythicMischiefEnv(cfg)) 
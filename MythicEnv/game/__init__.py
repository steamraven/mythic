"""
Game logic code
Subpackage, teams,  for team specific code

"""

import abc
import copy
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Generator, Generic, Optional, TypeVar
from MythicEnv import *
from MythicEnv.data import *

import numpy as np

# Bit flags for board
# TODO: Use Enum
DEST = [1 << 0, 1 << 1, 1 << 2]
DEST_MASK = 1 << 0 | 1 << 1 | 1 << 2
DEST_SHIFT = 0
KEEPER = 1 << 3
BOOKS = [1 << 4, 1 << 5]
BOOKS_MASK = 1 << 4 | 1 << 5
RESPAWN = 1 << 6
CLUTTER = 1 << 7
HORZ_WALL = 1 << 8
VERT_WALL = 1 << 9
WALLS = (VERT_WALL, HORZ_WALL)
PLAYER = [1 << 10, 1 << 11]
PLAYER_MASK = 1 << 10 | 1 << 11
PLAYER_SPECIAL = [1 << 12, 1 << 13]
PLAYER_SPECIAL_MASK = 1 << 12 | 1 << 13

DIAGONAL_WALL_MASKS = (
    (
        (0x1 | 0x8)  # upper left
        | ((0x2 | 0x4) << 4)  # lower right
        | ((0x2 | 0x8) << 8)  # horiz
        | ((0x1 | 0x4) << 12)  # vert
    ),
    (
        (0x1 | 0x2)  # upper right
        | ((0x4 | 0x8) << 4)  # lower left
        | ((0x2 | 0x8) << 8)  # horiz
        | ((0x1 | 0x4) << 12)  # vert
    ),
)


def board_to_action(x: int, y: int):
    """Convert board coordinate to an action int"""
    return y * 5 + x + len(Action)


def action_to_board(action: int):
    """Convert an action int to a board coordinate"""
    action -= len(Action)
    return (action % 5, action // 5)


@dataclass
class TeamData:
    """Team Specific Data"""

    name: str
    move_skill_costs: list[int]
    move_other_attr: str
    move_other_skill_costs: list[int]
    move_shelf_attr: str
    move_shelf_skill_costs: list[int]
    distract_skill_costs: list[int]
    legendary_attr: str
    after_lunch_attr: str


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
    """What type of action is available. What do the the available actions represent"""

    SELECT_SELF = 0
    SELECT_OPP = 1
    SELECT_MYTHIC = 2
    SELECT_DEST = 3
    SELECT_HORZ_WALL = 4
    SELECT_VERT_WALL = 5
    SELECT_SKILL = 6
    PASS = 7


class PlayYield(NamedTuple):
    """Intermediate yields of a Play coroutine"""

    to_play: int
    action_phase: ActionPhase
    actions_left: int
    action_type: ActionType
    available_actions: list[int]


class Player:
    """All the data relevent to a player"""

    def __init__(self, id_: int, team: "Team"):
        self.id_ = id_
        self.team = team
        self.mythics = set()
        self.move = team.data.move_skill_costs[0]
        self.move_tomes = 0
        self.move_other = team.data.move_other_skill_costs[0]
        self.move_other_tomes = 0
        # Move shelf boosted on first turn
        self.move_shelf = team.data.move_shelf_skill_costs[1]
        self.move_shelf_tomes = 0
        self.distract = team.data.distract_skill_costs[0]
        self.distract_tomes = 0
        self.legendary = 1
        self.after_lunch_special = 0
        self.tomes = 0
        self.score = 0
        self.occupying = None

    id_: int  # maps to index in game.players and PLAYERS and To_Play
    team: "Team"

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
    after_lunch_special: int

    tomes: int
    score: int

    occupying: Coordinate | None


# Type alias for Play sub-functions
# Play Coroutine that does not end game
PlayCoroutine = Generator[PlayYield, int, None]
# Play Coroutine that can end game
PlayOrDoneCoroutine = Generator[PlayYield, int, tuple[bool, int]]
PlaySkill = Callable[[Player], PlayOrDoneCoroutine]

T_Yield = TypeVar("T_Yield")
T_Return = TypeVar("T_Return")
T_Send = TypeVar("T_Send")

@dataclass
class Yield(Generic[T_Yield]):
    value: T_Yield

@dataclass
class Return(Generic[T_Return]):
    value: T_Return

@dataclass
class ClonableGenerator(abc.ABC, Generic[T_Yield, T_Send, T_Return]):
    step: int = 0

    @abc.abstractmethod
    def send(self, value: Optional[T_Send]) -> Yield[T_Yield] |  Return[T_Return] :
        ...
    @staticmethod
    def yield_(value: T_Yield):
        return Yield[T_Yield](value)
    @staticmethod
    def return_(value: T_Return):
        return Return[T_Return](value)

PlayGenerator = ClonableGenerator[PlayYield, int, None]
PlayOrDoneGenerator = ClonableGenerator[PlayYield, int, tuple[bool, int]]
PlaySkill_ = Callable[[Player], PlayOrDoneGenerator]


class Team(abc.ABC):
    """Team specific info"""

    data: TeamData
    game: "Optional[MythicMischiefGame]"

    def __init__(self, game: "Optional[MythicMischiefGame]"):
        self.game = game

    @abc.abstractmethod
    def play_move_other(self, player: Player) -> PlayOrDoneCoroutine: ...

    def play_move_horz_shelf(self, player: Player) -> PlayOrDoneCoroutine:
        return (yield from self.play_move_shelf(player, True))

    def play_move_vert_shelf(self, player: Player) -> PlayOrDoneCoroutine:
        return (yield from self.play_move_shelf(player, False))

    @abc.abstractmethod
    def play_move_shelf(self, player: Player, horz: bool) -> PlayOrDoneCoroutine: ...


class MythicMischiefGame:
    players: tuple[Player, Player]
    after_lunch: bool
    board: np.ndarray[tuple[int, int], np.dtype[np.uint16]]
    dest_card: DestCard
    start_layout: StartLayout
    dests: list[Coordinate]

    def __init__(self, player_teams: tuple[type[Team], type[Team]]):
        self.board = np.zeros((5, 5), np.uint16)
        self.after_lunch = False
        self.dest_card = BEFORE_LUNCH
        self.start_layout = START_LAYOUTS[0]
        self.dests = copy.deepcopy(self.dest_card.dests)
        for i, (x, y) in enumerate(self.dest_card.dests):
            self.board[x, y] |= DEST[i]
        for x, y in self.dest_card.books:
            self.board[x, y] |= BOOKS[0] | BOOKS[1]
        for x, y in self.dest_card.respawn:
            self.board[x, y] |= RESPAWN
        for x, y in self.start_layout.clutter:
            self.board[x, y] |= CLUTTER
        for x, y in self.start_layout.horz_walls:
            self.board[x, y] |= HORZ_WALL
        for x, y in self.start_layout.vert_walls:
            self.board[x, y] |= VERT_WALL

        x, y = self.keeper = (2, 2)
        self.board[x, y] |= KEEPER

        self.players = (
            Player(0, player_teams[0](self)),
            Player(1, player_teams[1](self)),
        )

        self.player_skills = [
            {
                Action.MOVE: self.play_move,
                Action.MOVE_OTHER: player.team.play_move_other,
                Action.MOVE_HORZ_SHELF: player.team.play_move_horz_shelf,
                Action.MOVE_VERT_SHELF: player.team.play_move_vert_shelf,
                Action.DISTRACT: self.play_distract,
            }
            for player in self.players
        ]

        # second player gets an extra tome to start
        self.players[1].tomes += 1

    def start_play(self) -> PlayOrDoneGenerator:
        """Play generator. Takes action, yield next available actions, and returns reward when done"""
        class PlayState(PlayOrDoneGenerator):
            gamestate: MythicMischiefGame
            def send(self, value: Optional[int]):
                gamestate = self.gamestate
                action = value
                # players = self.players
                players = gamestate.players

                if self.step == 0:
                    # init
                    assert action is None
                    self.step += 1

                if self.step == 1:
                    # Setup
                    # init: yield from self.place_mythics(players[0], True)
                    self.place_mythics = gamestate.place_mythics(players[0], True)
                    action = None
                    self.step += 1
                if self.step == 2:
                    # run: yield from self.place_mythics(players[0], True)
                    assert self.place_mythics
                    try:
                        return self.yield_(self.place_mythics.send(action))
                    except StopIteration:
                        pass
                    # Player 2 gets a tome, but does not affect start boost
                    # start_boost = players[1].move_shelf
                    self.start_boost = players[1].move_shelf
                    # init: yield from self.spend_tomes(players[1])
                    self.spend_tomes = gamestate.spend_tomes(players[1])
                    action = None
                    self.step += 1                    
                if self.step == 3:
                    # run: yield from self.spend_tomes(players[1])
                    assert self.spend_tomes
                    try:
                        self.yield_(self.spend_tomes.send(action))
                    except StopIteration:
                        pass
                    # players[1].move_shelf = start_boost
                    players[1].move_shelf = self.start_boost

                    # init: yield from self.place_mythics(players[1], True)
                    self.place_mythics = gamestate.place_mythics(players[1], True)
                    action = None
                    self.step += 1
                if self.step == 4:
                    # run: yield from self.place_mythics(players[1], True)
                    assert self.place_mythics
                    try:
                        return self.yield_(self.place_mythics.send(action))
                    except StopIteration:
                        pass
                    self.step += 1
                # Main loop 
                while True:
                    assert 5 <= self.step <= 9
                    # for player in players:
                    if self.step == 5:
                        # Use index instead of an iterator
                        self.player = 0
                        self.step += 1

                    while True:  # for loop
                        assert 6 <= self.step <= 9
                        if self.step == 6:          
                            # check for loop
                            if self.player == len(players):
                                self.step = 5 # End for loop
                                break                   
                            player = players[self.player]
                            # init: done, reward = yield from self.mythic_phase(player)
                            self.mythic_phase = gamestate.mythic_phase(player)
                            action = None
                            self.step += 1
        
                        if self.step == 7:
                            # run: done, reward = yield from self.mythic_phase(player)
                            assert self.mythic_phase
                            done: bool
                            reward: int
                            try: 
                                return self.yield_(self.mythic_phase.send(action))
                            except StopIteration as e:
                                done, reward = e.value
                            if done:
                                #return done, reward
                                return self.return_((done,reward))
                            
                            player = players[self.player]
                            assert not player.occupying
                            #init: done, reward = yield from self.keeper_phase(player)
                            self.keeper_phase = gamestate.keeper_phase(player)
                            action = None
                            self.step += 1
                        if  self.step == 8:
                            # run: done, reward = yield from self.keeper_phase(player)
                            assert self.keeper_phase
                            done: bool
                            reward: int
                            try:
                                return self.yield_(self.keeper_phase.send(action))
                            except StopIteration as e:
                                done, reward = e.value
                            if done:
                                #return done, reward
                                return self.return_((done, reward))
                            player = players[self.player]
                            # init: yield from self.cleanup_phase(player)
                            self.cleanup_phase = gamestate.cleanup_phase(player)
                            self.step += 1
                            action = None
                        if self.step == 9:
                            # run: yield from self.cleanup_phase(player)
                            assert self.cleanup_phase
                            try:
                                return self.yield_(self.cleanup_phase.send(action))
                            except StopIteration :
                                pass
                            self.player += 1
                            self.step = 6 #restart for loop
                            continue
                
        state = PlayState()
        # Deepcopy will only copy once
        state.gamestate = self
        return state


    def place_mythics(self, player: Player, anywhere: bool) -> PlayCoroutine:
        """Place available mythics in spot in an available spot"""
        any_mask = PLAYER_MASK | KEEPER | BOOKS_MASK
        respawn_mask = PLAYER_MASK | KEEPER

        available = []  # to make typechecker happy

        while len(player.mythics) < 3:
            if not anywhere:
                available = [
                    respawn
                    for respawn in self.dest_card.respawn
                    if not (self.board[respawn] & respawn_mask)
                ]
                if len(available) == 0:
                    # if we can't place now, we wont be able to anytime this call. Start searching anywhere
                    anywhere = True
            if anywhere:
                available = [
                    (x, y)
                    for x in range(5)
                    for y in range(5)
                    if not (self.board[x, y] & any_mask)
                ]

            action = yield PlayYield(
                player.id_,
                ActionPhase.PLACE_MYTHIC,
                3 - len(player.mythics),
                ActionType.SELECT_DEST,
                [board_to_action(x, y) for x, y in available],
            )
            pos = action_to_board(action)
            self.board[pos] |= PLAYER[player.id_]
            player.mythics.add(pos)

    def spend_tomes(self, player: Player) -> PlayCoroutine:
        """Spend all collected tomes on skills"""

        # Online game forces spend

        self.reset_skills(player)  # This resets starting boost. See start_play
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
                available,
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

    def mythic_phase(self, player: Player) -> PlayOrDoneCoroutine:
        """Phase where actions are performed by the mythics"""
        yield from self.place_mythics(player, False)

        # TODO: all skills/abilities
        while True:

            available: list[int] = [Action.PASS]

            if player.occupying:
                # Cannot stop movement on another players space.  Or activate other actins
                yield from self.play_move(player)
                continue

            skills = self.player_skills[player.id_]
            skill_coroutines: dict[int, PlayOrDoneCoroutine] = {
                a: skill(player) for a, skill in skills.items()
            }

            for a, r in skill_coroutines.items():
                y = next(r)
                assert y.to_play == player.id_
                assert len(y.available_actions) == 0 or y.available_actions == [a]
                available.extend(y.available_actions)

            action = yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, available
            )

            if action == Action.PASS:
                return False, 0

            # Because co-routine was alrady started, 
            # this will send a None back to the first yield of the selected routine
            # Instead of the selected action
            done, reward = yield from skill_coroutines[action]
            if done:
                return done, reward

    # Does not really return Done, but needs PlayOrDoneCoroutine for parent code
    def play_move(self, player: Player) -> PlayOrDoneCoroutine:
        """Play a move own mythic skill"""

        player_mask = PLAYER[player.id_]
        other_player_mask = PLAYER[player.id_ ^ 1]
        not_available_mask = KEEPER | player_mask

        def find_available_moves(
            pos: Coordinate, remaining: int
        ) -> list[tuple[Coordinate, int]]:
            available = list[tuple[Coordinate, int]]()

            # def inner(x: int, y: int, _: int):
            #     """Add space to available if reachable.  May recurse for occupied spots"""
            #     if self.board[x, y] & not_available_mask:
            #         return 0
            #     cost = 2 if self.board[x, y] & CLUTTER else 1

            #     # If the destination has another player, make sure there is an exit
            #     if self.board[x, y] & other_player_mask:
            #         if remaining > cost: 
            #             next_moves = find_available_moves((x, y), remaining - cost)
            #             if next_moves:
            #                 cost += min(a[1] for a in next_moves)
            #                 available.append(((x, y), cost))
            #         return 0

            #     if remaining >= cost:
            #         available.append(((x, y), cost))
            #     return 0

            for n in self.get_neighbors(pos):
                if self.board[n] & not_available_mask:
                    continue
                cost = 2 if self.board[pos] & CLUTTER else 1
                if self.board[n] & other_player_mask:
                    if remaining > cost: 
                        next_moves = find_available_moves(n, remaining - cost)
                        if next_moves:
                            cost += min(a[1] for a in next_moves)
                            available.append((n, cost))
                    continue
                
                if remaining >= cost:
                    available.append((n, cost))


            #self.check_neighbors(pos, inner, 0)
            return available

        available_moves = dict[Coordinate, list[tuple[Coordinate, int]]]()

        # If player has a mythic occupying another mythics stop, only that mythic can move
        if player.occupying:
            mythics = [player.occupying]
            if not (player.move):
                assert player.move
        else:
            mythics = player.mythics

        if player.move:
            for mythic in mythics:
                self.board[mythic] ^= player_mask
                available_moves[mythic] = find_available_moves(mythic, player.move)
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

        # This yield shoud retorn Action.Move,
        # but because of how this co-routine is called, 
        # the first yield may return either None, or Action.MOVE,
        # Either way, we don't care
        if any(v for v in available_moves.values()):
            yield PlayYield(
                player.id_,
                ActionPhase.USE_SKILL,
                1,
                ActionType.SELECT_SKILL,
                [Action.MOVE],
            )
        else:
            yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, []
            )

        # Select mythic to move from mythics that can move
        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE,
            2,
            ActionType.SELECT_SELF,
            [board_to_action(*k) for k, v in available_moves.items() if v],
        )

        mythic = action_to_board(action)
        available = available_moves[mythic]

        assert available

        # Choose destination from available move destinations
        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE,
            1,
            ActionType.SELECT_DEST,
            [board_to_action(*a[0]) for a in available],
        )

        # Move mythic
        dest = action_to_board(action)
        self.move_mythic(player, mythic, dest)

        if self.board[dest] & CLUTTER:
            cost = 2
        else:
            cost = 1

        player.move -= cost

        # Check for mythic occupying space of another
        player.occupying = dest if self.board[dest] & other_player_mask else None
        if player.occupying:
            if not (player.move):
                assert player.move

        return False, 0

    def play_distract(self, player: Player) -> PlayOrDoneCoroutine:
        """Play a distract skill"""
        available_moves = dict[Coordinate, list[int]]()
        # TODO: don't need to select mythic
        if player.distract:
            for mythic in player.mythics:
                dest_costs = self.calc_dest_costs(mythic)
                available = list[int]()
                best_cost = 255

                # Distract takes into account clutter for pathfinding (see calc_dest_costs)
                # But clutter not counted for the move. 
                # i.e. can be distrated onto clutter
                for n in self.get_neighbors(self.keeper):
                    n_cost = dest_costs[n]
                    if n_cost < best_cost:
                        available.clear()
                        best_cost = n_cost
                    if n_cost == best_cost:
                        available.append(board_to_action(*n))

                available_moves[mythic] = available

        # This yield shoud retorn Action.DISTRACT,
        # but because of how this co-routine is called, 
        # the first yield will return None,
        if any(v for v in available_moves.values()):
            yield PlayYield(
                player.id_,
                ActionPhase.USE_SKILL,
                1,
                ActionType.SELECT_SKILL,
                [Action.DISTRACT],
            )
        else:
            yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, []
            )

        action = yield PlayYield(
            player.id_,
            ActionPhase.DISTRACT,
            1,
            ActionType.SELECT_DEST,
            [board_to_action(*k) for k, v in available_moves.items() if v],
        )
        player.distract -= 1
        dest = action_to_board(action)
        done = self.move_keeper(dest)
        if done:
            return (yield from self.end_game(player, 1))

        return False, 0

    def keeper_phase(self, player: Player) -> PlayOrDoneCoroutine:
        """Move Keeper"""
        keeper_moves = self.dest_card.keeper_moves

        if self.dests:
            dest = self.dests[0]
        else:
            dest = (2, 2)
        dest_costs = self.calc_dest_costs(dest)

        while keeper_moves:

            available = list[int]()
            best_cost = 255

            for n in self.get_neighbors(self.keeper):
                if keeper_moves >= 2 or not (self.board[n] & CLUTTER):
                    # Can't move onto clutter if not enough moves
                    n_cost = dest_costs[n]
                    if n_cost < best_cost:
                        available.clear()
                        best_cost = n_cost
                    if n_cost == best_cost:
                        available.append(board_to_action(*n))

            assert available
            if True or self.all_keeper_moves or len(available) > 1:
                action = yield PlayYield(
                    player.id_,
                    ActionPhase.MOVE_KEEPER,
                    keeper_moves,
                    ActionType.SELECT_DEST,
                    available,
                )
            else:
                # If only one choice, no need to ask player
                action = available[0]

            x, y = action_to_board(action)
            done = self.move_keeper((x, y))
            if done:
                return (yield from self.end_game(player, 1))

            if self.board[x, y] & CLUTTER:
                keeper_moves -= 2
            else:
                keeper_moves -= 1

        # switch lunch or end game
        # TODO: Switch lunches
        # TODO: Proper end
        if not self.dests:
            if self.players[0].score > self.players[1].score:
                return (yield from self.end_game(self.players[0], 1))
            if self.players[1].score > self.players[0].score:
                return (yield from self.end_game(self.players[1], 1))
            return (yield from self.end_game(player, 0))

            

        return False, 0

    def end_game(self, player: Player, reward: int) -> PlayOrDoneCoroutine:
        #yield PlayYield(
        #    player.id_, ActionPhase.END_GAME, 1, ActionType.PASS, [Action.PASS]
        #)
        return True, reward
        yield

    def cleanup_phase(self, player: Player) -> PlayCoroutine:
        """Cleanup and reset skills, spend tomes and boosts"""
        yield from self.spend_tomes(player)

        # boosts
        boosts = [0, 0, 0, 0]
        for i in [2, 1]:

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
                player.id_, ActionPhase.BOOST, i, ActionType.SELECT_SKILL, available
            )
            if action == Action.MOVE:
                boosts[0] += 1
                player.move = player.team.data.move_skill_costs[
                    player.move_tomes * 2 + boosts[0]
                ]
            elif action == Action.MOVE_OTHER:
                boosts[1] += 1
                player.move_other = player.team.data.move_other_skill_costs[
                    player.move_other_tomes + boosts[1]
                ]
            elif action == Action.MOVE_HORZ_SHELF or action == Action.MOVE_VERT_SHELF:
                boosts[2] += 1
                player.move_shelf = player.team.data.move_shelf_skill_costs[
                    player.move_shelf_tomes + boosts[2]
                ]
            elif action == Action.DISTRACT:
                boosts[3] += 1
                player.distract = player.team.data.distract_skill_costs[
                    player.distract_tomes + boosts[3]
                ]

    def reset_skills(self, player: Player):
        """Reset all skills based on spent tomes"""
        player.move = player.team.data.move_skill_costs[player.move_tomes * 2]
        player.move_other = player.team.data.move_other_skill_costs[
            player.move_other_tomes
        ]
        player.move_shelf = player.team.data.move_shelf_skill_costs[
            player.move_shelf_tomes
        ]
        player.distract = player.team.data.distract_skill_costs[player.distract_tomes]

    def move_mythic(self, player: Player, old: Coordinate, new: Coordinate) -> None:
        """Actually move the mythic and do record keeping"""
        player_mask = PLAYER[player.id_]

        if not (0<=new[0]<=4 and 0<=new[1]<=4):
            assert (0<=new[0]<=4 and 0<=new[1]<=4)
        if not (self.board[old] & player_mask and old in player.mythics):
            assert self.board[old] & player_mask and old in player.mythics

        self.board[old] ^= player_mask
        player.mythics.remove(old)
        player.mythics.add(new)
        self.board[new] |= player_mask

        # check for book
        if self.board[new] & BOOKS[player.id_]:
            self.board[new] ^= BOOKS[player.id_]
            player.tomes += 1

    def move_keeper(self, dest: Coordinate) -> bool:
        """Actually move the keeper, adjust scores and check for endgame"""
        # Move Keeper

        if not (0<=dest[0]<=4 and 0<=dest[1]<=4):
            assert (0<=dest[0]<=4 and 0<=dest[1]<=4)

        if not (self.board[self.keeper] & KEEPER):
            assert self.board[self.keeper] & KEEPER

        self.board[self.keeper] ^= KEEPER

        self.board[dest] |= KEEPER
        self.keeper = dest

        # check if keeper reached dest
        if self.dests and self.keeper == self.dests[0]:
            self.board[self.dests[0]] ^= DEST[3 - len(self.dests)]
            _ = self.dests.pop(0)

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
            return True
        return False

    def move_wall(self, src: Wall, dest: Wall):
        if dest.wall_type == HORZ_WALL:
            if not (0<=dest.pos[0]<=4 and 0<=dest.pos[1]<=3):
                assert (0<=dest.pos[0]<=4 and 0<=dest.pos[1]<=3)
        else:
            if not (0<=dest.pos[0]<=3 and 0<=dest.pos[1]<=4):
                assert (0<=dest.pos[0]<=3 and 0<=dest.pos[1]<=4)

        self.board[src.pos] ^= src.wall_type
        self.board[dest.pos] |= dest.wall_type

    # def check_neighbors(
    #     self, pos: Coordinate, inner: Callable[[int, int, int], int], best_cost: int
    # ) -> int:
    #     """Call inner for each neighber, keeping track of a best_cost"""
    #     x, y = pos
    #     if y < 4 and not (self.board[x, y] & HORZ_WALL):
    #         best_cost = inner(x, y + 1, best_cost)
    #     if x < 4 and not (self.board[x, y] & VERT_WALL):
    #         best_cost = inner(x + 1, y, best_cost)
    #     if y > 0 and not (self.board[x, y - 1] & HORZ_WALL):
    #         best_cost = inner(x, y - 1, best_cost)
    #     if x > 0 and not (self.board[x - 1, y] & VERT_WALL):
    #         best_cost = inner(x - 1, y, best_cost)
    #     return best_cost

    def get_neighbors(self, pos: Coordinate) -> Generator[Coordinate]:
        x, y = pos
        if y < 4 and not (self.board[x, y] & HORZ_WALL):
            yield x, y+1
        if x < 4 and not (self.board[x, y] & VERT_WALL):
            yield x + 1, y
        if y > 0 and not (self.board[x, y - 1] & HORZ_WALL):
           yield x, y - 1
        if x > 0 and not (self.board[x - 1, y] & VERT_WALL):
            yield x - 1, y

    def calc_dest_costs(
        self, dest: Coordinate
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Calculate costs for every cell to destination. Used for pathfinding"""
        # TODO: Review Algorithm

        dest_costs = np.full((5, 5), 255, np.uint8)
        cost = 2 if self.board[dest] & CLUTTER else 1
        queue = [(dest, cost)]
        dest_costs[dest] = cost

        #def inner(x: int, y: int, base_cost: int):
        #    "add to dest_costs and queue if better cost"
        #    if self.board[x, y] & CLUTTER:
        #        cost = base_cost + 2
        #    else:
        #        cost = base_cost + 1
        #    if cost < dest_costs[x, y]:
        #        dest_costs[x, y] = cost
        #        queue.append(((x, y), cost))
        #    return base_cost

        #inner(*dest, 0)

        while queue:
            node, cost = queue.pop(0)
            for n in self.get_neighbors(node):
                
                if self.board[n] & CLUTTER:
                    n_cost = cost + 2
                else:
                    n_cost = cost + 1
                if n_cost < dest_costs[n]:
                    dest_costs[n] = n_cost
                    queue.append((n, n_cost))
            #self.check_neighbors(node, inner, cost)
        return dest_costs

    def line_of_sight(
        self,
        mythic: Coordinate,
        opp: Coordinate,
        min_distance: int = 1,
    ) -> Optional[Coordinate]:
        mask = KEEPER | PLAYER_MASK
        # Horizontal and vertical
        for coord in [0, 1]:
            other_coord = coord ^ 1
            if mythic[other_coord] == opp[other_coord]:
                if min_distance <= abs(mythic[coord] - opp[coord]):
                    direction = 1 if mythic[coord] < opp[coord] else -1
                    for i in range(
                        mythic[coord] + direction, opp[coord] + direction, direction
                    ):
                        pos = (
                            i if coord == 0 else mythic[0],
                            i if coord == 1 else mythic[1],
                        )
                        wall_pos = pos
                        if direction == 1:
                            wall_pos = (
                                i - 1 if coord == 0 else mythic[0],
                                i - 1 if coord == 1 else mythic[1],
                            )

                        if self.board[wall_pos] & WALLS[coord]:
                            return None
                        elif i != opp[coord] and self.board[pos] & mask:
                            return None
                    else:  # nothing found, has line of sight
                        if coord == 0:
                            return (direction, 0)
                        else:
                            return (0, direction)
                return None

        # diagonal
        x_direction = 1 if mythic[0] < opp[0] else -1
        y_direction = 1 if mythic[1] < opp[1] else -1
        if mythic[0] - opp[0] == x_direction * y_direction * (mythic[1] - opp[1]):
            if min_distance <= abs(mythic[0] - opp[0]):
                for pos in zip(
                    range(mythic[0] + x_direction, opp[0] + x_direction, x_direction),
                    range(mythic[1] + y_direction, opp[1] + y_direction, y_direction),
                ):
                    wall_pos = (
                        pos[0] - 1 if x_direction == 1 else pos[0],
                        pos[1] - 1 if y_direction == 1 else pos[1],
                    )

                    walls = self.cross_wall_bitset(wall_pos)
                    i = 0 if x_direction * y_direction == 1 else 1
                    if self.check_wall_bitset(walls, DIAGONAL_WALL_MASKS[i]):
                        break
                    elif pos[0] != opp[0] and self.board[pos] & mask:
                        break
                else:
                    return (x_direction, y_direction)
            return None

    def cross_wall_bitset(self, pos: Coordinate) -> int:
        """Create a bitset of walls at the bottom right intersection of position"""
        acc = 0
        if not (0<=pos[0]<=3 and 0<=pos[1]<=3):
            assert 0<=pos[0]<=3 and 0<=pos[1]<=3
        x, y = pos
        value: int = self.board[pos]
        acc |= 0x1 if value & VERT_WALL else 0  # Up
        acc |= 0x2 if self.board[x + 1, y] & HORZ_WALL else 0  # right
        acc |= 0x4 if self.board[x, y + 1] & VERT_WALL else 0  # down
        acc |= 0x8 if value & HORZ_WALL else 0  # left
        return acc

    @staticmethod
    def check_wall_bitset(
        wall_bitset: int, test_bitset: int, count: int = 4, shift: int = 4
    ) -> bool:
        """Check a wall mask against a set of masks"""
        mask = (1 << shift) - 1
        return any(
            wall_bitset == ((test_bitset >> i) & mask)
            for i in range(0, count * shift, shift)
        )

    # def left_wall(self, x: int, y: int) -> bool:
    #     return self.board[x - 1, y] & VERT_WALL

    # def right_wall(self, x: int, y: int) -> bool:
    #     return self.board[x, y] & VERT_WALL

    # def top_wall(self, x: int, y: int) -> bool:
    #     return self.board[x, y - 1] & HORZ_WALL

    # def bottom_wall(self, x: int, y: int) -> bool:
    #     return self.board[x, y] & HORZ_WALL

    def test_wall_moves(
        self,
        src_wall: Wall,
        available: dict[Wall, set[Wall]],
        dest_walls: list[Wall],
    ) -> None:
        """Add legal walls moves to available"""
        max_src_vert = 4 if src_wall.wall_type == VERT_WALL else 5
        max_src_horz = 4 if src_wall.wall_type == HORZ_WALL else 5
        if (
            0 <= src_wall.pos[0] < max_src_vert
            and 0 <= src_wall.pos[1] < max_src_horz
            and self.board[src_wall.pos] & src_wall.wall_type
        ):
            # Test Move: first step, remove current location (add back at end)
            self.board[src_wall.pos] ^= src_wall.wall_type

            for dest_wall in dest_walls:
                # Check dest
                max_dest_vert = 4 if dest_wall.wall_type == VERT_WALL else 5
                max_dest_horz = 4 if dest_wall.wall_type == HORZ_WALL else 5
                if (
                    0 <= dest_wall.pos[0] < max_dest_vert
                    and 0 <= dest_wall.pos[1] < max_dest_horz
                    and not self.board[dest_wall.pos] & dest_wall.wall_type
                ):
                    self.board[dest_wall.pos] |= dest_wall.wall_type
                    if self.check_wall_invariant():
                        available[src_wall].add(dest_wall)
                    self.board[dest_wall.pos] ^= dest_wall.wall_type

            # Add back current location as if nothing happened
            self.board[src_wall.pos] |= src_wall.wall_type

    def check_wall_invariant(self) -> bool:
        """Check to make sure all spaces are available"""
        # Breadth first search from top left.  Should hit every cell
        #
        seen = np.zeros((5, 5), np.bool_)
        queue = [(0, 0)]

        # def inner(x: int, y: int, _: int) -> int:
        #     queue.append((x, y))
        #     return 0

        while queue:
            pos = queue.pop(0)
            if not seen[pos]:
                seen[pos] = True
                #self.check_neighbors(pos, inner, 0)
                queue.extend(self.get_neighbors(pos))

        x = seen.all()
        return bool(x)

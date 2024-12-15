"""
Team specific data and logic for Frankentein's Monsters
"""

from MythicEnv.game import (
    Team,
    TeamData,
    Player,
    PlayYield,
    ActionPhase,
    Action,
    ActionType,
    KEEPER,
    PLAYER_MASK,
    HORZ_WALL,
    VERT_WALL,
    board_to_action,
    action_to_board,
    MythicMischiefGame,
)
from MythicEnv.game import (
    PlayOrDoneGenerator,
    PlayOrDoneGeneratorImpl,
    PlayOrDoneGeneratorImpl_Return,
    Yield,
    Return,
)
from MythicEnv import *
from collections import defaultdict

MonsterData = TeamData(
    name="Monster",
    move_skill_costs=[3, 4, 5, 6, 7, 8, 9],
    move_other_attr="Throw",
    move_other_skill_costs=[1, 1, 2, 2, 3],
    move_shelf_attr="Barge",
    move_shelf_skill_costs=[0, 1, 2, 2, 3],
    distract_skill_costs=[0, 1, 1, 2, 3],
    legendary_attr="Intemidate",
    after_lunch_attr="Trap",
)


class Monster(Team):
    data = MonsterData

    def play_move_other(
        self, game: MythicMischiefGame, player: Player
    ) -> PlayOrDoneGenerator:
        # Monster: Throw
        # Move an ally or enemy from a
        # neighboring space to the exact
        # opposite neighboring space.
        # • You can Throw onto a Cluttered space.
        # • You cannot Throw over Bookshelves.

        class PlayMoveOtherState(PlayOrDoneGeneratorImpl):

            def send_impl(self, value: int | None) -> PlayOrDoneGeneratorImpl_Return:
                action = value
                # TODO: don't need to select mythic, just opponent and dest

                other_player = game.players[player.id_ ^ 1]
                mask = KEEPER | PLAYER_MASK

                if self.next_step():
                    self.available_moves = dict[
                        Coordinate, dict[Coordinate, Coordinate]
                    ]()
                    if player.move_other:
                        for mythic in player.mythics:
                            available = self.available_moves[mythic] = dict[
                                Coordinate, Coordinate
                            ]()
                            for opp in other_player.mythics:
                                assert mythic != opp
                                if (
                                    abs(mythic[0] - opp[0]) <= 1
                                    and abs(mythic[1] - opp[1]) <= 1
                                ):
                                    direction = game.line_of_sight(mythic, opp)
                                    if direction:
                                        over = (-1 * direction[0], -1 * direction[1])
                                        if (
                                            0 <= over[0] <= 4
                                            and 0 <= over[1] <= 4
                                            and not game.board[over] & mask
                                            and game.line_of_sight(mythic, over)
                                        ):
                                            available[opp] = over

                    if any(v for v in self.available_moves.values()):
                        return Yield(
                            PlayYield(
                                player.id_,
                                ActionPhase.USE_SKILL,
                                1,
                                ActionType.SELECT_SKILL,
                                [Action.MOVE_OTHER],
                            )
                        )
                    else:
                        return Yield(
                            PlayYield(
                                player.id_,
                                ActionPhase.USE_SKILL,
                                1,
                                ActionType.SELECT_SKILL,
                                [],
                            )
                        )
                if self.next_step():
                    # The previous yield should return Action.MOVE_OTHER,
                    # but because of how this co-routine is called,
                    # the previous yield will return None,
                    assert action is None or action == Action.MOVE_OTHER
                    return Yield(
                        PlayYield(
                            player.id_,
                            ActionPhase.MOVE_OPP,
                            2,
                            ActionType.SELECT_SELF,
                            [
                                board_to_action(*k)
                                for k, v in self.available_moves.items()
                                if v
                            ],
                        )
                    )
                if self.next_step():
                    assert action is not None

                    self.mythic = action_to_board(action)

                    return Yield(
                        PlayYield(
                            player.id_,
                            ActionPhase.MOVE_OPP,
                            1,
                            ActionType.SELECT_OPP,
                            [
                                board_to_action(*k)
                                for k in self.available_moves[self.mythic].keys()
                            ],
                        )
                    )
                if self.next_step():
                    assert action is not None
                    player.move_other -= 1
                    opp = action_to_board(action)
                    # move opponent mythic
                    dest = self.available_moves[self.mythic][opp]
                    game.move_mythic(other_player, opp, dest)

                return Return((False, 0))

        return PlayMoveOtherState()

    def play_move_shelf(
        self, game: MythicMischiefGame, player: Player, horz: bool
    ) -> PlayOrDoneGenerator:
        # Monstor: Barge
        # Rotate a Bookshelf into
        # the opposing adjacent space.
        # • You cannot block off a section of the
        # Library completely.
        # • You cannot place a Bookshelf onto the outer
        # edge of the Library or another Bookshelf.

        class PlayMoveShelfState(PlayOrDoneGeneratorImpl):
            def send_impl(self, value: int | None) -> PlayOrDoneGeneratorImpl_Return:
                action = value
                wall_type = HORZ_WALL if horz else VERT_WALL
                o_wall_type = VERT_WALL if horz else HORZ_WALL

                if self.next_step():
                    self.available_moves: dict[Wall, set[Wall]] = defaultdict(set)

                    if player.move_shelf:
                        for mythic in player.mythics:
                            left = (mythic[0] - 1, mythic[1])
                            up = (mythic[0], mythic[1] - 1)

                            # don't need to test src for >=4 since it has no walls
                            game.test_wall_moves(
                                Wall(wall_type, mythic),
                                self.available_moves,
                                # Horz
                                (
                                    [
                                        Wall(o_wall_type, (mythic[0], mythic[1] + 1)),
                                        Wall(
                                            o_wall_type, (mythic[0] - 1, mythic[1] + 1)
                                        ),
                                    ]
                                    if horz
                                    else
                                    # vert
                                    [
                                        Wall(o_wall_type, (mythic[0] + 1, mythic[1])),
                                        Wall(
                                            o_wall_type, (mythic[0] + 1, mythic[1] - 1)
                                        ),
                                    ]
                                ),
                            )
                            pos = up if horz else left
                            if pos[0] >= 0 and pos[1] >= 0:
                                game.test_wall_moves(
                                    Wall(wall_type, pos),
                                    self.available_moves,
                                    [
                                        Wall(o_wall_type, pos),
                                        Wall(
                                            o_wall_type, (mythic[0] - 1, mythic[1] - 1)
                                        ),
                                    ],
                                )

                    if any(
                        src.wall_type == wall_type
                        for src in self.available_moves.keys()
                    ):
                        return Yield(
                            PlayYield(
                                player.id_,
                                ActionPhase.USE_SKILL,
                                1,
                                ActionType.SELECT_SKILL,
                                [
                                    (
                                        Action.MOVE_HORZ_SHELF
                                        if horz
                                        else Action.MOVE_VERT_SHELF
                                    )
                                ],
                            )
                        )
                    else:
                        return Yield(
                            PlayYield(
                                player.id_,
                                ActionPhase.USE_SKILL,
                                1,
                                ActionType.SELECT_SKILL,
                                [],
                            )
                        )
                if self.next_step():
                    # This previous ;oiud should return Action.MOVE_SHELF,
                    # but because of how this co-routine is called,
                    # the previous yield will return None,
                    assert action is None or action == (
                        Action.MOVE_HORZ_SHELF if horz else Action.MOVE_VERT_SHELF
                    )
                    return Yield(
                        PlayYield(
                            player.id_,
                            (
                                ActionPhase.MOVE_HORZ_SHELF
                                if horz
                                else ActionPhase.MOVE_VERT_SHELF
                            ),
                            2,
                            (
                                ActionType.SELECT_HORZ_WALL
                                if horz
                                else ActionType.SELECT_VERT_WALL
                            ),
                            [
                                board_to_action(*src.pos)
                                for src in self.available_moves.keys()
                            ],
                        )
                    )
                if self.next_step():
                    assert action is not None
                    self.src = Wall(wall_type, action_to_board(action))

                    return Yield(
                        PlayYield(
                            player.id_,
                            (
                                ActionPhase.MOVE_HORZ_SHELF
                                if horz
                                else ActionPhase.MOVE_VERT_SHELF
                            ),
                            1,
                            (
                                ActionType.SELECT_VERT_WALL
                                if horz
                                else ActionType.SELECT_HORZ_WALL
                            ),
                            [
                                board_to_action(*dest.pos)
                                for dest in self.available_moves[self.src]
                            ],
                        )
                    )
                if self.next_step():
                    assert action is not None
                    player.move_shelf -= 1
                    dest = Wall(o_wall_type, action_to_board(action))

                    game.move_wall(self.src, dest)

                return Return((False, 0))

        return PlayMoveShelfState()

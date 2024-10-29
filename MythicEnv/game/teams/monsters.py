from MythicEnv.game import (
    Team,
    TeamData,
    PlayOrDoneCoroutine,
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

)
from MythicEnv import *
from collections import defaultdict

MonsterData = TeamData(
    id_=1,
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

    def play_move_other(self, player: Player) -> PlayOrDoneCoroutine:
        # Monster: Throw
        # Move an ally or enemy from a
        # neighboring space to the exact
        # opposite neighboring space.
        # • You can Throw onto a Cluttered space.
        # • You cannot Throw over Bookshelves.

        # TODO: don't need to select mythic, just opponent and dest

        game = self.game
        assert game
        other_player = game.players[player.id_ ^ 1]
        mask = KEEPER | PLAYER_MASK
        available_moves = dict[Coordinate, dict[Coordinate, Coordinate]]()
        if player.move_other:
            for mythic in player.mythics:
                available = available_moves[mythic] = dict[Coordinate, Coordinate]()
                for opp in other_player.mythics:
                    if abs(mythic[0] - opp[0]) <= 1 and abs(mythic[1] - opp[1]) <= 1:
                        direction = game.line_of_sight(mythic, opp)
                        if direction:
                            over = (-1 * direction[0], -1 * direction[1])
                            if not game.board[over] & mask and game.line_of_sight(
                                mythic, over
                            ):
                                available[opp] = over
        if any(v for v in available_moves.values()):
            yield PlayYield(
                player.id_,
                ActionPhase.USE_SKILL,
                1,
                ActionType.SELECT_SKILL,
                [Action.MOVE_OTHER],
            )
        else:
            yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, []
            )
        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE_OPP,
            2,
            ActionType.SELECT_SELF,
            [board_to_action(*k) for k, v in available_moves.items() if v],
        )

        mythic = action_to_board(action)

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE_OPP,
            1,
            ActionType.SELECT_OPP,
            [board_to_action(*k) for k in available_moves[mythic].keys()],
        )
        player.move_other -= 1
        opp = action_to_board(action)
        # move opponent mythic
        dest = available_moves[mythic][opp]
        game.move_mythic(other_player, opp, dest)

        return False, 0

    def play_move_shelf(self, player: Player, horz: bool) -> PlayOrDoneCoroutine:
        # Monstor: Barge
        # Rotate a Bookshelf into
        # the opposing adjacent space.
        # • You cannot block off a section of the
        # Library completely.
        # • You cannot place a Bookshelf onto the outer
        # edge of the Library or another Bookshelf.
        game = self.game
        assert game
        wall_type = HORZ_WALL if horz else VERT_WALL
        o_wall_type = VERT_WALL if horz else HORZ_WALL

        available_moves: dict[Wall, set[Wall]] = defaultdict(set)

        if player.move_shelf:
            for mythic in player.mythics:
                left = (mythic[0] - 1, mythic[1])
                up = (mythic[0], mythic[1] - 1)

                # don't need to test src for >=4 since it has no walls
                game.test_walls(
                    Wall(wall_type, mythic),
                    available_moves,
                    # Horz
                    (
                        [
                            Wall(o_wall_type, (mythic[0], mythic[1] + 1)),
                            Wall(o_wall_type, (mythic[0] - 1, mythic[1] + 1)),
                        ]
                        if horz
                        else
                        # vert
                        [
                            Wall(o_wall_type, (mythic[0] + 1, mythic[1])),
                            Wall(o_wall_type, (mythic[0] + 1, mythic[1] - 1)),
                        ]
                    ),
                )
                pos = up if horz else left
                if pos[0] >= 0 and pos[1] >= 0:
                    game.test_walls(
                        Wall(wall_type, pos),
                        available_moves,
                        [
                            Wall(o_wall_type, pos),
                            Wall(o_wall_type, (mythic[0] - 1, mythic[1] - 1)),
                        ],
                    )

        if any(src.wall_type == wall_type for src in available_moves.keys()):
            yield PlayYield(
                player.id_,
                ActionPhase.USE_SKILL,
                1,
                ActionType.SELECT_SKILL,
                [Action.MOVE_HORZ_SHELF if horz else Action.MOVE_VERT_SHELF],
            )
        else:
            yield PlayYield(
                player.id_, ActionPhase.USE_SKILL, 1, ActionType.SELECT_SKILL, []
            )

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE_HORZ_SHELF if horz else ActionPhase.MOVE_VERT_SHELF,
            2,
            ActionType.SELECT_HORZ_WALL if horz else ActionType.SELECT_VERT_WALL,
            [board_to_action(*src.pos) for src in available_moves.keys()],
        )
        src = Wall(wall_type, action_to_board(action))

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE_HORZ_SHELF if horz else ActionPhase.MOVE_VERT_SHELF,
            1,
            ActionType.SELECT_VERT_WALL if horz else ActionType.SELECT_HORZ_WALL,
            [board_to_action(*dest.pos) for dest in available_moves[src]],
        )

        player.move_shelf -= 1
        dest = Wall(o_wall_type, action_to_board(action))

        game.board[src.pos] ^= src.wall_type
        game.board[dest.pos] |= dest.wall_type

        return False, 0

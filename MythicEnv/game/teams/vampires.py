"""
Team specific data for Vampires
"""

from MythicEnv.game import (
    Team,
    TeamData,
    PlayOrDoneCoroutine,
    Player,
    PlayYield,
    ActionPhase,
    Action,
    ActionType,
    HORZ_WALL,
    VERT_WALL,
    board_to_action,
    action_to_board,
    MythicMischiefGame,
)
from MythicEnv import *
from collections import defaultdict

VampireData = TeamData(
    name="Vampire",
    move_skill_costs=[3, 4, 5, 6, 7, 8, 9],
    move_other_attr="Lure",
    move_other_skill_costs=[1, 1, 2, 2, 3],
    move_shelf_attr="Conceal",
    move_shelf_skill_costs=[0, 1, 2, 2, 3],
    distract_skill_costs=[0, 1, 1, 2, 3],
    legendary_attr="Trance",
    after_lunch_attr="Blood",
)


# TODO: Add more teams
class Vampire(Team):
    data = VampireData

    def play_move_other(self, game: MythicMischiefGame, player: Player) -> PlayOrDoneCoroutine:
        # Vampire: Lure
        # Move an ally or enemy 1 space
        # orthogonally or diagonally
        # directly toward you.
        # • You can Lure players from any
        # unobstructed distance. See pg. 21.
        # • You can Lure onto a Cluttered space.

        available_dict = dict[Coordinate, dict[Coordinate, Coordinate]]()
        other_player = game.players[player.id_ ^ 1]

        if player.move_other:
            for mythic in player.mythics:
                available = available_dict[mythic] = dict[Coordinate, Coordinate]()

                # Check line of sight to each opponent
                for opp in other_player.mythics:
                    direction = game.line_of_sight(mythic, opp, 2)
                    if direction:
                        available[opp] = direction


        # This yield should return Action.MOVE_OTHER,
        # but because of how this co-routine is called, 
        # the first yield will return None,
        if any(v for v in available_dict.values()):
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
            [board_to_action(*k) for k, v in available_dict.items() if v],
        )

        mythic = action_to_board(action)

        action = yield PlayYield(
            player.id_,
            ActionPhase.MOVE_OPP,
            1,
            ActionType.SELECT_OPP,
            [board_to_action(*k) for k in available_dict[mythic].keys()],
        )

        opp = action_to_board(action)

        # move opponent mythic
        player.move_other -= 1
        direction = available_dict[mythic][opp]
        dest = (opp[0] - direction[0], opp[1] - direction[1])
        game.move_mythic(other_player, opp, dest)

        return False, 0

    def play_move_shelf(self, game: MythicMischiefGame, player: Player, horz: bool) -> PlayOrDoneCoroutine:
        # Vampire: Conceal
        # Rotate a Bookshelf around your space.
        # • You cannot block off a section of the
        # Library completely.
        # • You cannot place a Bookshelf onto the outer
        # edge of the Library or another Bookshelf.

        wall_type = HORZ_WALL if horz else VERT_WALL
        o_wall_type = VERT_WALL if horz else HORZ_WALL

        available_moves: dict[Wall, set[Wall]] = defaultdict(set)
        if player.move_shelf:
            for mythic in player.mythics:
                left = (mythic[0] - 1, mythic[1])
                up = (mythic[0], mythic[1] - 1)

                # don't need to test src for >=4 since it has no walls

                game.test_wall_moves(
                    Wall(wall_type, mythic),
                    available_moves,
                    [
                        Wall(o_wall_type, mythic),
                        Wall(o_wall_type, left if horz else up),
                    ],
                )
                pos = up if horz else left
                if pos[0] >= 0 and pos[1] >= 0:
                    game.test_wall_moves(
                        Wall(wall_type, pos),
                        available_moves,
                        [
                            Wall(o_wall_type, mythic),
                            Wall(o_wall_type, left if horz else up),
                        ],
                    )

        # This yield should return Action.MOVE_SHELF,
        # but because of how this co-routine is called, 
        # the first yield will return None,
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

        game.move_wall(src, dest)

        return False, 0

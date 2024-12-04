import copy
import random
from typing import Any, Optional

import gym.spaces
import numpy as np
import pygame

from easydict import EasyDict

from ding.envs.env.base_env import BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
import pygame.freetype
import pygame.locals
from gymnasium import spaces
import gym

from . import *
from MythicEnv.renderable import *
from MythicEnv.data import *
from MythicEnv.play import PlayableEnv, play_env
from MythicEnv.game import *
from MythicEnv.game.teams import *
from MythicEnv.surface_render import MythicMischiefRenderable, find_font

# from typing import TypeVar
# YIELD = TypeVar("YIELD")
# RESULT = TypeVar("RESULT")
# SEND = TypeVar("SEND")

# def yield_from_send(gen: Generator[YIELD, SEND, RESULT], value: SEND) -> Generator[YIELD, SEND, RESULT]:
#     """Utility function to start a yield from with a value for an already started generator"""
#     try:
#         while True:
#             value = yield gen.send(value)
#     except StopIteration as e:
#         return e.value


# Assumptions
# Cannot "Stop", or select another mythic, when moving a mythic
# Cannot move through own mythics

# Future assumptions
# Only one Decay per spot


ACTION_SPACE_LEN = 8+5*5
assert ACTION_SPACE_LEN == len(Action) + 5*5

@ENV_REGISTRY.register("mythic_mischief_v0")
class MythicMischiefEnv(PlayableEnv[np.int8, np.int64]):
    """Env for Mythic Mischief"""

    render_mode: str
    battle_mode: str
    game_state: Optional[MythicMischiefGame]
    play: Optional[PlayOrDoneCoroutine]
    to_play: int
    available_action_phase: Optional[ActionPhase]
    available_actions_left: int
    available_action_type: Optional[ActionType]
    available_actions: list[int]
    cfg: dict[str, Any]



    config: dict[str, Any] = dict(
        # (str) The name of the environment registered in the environment registry.
        env_id="MythicMischief_v0",
        # (str) The mode of the environment when take a step.
        battle_mode="self_play_mode",
        # (str) The mode of the environment when doing the MCTS.
        battle_mode_in_simulation_env="self_play_mode",
        # (str) The render mode. Options are 'None', 'human', 'rgb_array'
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (str) The type of the bot of the environment.
        # bot_action_type='rule',
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0,
        # (float) The probability that an expert agent(the bot) is used instead of the learning agent.
        prob_expert_agent=0,
        # (float) The probability that a random action will be taken when calling the bot.
        prob_random_action_in_bot=0.,
        # (float) The scale of the render screen.
        screen_scaling=1,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=False,
        # (bool) Whether to scale the observation.
        scale=False,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        stop_value=2,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"  # type: ignore
        return cfg

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        # Load the config.

        assert cfg
        self.cfg = cfg

        self.render_mode = cfg["render_mode"]

        self.battle_mode = cfg["battle_mode"]
        assert self.battle_mode in ["self_play_mode", "play_with_bot_mode", "eval_mode"]
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = "self_play_mode"

        # play all keeper moves, even when there is only one option
        self.all_keeper_moves = False

        # ensure safe defaults. Most set in reset

        self.available_actions = list[int]()
        self.available_action_type: Optional[ActionType] = None
        self.confirming_action = None

        self.team_ids = {
            t:i for i,t in enumerate(Team.__subclasses__())
        }
        assert len(self.team_ids) <= 4
        self.game_state = None


        # Setup pygame GUI
        if self.render_mode == "surface":

            if not pygame.get_init():
                pygame.init()
            if not pygame.freetype.get_init():
                pygame.freetype.init()

            # Choose a font size
            # GNU unifont has a LOT of unicode characters
 
            self.renderable = MythicMischiefRenderable(self, find_font(20))
            
           
            self.surface_size = self.renderable.size()

        # TODO: Define action_space and observation space

        # Crate easier names for actions
        self.action_names = [str(action) for action in Action] + [
            f"{x},{y}"
            for x, y in (action_to_board(i + len(Action)) for i in range(5 * 5))
        ]




    def reset(
        self,
        start_player_index: int = 0,
        init_state: Optional[Any] = None,
        replay_name_suffix: Optional[str] = None,
    ) -> dict[str, Any]:
        """Reset the game state"""

        self.confirming_action = None

        # TODO: Randomize cards

        # initialize board
        # Player initialization
        # TODO: Randomize players
        self.game_state = MythicMischiefGame((Vampire, Monster))

        # for representing observations
        assert self.game_state.dest_card.keeper_moves <= 4
        assert len(self.game_state.dest_card.books) <= 4

        # test data
        if False:
            # Random starting positions
            mask = PLAYER_MASK | KEEPER | BOOKS_MASK

            available = [
                (i, j)
                for i in range(5)
                for j in range(5)
                if not (self.board[i][j] & mask)
            ]

            starts = random.sample(available, 6)

            for x, y in starts[:3]:
                self.board[x, y] |= PLAYER[0]
                self.players[0].mythics.add((x, y))
            for x, y in starts[3:]:
                self.board[x, y] |= PLAYER[1]
                self.players[1].mythics.add((x, y))

        # create play generator/co-routine
        self.play = self.game_state.start_play()

        # Start the play sequence
        (
            self.to_play,
            self.available_action_phase,
            self.available_actions_left,
            self.available_action_type,
            self.available_actions,
        ) = next(self.play)

        self.history = []

        return self.get_observation()

    def confirm_action(self, action: int):
        """Confirm if action is valid and highlight in gui"""
        if action in self.available_actions:
            self.confirming_action = action
            return True
        self.confirming_action = None
        return False

    def step(self, action: int) -> BaseEnvTimestep:
        """Step through play"""
        action = int(action)
        assert (
            action in self.available_actions
        ), f"invalid action: {action}. {action_to_board(action) if action >= len(Action) else ''}  "
        # We are taking the action, so confirmation logic is reset
        self.confirming_action = None
        self.history.append((
            self.to_play, 
            self.available_action_phase.name,
            self.available_action_type.name,
            action, 
            Action(action).name if action < len(Action) else "", 
            action_to_board(action) if action >= len(Action) else None
        ))

        game_state = self.game_state
        assert game_state
        assert self.play

        try:
            # Send action to Play co-routine and get availible next steps
            # This results are what the co-routine yields
            (
                self.to_play,
                self.available_action_phase,
                self.available_actions_left,
                self.available_action_type,
                self.available_actions,
            ) = self.play.send(action)
            assert (
                self.available_action_phase is not None
                and self.available_actions_left
                and self.available_action_type is not None
            )

            meta: dict[str, Any] = {
                "player_name": game_state.players[self.to_play].team.data.name.title(),
                "action_phase": self.available_action_phase.name.replace(
                    "_", " "
                ).title(),
                "actions_left": self.available_actions_left,
                "action_type": self.available_action_type.name.replace(
                    "_", " "
                ).title(),
            }

            return BaseEnvTimestep(
                self.get_observation(),
                np.array(0.0).astype(np.float32),
                False,
                meta,
            )
        except StopIteration as e:
            # Play co-routine returned
            done, reward = e.value
        assert done

        reward = np.array(float(reward)).astype(np.float32)

        # Done.  Ensure to_play for calling logic
        meta: dict[str, Any] = {
            "player_name": game_state.players[self.to_play].team.data.name.title(),
            "eval_episode_return":  -reward if self.to_play == 1 else reward
        }
        return BaseEnvTimestep(self.get_observation(), reward, True, meta)

    def get_observation(self) -> dict[str, Any]:
        obs = np.zeros((21, 5, 5), np.float32)
        game_state = self.game_state
        assert game_state
        _board = game_state.board

        # c:=c+1   =>  ++c (pre-increment)
        c = -1

        def board(a: int):
            return (_board & a).astype(np.bool_)

        for d in DEST:
            obs[c:=c+1, board(d)] = 1
        obs[c:=c+1, board(KEEPER)] = 1
        obs[c:=c+1, board(CLUTTER)] = 1
        obs[c:=c+1, board(HORZ_WALL)] = 1
        obs[c:=c+1, board(VERT_WALL)] = 1
        obs[c:=c+1, board(BOOKS[0])] = 1
        obs[c:=c+1, board(PLAYER[0])] = 1
        # obs[c:=c+1, :, :] = 0  team1 after lunch
        obs[c:=c+1, board(BOOKS[1])] = 1
        obs[c:=c+1, board(PLAYER[1])] = 1
        # obs[c:=c+1] team3 after lunch


        # Player state (each 4 levels)
        # TODO: Swap players?
        c=c+1
        for player in game_state.players:

            # (0,0) -> (0,4)
            # TODO: Handle more than 4 teams
            team_id = self.team_ids[type(player.team)]
            assert 0<=team_id<=3
            obs[c + team_id, 0, 0] = 1

            # 0 < Score <= 4
            score = player.score
            score1 = min(score, 4)
            if score1:
                assert 1<=score1<=4
                obs[c: c + score1, 0, 1] = 1
            # 4 < score <= 8
            score2 = min(score - 4, 4)
            if score2 > 0:
                assert 1<=score2<=4
                obs[c: c + score2, 0, 2] = 1
            # score == 9 on game state channel
            # score >= 10 not counted as that is endgame

            # More action phases later and in game state channel
            if self.available_action_phase == ActionPhase.BOOST:
                assert 1<=self.available_actions_left <= 4
                obs[c: c + self.available_actions_left, 0, 3] = 1
            if self.available_action_phase == ActionPhase.SPEND_TOME:
                assert 1<=self.available_actions_left <= 4
                obs[c: c + self.available_actions_left, 0, 4] = 1

            # (1,0) -> (1,4)
            if player.move_tomes:
                assert 1<=player.move_tomes<=4
                obs[c: c + player.move_tomes, 1, 0] = 1
            if player.move_other_tomes:
                assert 1<=player.move_other_tomes<=4
                obs[c: c + player.move_other_tomes, 1, 1] = 1
            if player.move_shelf_tomes:
                assert 1<=player.move_shelf_tomes<=4
                obs[c: c + player.move_shelf_tomes, 1, 2] = 1
            if player.distract_tomes:
                assert 1<=player.distract_tomes<=4
                obs[c: c + player.distract_tomes, 1, 3] = 1
            if player.tomes:
                assert 1<=player.tomes<=4
                obs[c: c + player.tomes, 1, 4] = 1

            # (2,0) -> (2,4)
            # 0 < moves <= 4
            moves = player.move
            moves1 = min(moves, 4)
            if moves1:
                assert 1<=moves1<=4
                obs[c: c + moves1, 2, 0] = 1
            # 4 < moves <= 8 see below [3,0]
            # moves == 9 is handled in game state channel
            if player.move_other:
                assert 1<=player.move_other<=4
                obs[c: c + player.move_other, 2, 1] = 1
            if player.move_shelf:
                assert 1<=player.move_shelf<=4
                obs[c: c + player.move_shelf, 2, 2] = 1
            if player.distract:
                assert 1<=player.distract<=4
                obs[c: c + player.distract, 2, 3] = 1
            if player.legendary:
                obs[c, 2, 4] = 1

            # 3,0 -> (3,4)
            # 4 < moves <= 8
            moves2 = min(player.move - 4, 4)
            if moves2 > 0:
                assert 1<=moves2<=4
                obs[c: c + moves2, 3, 0] = 1
            # # 0 <= moves <= 4 see above [2,0]
            # moves == 9 is handled in game state channel

            # More action phases earlier and in game state channel
            if self.available_action_phase == ActionPhase.MOVE_OPP:
                assert 1<=self.available_actions_left<=4
                obs[c: c + self.available_actions_left, 3, 1] = 1
            if self.available_action_phase == ActionPhase.MOVE_HORZ_SHELF:
                assert 1<=self.available_actions_left<=4
                obs[c: c + self.available_actions_left, 3, 2] = 1
            if self.available_action_phase == ActionPhase.DISTRACT:
                assert 1<=self.available_actions_left<=4
                obs[c: c + self.available_actions_left, 3, 3] = 1
            if self.available_action_phase == ActionPhase.LEGENDARY:
                assert 1<=self.available_actions_left<=4
                obs[c: c + self.available_actions_left, 3, 4] = 1

            # (4,0) -> (4,4)
            if self.available_action_phase == ActionPhase.MOVE:
                obs[c: c + self.available_actions_left, 4, 0] = 1
            if self.available_action_phase == ActionPhase.PLACE_MYTHIC:
                obs[c: c + self.available_actions_left, 4, 1] = 1
            if self.available_action_phase == ActionPhase.MOVE_VERT_SHELF:
                obs[c: c + self.available_actions_left, 4, 2] = 1
            if self.available_action_phase == ActionPhase.AFTER_LUNCH:
                obs[c: c + self.available_actions_left, 4, 3] = 1

            if player.after_lunch_special:
                assert 1<=player.after_lunch_special<=4
                obs[c: c + player.after_lunch_special, 4, 4] = 1

            c=c+4

        # General Game State
        # (0,0) -> (0,4)
        if game_state.after_lunch:
            obs[c, 0, 0] = 1
        if game_state.players[0].score == 9:
            obs[c, 0, 1] = 1
        if game_state.players[1].score == 9:
            obs[c, 0, 2] = 1
        if game_state.players[0].move == 9:
            obs[c, 0, 3] = 1
        if game_state.players[1].move == 9:
            obs[c, 0, 4] = 1

        # (1,0) -> (1,4)
        # Available Action Types on General Game State channel
        # TODO: Split? We have extra space
        if self.available_action_type in (
            ActionType.SELECT_SELF,
            ActionType.SELECT_MYTHIC,
        ):
            obs[c, 1, 0] = 1
        # TODO: Split?  We have extra space
        if self.available_action_type in (
            ActionType.SELECT_OPP,
            ActionType.SELECT_MYTHIC,
        ):
            obs[c, 1, 1] = 1
        if self.available_action_type == ActionType.SELECT_DEST:
            obs[c, 1, 2] = 1
        if self.available_action_type == ActionType.SELECT_HORZ_WALL:
            obs[c, 1, 3] = 1
        if self.available_action_type == ActionType.SELECT_VERT_WALL:
            obs[c, 1, 4] = 1

        # (2,0) -> (2,4)
        if self.available_action_type == ActionType.SELECT_SKILL:
            obs[c, 2, 0] = 1
        # Available Action Phases with more than one actions left are on player channels above.
        # The ones here on General Game State channel have 1 or unknown actions left
        if self.available_action_phase == ActionPhase.MOVE_KEEPER:
            obs[c, 2, 1] = 1
        if self.available_action_phase == ActionPhase.DECAY_MOVE:
            obs[c, 2, 2] = 1

        # (3,0) -> (4,4)
        # Non-Position Available Actions on general game state channel
        mapping: dict[int, tuple[int, int]] = {
            Action.PASS: (3, 0),
            Action.MOVE: (3, 1),
            Action.MOVE_OTHER: (3, 2),
            Action.MOVE_HORZ_SHELF: (3, 3),
            Action.MOVE_VERT_SHELF: (3, 4),
            Action.LEGENDARY: (4, 0),
            Action.AFTER_LUNCH: (4, 1),
        }
        for action in self.available_actions:
            if action in mapping:
                obs[c, mapping[action]] = 1

        # Available Action positions as separate channel
        c=c+1
        if self.available_action_type in [
            ActionType.SELECT_SELF,
            ActionType.SELECT_OPP,
            ActionType.SELECT_MYTHIC,
            ActionType.SELECT_DEST,
            ActionType.SELECT_HORZ_WALL,
            ActionType.SELECT_VERT_WALL,
        ]:
            for a in self.available_actions:
                x, y = action_to_board(a)
                obs[c, x, y] = 1


        assert c == 20
        assert obs.shape==(21,5,5)

        # Action Mask
        action_mask = np.zeros(ACTION_SPACE_LEN, "int8")
        for i in self.available_actions:
            action_mask[i] = 1
        assert action_mask.shape==(33,)

        return {
            'observation': obs,
            'action_mask': action_mask,
            'to_play': self.to_play
        }

    @property
    def legal_actions(self) -> list[int]:
        return self.available_actions
    

    action_space = cast(gym.spaces.Space[np.int64], spaces.Discrete(ACTION_SPACE_LEN))
    observation_space = cast(gym.spaces.Space[np.int8], spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(23, 5, 5), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_LEN,), dtype=np.int8),
                "to_play": spaces.Discrete(2),
            }
        )
    )
    reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def render(
        self, mode: Optional[str] = None
    ) -> None | np.ndarray[tuple[int, int, int], Any] | pygame.Surface:
        """Render current state for human review"""
        render_mode = self.render_mode
        assert mode is None or mode == render_mode
        if render_mode in ("rgb_array", "surface"):
            # for rgb_array, we still use pygame, then generate a ndarray
            screen = pygame.Surface(self.surface_size)
            screen.fill(BLACK)

            self.renderable.render(screen)

            if mode == "rgb_array":
                return pygame.surfarray.pixels3d(screen)  # type: ignore
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
        random.seed(self._seed)




if __name__ == "__main__":
    cfg = MythicMischiefEnv.default_config()
    cfg["render_mode"] = "surface"
    play_env(MythicMischiefEnv(cfg))

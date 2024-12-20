import abc
import random
from typing import TYPE_CHECKING, Any, Optional, TypeVar, TypedDict, cast, Generic

from ding.envs.env.base_env import BaseEnv
import pygame
import pygame.freetype
import gymnasium.spaces
import numpy as np

from MythicEnv.renderable import Text, RenderableEnv, Flex, Box, BLACK, toggle_blink

ObsType = TypeVar("ObsType", bound=np.generic)
ActType = TypeVar("ActType")


# BaseEnv is not actually a generic, but its parent gym.Env is.
# BaseEnv is declared generic in the typing data
# This is hack to get around this and keep typing info, without changing ding

if TYPE_CHECKING:
    class _BaseEnv(BaseEnv[ObsType, ActType]):
       pass
else:
    class _BaseEnv(BaseEnv, Generic[ObsType, ActType]):
        pass

class ObservationDict( TypedDict, Generic[ObsType]):
    observation: np.ndarray[tuple[int,int,int], np.dtype[ObsType]]
    action_mask: np.ndarray[tuple[int], np.dtype[np.uint8]]
    to_play: int

class ObservationSpaceDictInit(TypedDict, Generic[ObsType]):
    observation: gymnasium.Space[ObsType]
    action_mask: gymnasium.Space[np.uint8]
    to_play: gymnasium.Space[np.uint64]

    
class ObservationSpaceDict(gymnasium.spaces.Dict):
    pass

class PlayableEnv( _BaseEnv[ObsType, ActType] ):
    """An env that can be played"""

    action_names: list[str]

    @abc.abstractmethod
    def confirm_action(self, action: int) -> bool: ...


def prompt(to_play: int, buffer: str, confirmed: bool, meta: dict[str, Any]):
    """Dynamic prompt"""
    # Own function to allow easy use of format strings
    player_name = f" <{meta['player_name']}>" if "player_name" in meta else ""
    actions_left = f" ({meta['actions_left']})" if "actions_left" in meta else ""
    action_phase = (
        f"{meta['action_phase']}{actions_left}: " if "action_phase" in meta else ""
    )
    action_type = meta["action_type"] if "action_type" in meta else "Specify Action"

    return f"{action_phase}{action_type} for Player {to_play}{player_name}: {buffer}{'   (Enter to play)' if confirmed else ''}"


def save_recording(recording: list[str]):
    with open("recording.txt", "w") as f:
        f.writelines([l + "\n" for l in recording])

def load_script() -> list[str]:
    with open("script.txt") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


def play_env(env: PlayableEnv[ActType, ObsType], seed: Optional[int] = None):
    """Play a game"""
    if seed is not None:
        env.seed(seed)
    env.reset()

    # Reversk the map so we can generate actions
    name_to_action = {k: v for v, k in enumerate(env.action_names)}

    running = True
    done = True  # Causes a first reset
    obs: Optional[dict[str, Any]] = None
    action: Optional[int] = None
    meta = dict[str, Any]()

    # Pygame, yay!
    pygame.init()
    pygame.freetype.init()
    font = pygame.freetype.SysFont("unifont", 20)

    # Play GUI consists of the ENV and a dynbamic prompt
    max_prompt = max(len(prompt(1, action, True, {})) for action in env.action_names)
    curr_prompt = ""

    class ActionBar(Text):
        """Renderable Text Subcalss for dynamic prompt"""

        def text(self) -> str:
            return curr_prompt

    display = Flex(
        horz=False, items=[RenderableEnv(env), Box(ActionBar(font, max_prompt))]
    )

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
    env.seed(1)
    random.seed(1)
    script = []
    if True:
        script = load_script()
    recording = list[str]()
    while running:
        curr_prompt = prompt(to_play, buffer, confirmed, meta)
        if done:
            # Restart
            done = False
            if seed is not None:
                env.seed(seed)
            obs = env.reset()
            assert obs
            to_play = cast(int, obs["to_play"])
            buffer = ""
            confirmed = False
            action = None
        else:
            # Execute next step in script
            if script:
                if script[0] == 'random':
                    if obs and "action_mask" in obs:
                        available_actions = [i for i,v in enumerate(obs['action_mask']) if v]
                        action = random.choice(available_actions)
                        print(f"Playing Action: {script[0]}: {action}")
                        script.pop(0)
                else:
                    print(f"Playing Action: {script[0]}")
                    action = name_to_action[script.pop(0)]

            # Execute pending actions
            if action is not None:
                # prev_obs = obs
                recording.append(env.action_names[action])
                save_recording(recording)
                obs, _, done, meta = env.step(action)
                assert done or obs is not None
                to_play = cast(int, obs["to_play"])
                action = None

            # Render data
            if obs is not None:
                # rendered = env.render()
                # rendered = pygame.transform.scale(rendered, video_size)
                screen.fill(BLACK)
                display.render(screen)

            # Process all pending events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.QUIT:
                    running = False

                # elif event.type == pygame.locals.VIDEORESIZE:
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
                            confirmed = env.confirm_action(-1)
                elif event.type == pygame.KEYDOWN:
                    buffer = buffer + event.unicode
                    if buffer in name_to_action:
                        confirmed = env.confirm_action(name_to_action[buffer])
                    else:
                        confirmed = env.confirm_action(-1)

                # Timer event for blinking
                elif event.type == pygame.USEREVENT:
                    toggle_blink()

            pygame.display.flip()
            clock.tick(10)
    pygame.quit()
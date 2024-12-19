from typing import Any, Callable, NamedTuple, Optional, TypeVar, Generic, Generator, Iterable, Iterator, cast
import abc
from dataclasses import dataclass


T_Yield = TypeVar("T_Yield")
T_Return = TypeVar("T_Return")
T_Send = TypeVar("T_Send")
T = TypeVar("T")
S = TypeVar("S")


class NIL:
    pass


Nil = NIL()


@dataclass
class Yield(Generic[T_Yield]):
    value: T_Yield


@dataclass
class Return(Generic[T_Return]):
    value: T_Return


class Break:
    pass


@dataclass
class ClonableGenerator(abc.ABC, Generic[T_Yield, T_Send, T_Return]):
    step: int = 0

    @abc.abstractmethod
    def send(self, value: Optional[T_Send]) -> Yield[T_Yield] | Return[T_Return]: ...

    def as_generator(self) -> Generator[T_Yield, T_Send, T_Return]:
        # Note: the return value is NOT clonable.
        # Used for compatibity when converting
        v = None
        while True:
            y_or_r = self.send(v)
            if isinstance(y_or_r, Yield):
                yield y_or_r.value
            else:
                return y_or_r.value


@dataclass
class YieldFrom(Generic[T_Yield, T_Send, T_Return]):
    gen: (
        ClonableGenerator[T_Yield, T_Send, T_Return]
        | Generator[T_Yield, T_Send, T_Return]
    )


class ReturnYieldFrom(
    Generic[T_Yield, T_Send, T_Return], YieldFrom[T_Yield, T_Send, T_Return]
):
    pass


class LoopState:
    end_step: Optional[int] = None


@dataclass
class ForLoopState(LoopState, Generic[T]):
    base_it: Iterator[T]
    start_value: T | NIL = Nil


@dataclass
class WhileLoopState(LoopState):
    start_value: bool = False


class LoopStateContainer(NamedTuple):
    by_step: dict[int, LoopState]
    stack: list[LoopState]


class ClonableGeneratorImpl(ClonableGenerator[T_Yield, T_Send, T_Return]):
    _yield_from: YieldFrom[T_Yield, T_Send, Any] | None
    yield_from_result: Any
    loop_state: LoopStateContainer

    def __init__(self):
        self._yield_from = None
        self._next_auto_step = 0

    @abc.abstractmethod
    def send_impl(
        self, value: Optional[T_Send]
    ) -> (
        Yield[T_Yield]
        | Return[T_Return]
        | YieldFrom[T_Yield, T_Send, Any]
        | Break
    ): ...

    def send(self, value: Optional[T_Send]) -> Yield[T_Yield] | Return[T_Return]:
        while True:
            if self._yield_from is None:
                # Auto step starts afresh each send
                self._next_auto_step = 0

                s = self.send_impl(value)
                value = None
                if isinstance(s, (Yield, Return)):
                    self.complete_step()
                    return s
                if isinstance(s, Break):
                    self.complete_step()
                    # notify loop that it is done
                    self.loop_state.stack.pop(-1).end_step = self.step
                    continue
                assert isinstance(s, YieldFrom)
                self._yield_from = s
                    
            assert self._yield_from is not None
            if isinstance(self._yield_from.gen, ClonableGenerator):
                y_or_r = self._yield_from.gen.send(value)
                if isinstance(y_or_r, Yield):
                    return y_or_r
                self.yield_from_result = y_or_r.value
            else:
                try:
                    if value is None:
                        return Yield(next(self._yield_from.gen))
                    else:
                        return Yield(self._yield_from.gen.send(value))
                except StopIteration as e:
                    self.yield_from_result = e.value
            self.complete_step()
            yf = self._yield_from
            self._yield_from = None
            value = None
            if isinstance(yf, ReturnYieldFrom):
                yfr = self.yield_from_result
                self.yield_from_result = None
                return Return(yfr)

    def next_step(self, custom_step: Optional[int] = None) -> bool:
        "Check step condition"
        # next_auto_step gets updated everytime through send_impl.
        # However, if these are always top level, next_auto_step will have the
        # same value each time through send_impl.  We can then check against
        # step, which is preserved across send_impl to track state
        s = self._next_auto_step
        self._next_auto_step += 1
        # if self.step == -1:
        #    self.step = s
        return self.step == s

    def skip_next_step(self):
        self.step += 1
        self.complete_step()

    def complete_step(self):
        self.step += 1

    def for_loop(
        self,
        i: Iterable[T],
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> Iterator[T]:
        # this gets called once for each send_impl.  It sets up its own
        # iterator to keep track of state
        # The inner iterater starts with returning whatever the base_it
        # returned last.  This way, each call through send_impl gets the
        # same value.  Only increments base_it if it is looped more than once
        # in a single send_impl call

        # loop_state isn't created unless it is needed
        try:
            loop_state = self.loop_state
        except AttributeError:
            loop_state = self.loop_state = LoopStateContainer({}, [])

        # # auto_step logic
        if start_step is None:
            start_step = self._next_auto_step
            self._next_auto_step += 1
        else:
            self._next_auto_step = start_step + 1

        if self.step == start_step:
            base_it = iter(i)

            # We can't increment iterator here. We want to do that within the
            # inner iterator, so StopIteration is handled corrctly
            state = ForLoopState[T](base_it)
            loop_state.by_step[start_step] = state
            loop_state.stack.append(state)
        else:
            state = loop_state.by_step[start_step]
            assert isinstance(state, ForLoopState)
            state = cast(ForLoopState[T], state)  # satisfy generic type check

            if state.end_step is not None:
                # loop complete, skip inner part of loop
                # ensure steps after ase autonumbered consistently
                if end_step is not None:
                    state.end_step = end_step
                self._next_auto_step = state.end_step
                return iter(())

        outer = self

        class inner_it(Iterator[S]):
            def __init__(self):
                self.first_run = True

            def __next__(self) -> T:
                assert start_step <= outer.step
                if self.first_run:
                    # First run always returns the last value from base_it
                    self.first_run = False
                    if not isinstance(state.start_value, NIL):
                        return state.start_value

                # Advance iterator and check for exit
                try:
                    next_value = next(state.base_it)
                except StopIteration:
                    # iterator done, break loop
                    outer.complete_step()
                    loop_state.stack.pop(-1)
                    # save end_step to ensure next passes preserve
                    if end_step is not None:
                        outer.step = end_step
                    state.end_step = outer._next_auto_step = outer.step
                    raise

                # save across send_impl. This will be returned in the next send_impl
                state.start_value = next_value
                # the loop starts at the beginning
                outer.step = outer._next_auto_step = start_step + 1
                return next_value

            def __iter__(self):
                return self

        return inner_it[T]()

    def while_loop(
        self,
        condition: bool | Callable[[], bool],
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ):
        # loop state isn't created unless it is needed
        try:
            loop_state = self.loop_state
        except AttributeError:
            loop_state = self.loop_state = LoopStateContainer({}, [])

        # # auto_step logic
        if start_step is None:
            start_step = self._next_auto_step
            self._next_auto_step += 1
        else:
            self._next_auto_step = start_step + 1

        if self.step == start_step:
            state = WhileLoopState()
            loop_state.by_step[start_step] = state
            loop_state.stack.append(state)
        else:
            state = loop_state.by_step[start_step]
            assert isinstance(state, WhileLoopState)

            if state.end_step is not None:
                # Loop complete Skip inner part of loop
                # ensure steps after ase autonumbered consistently
                self._next_auto_step = state.end_step
                return iter(())

        outer = self

        class inner_it(Iterator[None]):
            def __init__(self):
                self.first_run = True

            def __next__(self) -> None:
                assert start_step <= outer.step
                if self.first_run:
                    # First run always returns the last value condition
                    self.first_run = False
                    if state.start_value:
                        # Condition True, continue loop
                        return None

                # check condition
                if isinstance(condition, Callable):
                    next_value = condition()
                else:
                    next_value = condition

                if not next_value:
                    # Condition false, break loop
                    outer.complete_step()
                    loop_state.stack.pop(-1)
                    # save end_step to ensure next passes preserve
                    if end_step is not None:
                        outer.step = end_step
                    state.end_step = outer._next_auto_step = outer.step
                    raise StopIteration

                # save across send_impl. This will be returned in the next send_impl
                state.start_value = next_value
                outer.step = outer._next_auto_step = (
                    start_step + 1
                )  # the loop starts at the beginning
                return None

            def __iter__(self):
                return self

        return inner_it()
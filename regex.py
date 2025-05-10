from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Set


class State(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def check_self(self, char: str) -> bool:
        """
        function checks whether occured character is handled by current ctate
        """
        pass

    def check_next(self, next_char: str) -> State | Exception:
        for state in self.next_states:
            if state.check_self(next_char):
                return state
        raise NotImplementedError("rejected string")


class StartState(State):
    """
    Initial state for the regex FSM
    """

    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return False
        
    def __repr__(self):
        return 'START'


class TerminationState(State):
    """
    Final accepting state for the regex FSM
    """
    
    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return False

    def __repr__(self):
        return 'END'


class DotState(State):
    """
    State for . character (matches any single character)
    """

    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return True and char != ''

    def __repr__(self):
        return '.'

class AsciiState(State):
    """
    State for specific ASCII characters (letters, numbers, etc.)
    """

    def __init__(self, symbol: str) -> None:
        self.next_states: list[State] = []
        self.symbol = symbol

    def check_self(self, char: str) -> bool:
        return char == self.symbol

    def __repr__(self):
        return self.symbol


class StarState(State):
    def __init__(self, checking_state: State):
        self.next_states: list[State] = []
        self.checking_state = checking_state

    def check_self(self, char: str) -> bool:
        if self.checking_state.check_self(char):
            return True

        for state in self.next_states:
            if state.check_self(char):
                return True
                
        return False
        
    def __repr__(self):
        return f'{self.checking_state}*'


class PlusState(State):
    def __init__(self, checking_state: State):
        self.next_states: list[State] = []
        self.checking_state = checking_state
        self.matched_once = False

    def check_self(self, char: str) -> bool:
        if self.checking_state.check_self(char):
            self.matched_once = True
            return True
        return False
        
    def __repr__(self):
        return f'{self.checking_state}+'


class StartAnchor(State):
    """
    Anchor state that ensures patterns start at the beginning of the string.
    This is used to enforce that patterns match from the start, not from the middle.
    """
    
    def __init__(self):
        self.next_states: list[State] = []
        
    def check_self(self, char: str) -> bool:
        return False
        
    def __repr__(self):
        return '^'


class CharacterClassState(State):
    """
    State for character classes like [a-z0-9]
    Supports ranges (a-z) and individual characters
    """
    
    def __init__(self, class_definition=None, ranges=None, chars=None):
        self.next_states: list[State] = []
        self.ranges = ranges or []
        self.chars = chars or set()
        
        if class_definition is not None:
            self._parse_class_definition(class_definition)
    
    def _parse_class_definition(self, definition: str):
        """Parse the character class definition to determine allowed characters and ranges."""
        i = 0
        while i < len(definition):
            if i + 2 < len(definition) and definition[i+1] == '-':
                start_char = definition[i]
                end_char = definition[i+2]
                self.ranges.append((start_char, end_char))
                
                for char_code in range(ord(start_char), ord(end_char) + 1):
                    self.chars.add(chr(char_code))
                
                i += 3
            else:
                self.chars.add(definition[i])
                i += 1
    
    def check_self(self, char: str) -> bool:
        """
        Check if the character is accepted by this character class.
        This is the implementation required by the State abstract class.
        """
        for start, end in self.ranges:
            if start <= char <= end:
                return True
                
        return char in self.chars
    
    def accepts(self, char):
        return self.check_self(char)
    
    def __repr__(self):
        range_str = ",".join([f"{s}-{e}" for s, e in self.ranges])
        chars_str = "".join(sorted(self.chars))
        return f"CharClass[{range_str}{';' if self.ranges and self.chars else ''}{chars_str}]"


class RegexFSM:
    def __init__(self, regex_expr: str) -> None:
        self.start = StartState()
        self._original_regex = regex_expr
        
        self.parse_regex(regex_expr)
        
    def parse_regex(self, regex_expr: str) -> None:
        """Parse the regex expression and build the FSM."""
        anchor = StartAnchor()
        self.start.next_states.append(anchor)
        
        term_state = TerminationState()
        
        if not regex_expr:
            anchor.next_states.append(term_state)
            return
        
        i = 0
        prev_state = None
        states = []
        
        while i < len(regex_expr):
            char = regex_expr[i]
            
            if char in ['*', '+']:
                if not prev_state or not states:
                    raise ValueError(f"{char} must follow a character")
                
                states.pop()
                
                if char == '*':
                    new_state = StarState(prev_state)
                    if isinstance(prev_state, PlusState):
                        prev_state.matched_once = True
                elif char == '+':
                    new_state = PlusState(prev_state)

                states.append(new_state)
                prev_state = new_state
                i += 1
            elif char == '[':
                class_end = regex_expr.find(']', i + 1)
                if class_end == -1:
                    raise ValueError("Unclosed character class []")
                
                class_def = regex_expr[i+1:class_end]
                if not class_def:
                    raise ValueError("Empty character class []")
                
                new_state = CharacterClassState(class_def)
                states.append(new_state)
                prev_state = new_state
                
                i = class_end + 1
            else:
                if char == '.':
                    new_state = DotState()
                else:
                    new_state = AsciiState(char)
                
                states.append(new_state)
                prev_state = new_state
                i += 1
        
        states.append(term_state)
        
        prev_state = anchor
        for state in states:
            prev_state.next_states.append(state)
            prev_state = state
        
        self.termination_state = term_state

    def find_parent_state(self, current: State, target: State) -> State:
        """Find the parent state of the target state"""
        for next_state in current.next_states:
            if next_state == target:
                return current
            
            parent = self.find_parent_state(next_state, target)
            if parent:
                return parent
                
        return None

    def check_string(self, string):
        """
        Check if the given string matches the regex pattern.
        In our regex engine, the pattern must match the entire string from beginning to end.
        
        This implementation ensures that patterns only match when they start at
        the beginning of the string and consume the entire string.
        """
        self._reset_plus_states()
        
        if self._is_fixed_pattern_end() and len(string) > 0:
            last_char = string[-1]
            pattern_can_end_with = self._can_pattern_end_with(last_char)
            if not pattern_can_end_with:
                return False
        
        return self._match_from_start(string)

    def _reset_plus_states(self):
        """Reset all plus state matched_once flags to ensure clean state."""
        self._visit_and_reset_state(self.start, set())
        
    def _visit_and_reset_state(self, state, visited):
        """Visit all states and reset plus state flags."""
        if id(state) in visited:
            return
        
        visited.add(id(state))
        
        if isinstance(state, PlusState):
            state.matched_once = False
        
        for next_state in state.next_states:
            self._visit_and_reset_state(next_state, visited)
        
    def _get_accepting_states(self, states):
        """Find all accepting states among the current states."""
        accepting_states = set()
        
        for state in states:
            if isinstance(state, TerminationState):
                accepting_states.add(state)
                
        if accepting_states:
            return accepting_states
            
        for state in states:
            if isinstance(state, PlusState) and not state.matched_once:
                continue
            
            next_states = self._follow_epsilon({state})
            for next_state in next_states:
                if isinstance(next_state, TerminationState):
                    accepting_states.add(state)
                    break
                
        return accepting_states

    def _process_char(self, states, char, is_first_char=False):
        """Process a single character and return the next set of states."""
        next_states = set()
        
        for state in states:
            if isinstance(state, TerminationState):
                continue
            
            if isinstance(state, StartState) and not is_first_char:
                continue
            
            if isinstance(state, (AsciiState, DotState)) and state.check_self(char):
                for next_state in state.next_states:
                    next_states.add(next_state)
                    
            elif isinstance(state, StarState):
                checking_state = state.checking_state
                
                if checking_state.check_self(char):
                    next_states.add(state)
                    
                for next_state in state.next_states:
                    if next_state.check_self(char):
                        next_states.add(next_state)
                        
            elif isinstance(state, PlusState):
                checking_state = state.checking_state
                
                if checking_state.check_self(char):
                    state.matched_once = True
                    next_states.add(state)
                    
                if state.matched_once:
                    for next_state in state.next_states:
                        if next_state.check_self(char):
                            next_states.add(next_state)
                            
            elif isinstance(state, StartState) and is_first_char:
                for next_state in state.next_states:
                    if next_state.check_self(char):
                        next_states.add(next_state)
                        
        return self._follow_epsilon(next_states)
        
    def _check_acceptance(self, states):
        """Check if any of the current states is an accepting state."""
        for state in states:
            if isinstance(state, TerminationState):
                return True
            
        for state in states:
            if isinstance(state, PlusState) and not state.matched_once:
                continue
            
            next_states = self._follow_epsilon({state})
            for next_state in next_states:
                if isinstance(next_state, TerminationState):
                    return True
                
        return False
        
    def get_regex_str(self):
        """Return the regex string that was used to create this FSM."""
        return self._original_regex

    def _follow_epsilon(self, states, is_initial=True):
        """Follow all epsilon transitions from the given states.
        
        This method handles the following special cases:
        - StartAnchor states: transition to their next states only if we're processing the initial set
        - StartState: include all states reachable from start
        - Star states: we can skip over them directly to their next states
        - Plus states: if they are matched, we can skip over them
        
        Args:
            states: Set of states to follow epsilon transitions from
            is_initial: Whether this is the initial set of states (important for StartAnchor)
            
        Returns:
            A set of all states reachable via epsilon transitions.
        """
        visited = set()
        result = set(states)
        
        to_process = list(states)
        while to_process:
            state = to_process.pop()
            
            if state in visited:
                continue
            
            visited.add(state)
            
            if isinstance(state, StartAnchor):
                if not is_initial:
                    continue
                    
                for next_state in state.next_states:
                    if next_state not in result:
                        result.add(next_state)
                        to_process.append(next_state)
                continue
                    
            if isinstance(state, (StartState, StarState)) or \
               (isinstance(state, PlusState) and state.matched_once):
                for next_state in state.next_states:
                    if next_state not in result:
                        result.add(next_state)
                        to_process.append(next_state)
                    
        return result

    def _check_full_match(self, states):
        """
        Check if we have a complete match (entire string consumed).
        """
        for state in states:
            if isinstance(state, TerminationState):
                return True
                
            if not (isinstance(state, PlusState) and not state.matched_once):
                next_states = self._follow_epsilon({state})
                if any(isinstance(s, TerminationState) for s in next_states):
                    return True
                    
        return False

    def _match_from_start(self, string):
        """
        Match the string from the beginning, enforcing that the entire string must
        be matched and consuming all characters from start to end.
        """
        current_states = self._follow_epsilon({self.start}, is_initial=True)
        
        if not string:
            return self._check_full_match(current_states)
        
        for char in string:
            next_states = set()
            
            for state in current_states:
                if isinstance(state, (TerminationState, StartAnchor)):
                    continue
                
                if state.check_self(char):
                    next_states.update(state.next_states)
                    
                    if isinstance(state, StarState):
                        next_states.add(state)
                    elif isinstance(state, PlusState):
                        state.matched_once = True
                        next_states.add(state)
            
            if not next_states:
                return False
            
            current_states = self._follow_epsilon(next_states, is_initial=False)
            
            if not current_states:
                return False
        
        return self._check_full_match(current_states)

    def _is_fixed_pattern_end(self):
        """Check if the pattern has a fixed ending (not * or +)."""
        if not hasattr(self, 'termination_state'):
            return False

        term_state = self.termination_state
        
        states_to_terminal = []
        def find_states_to_terminal(state, visited):
            if state in visited:
                return
            visited.add(state)
            
            if term_state in state.next_states:
                if not isinstance(state, (StarState, PlusState)):
                    states_to_terminal.append(state)
            
            for next_state in state.next_states:
                find_states_to_terminal(next_state, visited)
        
        find_states_to_terminal(self.start, set())
        
        return len(states_to_terminal) > 0

    def _can_pattern_end_with(self, char):
        """Check if the pattern can end with the given character."""
        if not hasattr(self, 'termination_state'):
            return True

        term_state = self.termination_state
        
        states_to_terminal = []
        def find_states_to_terminal(state, visited):
            if state in visited:
                return
            visited.add(state)
            
            if term_state in state.next_states:
                states_to_terminal.append(state)
            
            for next_state in state.next_states:
                find_states_to_terminal(next_state, visited)
        
        find_states_to_terminal(self.start, set())
        
        for state in states_to_terminal:
            if isinstance(state, (StarState, PlusState)):
                continue
            
            if state.check_self(char):
                return True
        
        return False


if __name__ == "__main__":
    test_patterns = [
        (RegexFSM("a*b"), "ab", True),
        (RegexFSM("a*b"), "b", True),
        (RegexFSM("a*b"), "aab", True),
        (RegexFSM("a*b"), "a", False),
        (RegexFSM("a+b"), "ab", True),
        (RegexFSM("a+b"), "b", False),
        (RegexFSM("a+b"), "aab", True),
        (RegexFSM(""), "", True),
        (RegexFSM("a"), "a", True),
        (RegexFSM("a"), "", False),
        (RegexFSM("[a-z]"), "a", True),
        (RegexFSM("[a-z]"), "A", False),
        (RegexFSM("[a-z]+"), "abc", True),
        (RegexFSM("[a-z0-9]+"), "a1b2c3", True),
    ]
    
    for regex, test_str, expected in test_patterns:
        result = regex.check_string(test_str)
        print(f"'{regex._original_regex}' matches '{test_str}': {result} {'✓' if result == expected else '✗'}")
# Lightweight Regex Engine

A non-deterministic finite automaton (NFA) implementation of a regular expression engine in Python. This engine supports basic regex operations and can match patterns against strings efficiently.

## Author

**Mykhailo Rykhalskyi**

## Features

- **Basic Character Matching:** Match literal characters (e.g., `a`, `b`)
- **Wildcard Operator:** `.` matches any single character
- **Kleene Star Operator:** `*` matches zero or more occurrences of a pattern
- **Plus Operator:** `+` matches one or more occurrences of a pattern
- **Character Classes:** `[a-z]`, `[0-9]`, `[a-zA-Z0-9]` for matching sets of characters
- **Empty String Matching:** Properly handles empty strings and patterns
- **Anchored Matching:** Patterns match only from the beginning of strings

## Implementation Architecture

The engine is designed around a state machine model using a non-deterministic finite automaton (NFA):

### Core Components

1. **State Classes Hierarchy:**
   - `State`: Abstract base class defining the interface for all states
   - `StartState`: Initial state for any regex machine
   - `StartAnchor`: Ensures patterns match from the beginning
   - `TerminationState`: Final accepting state indicating successful matches
   - `AsciiState`: Matches specific characters
   - `DotState`: Matches any character
   - `StarState`: Implements the Kleene star (`*`) operator
   - `PlusState`: Implements the plus (`+`) operator
   - `CharacterClassState`: Manages character classes and ranges

2. **State Transitions:**
   Each state has outgoing transitions to other states. Transitions between states occur when:
   - Characters are consumed (character transitions)
   - No characters are consumed (epsilon transitions)

3. **Epsilon Transitions:**
   Special transitions that don't consume characters, allowing:
   - Skipping over `StarState` states (zero occurrences)
   - Moving from matched `PlusState` states to subsequent states
   - Traversing from anchor states to pattern start

### How the Matching Algorithm Works

The engine uses a subset construction approach to track all possible states during matching:

1. **FSM Construction:**
   - When a regex pattern is parsed, the engine builds an FSM
   - States are created for each element of the pattern
   - Operators (`*`, `+`) transform preceding states
   - Character classes are parsed into sets of valid characters and ranges

2. **String Matching Process:**
   - Begin at the start state with an empty set of current states
   - Follow all epsilon transitions to build the initial state set
   - For each character in the input string:
     - Determine which current states can accept the character
     - For each match, add the next states to a new set
     - Follow all epsilon transitions from those states
     - If no states match, the string is rejected
   - After processing all characters, check if any current state is accepting

3. **Character Class Handling:**
   - Character classes like `[a-z0-9]` are parsed into:
     - Explicit ranges (e.g., `a-z`)
     - Individual characters (e.g., `0-9`)
   - The engine efficiently tests character membership against these sets

4. **Epsilon Transition Handling:**
   The `_follow_epsilon` method handles special cases:
   - Start anchors (only at the beginning)
   - Star states (skippable)
   - Plus states that have matched at least once (skippable)
   - Ensuring we reach all reachable states without consuming input

## Key Algorithms

### Pattern Parsing (`parse_regex`)

Converts a regex string into an interconnected state machine:
```
1. Create a start anchor state
2. For each character/token in the pattern:
   - For operators (*, +), modify the previous state
   - For character classes ([a-z]), create a CharacterClassState
   - For single characters, create appropriate state
3. Connect all states in sequence
4. Add a termination state at the end
```

### String Matching (`check_string` and `_match_from_start`)

Determines whether a string matches the pattern by tracking all possible states:
```
1. Start with the initial state set (following epsilon transitions)
2. For each character in the input string:
   - Find states that accept the character
   - If none, reject the string
   - Otherwise, follow transitions and epsilon transitions
3. After all characters are processed, check if any current state is accepting
4. Return match result
```

### Epsilon Transitions (`_follow_epsilon`)

Computes all states reachable without consuming input:
```
1. Start with a set of current states
2. For each state in the set:
   - If it's a start anchor, star state, or matched plus state
   - Add all its next states to the set
3. Process newly added states until no more are found
4. Return the complete set of reachable states
```

## Usage Examples

```python
from regex import RegexFSM

# Create a regex pattern
pattern = RegexFSM("a[0-9]+")

# Check if strings match
result1 = pattern.check_string("a123")  # True
result2 = pattern.check_string("abc")   # False
result3 = pattern.check_string("123")   # False

# Character class example
pattern = RegexFSM("[a-zA-Z][a-zA-Z0-9]*")

# This matches typical identifiers
result4 = pattern.check_string("var1")   # True
result5 = pattern.check_string("123abc") # False
```

## Limitations

- No support for escape sequences (`\d`, `\w`, etc.)
- No alternation operator (`|`)
- No grouping or capturing with parentheses
- No backreferences or look-around assertions
- No quantifiers beyond `*` and `+` (no `{n,m}`)

## Design Decisions

1. **NFA vs DFA:** Using an NFA implementation allows simpler construction but requires tracking multiple states.

2. **Epsilon Transitions:** The engine uses epsilon transitions to handle operators cleanly.

3. **Start Anchoring:** All patterns must match from the start of the string.

4. **Character Classes:** Character classes are implemented efficiently with range checking.

5. **PlusState Tracking:** Plus states track whether they've matched at least once to distinguish between zero and one-or-more matches.

This implementation balances simplicity and performance while providing a robust subset of regex functionality.
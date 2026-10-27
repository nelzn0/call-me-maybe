"""Constrained decoding state machine that keeps the output valid JSON."""

from enum import Enum, auto
import string


class State(Enum):
    """The steps we go through while writing one function call."""
    F_START = auto()
    F_NAME = auto()
    F_PARAM = auto()
    P_OPEN = auto()
    P_NEXT = auto()
    P_NAME = auto()
    P_COLON = auto()
    P_AFTER_VALUE = auto()
    P_CLOSE = auto()
    STR_OPEN = auto()
    IN_STRING = auto()
    IN_ESCAPE = auto()
    NUM_START = auto()
    NUM_NEGATIVE = auto()
    NUM_INT = auto()
    NUM_DECIMAL_POINT = auto()
    NUM_DECIMAL = auto()
    DONE = auto()


# fixed characters only. digits and names are checked in next_state
TRANSITIONS: dict[State, dict[str, State]] = {
    State.P_OPEN: {'"': State.P_NAME},
    State.P_NEXT: {' ': State.P_NEXT, '"': State.P_NAME},
    # , starts the next parameter, } closes the params
    State.P_AFTER_VALUE: {',': State.P_NEXT, '}': State.P_CLOSE},
    State.P_CLOSE: {'}': State.DONE},
    State.STR_OPEN: {'"': State.IN_STRING, " ": State.STR_OPEN},
    State.IN_STRING: {'"': State.P_AFTER_VALUE},
    State.NUM_START: {'-': State.NUM_NEGATIVE, " ": State.NUM_START},
    # numbers have no closing quote, so , and } end the number here
    State.NUM_INT: {
        '.': State.NUM_DECIMAL_POINT, ',': State.P_NEXT, '}': State.P_CLOSE
    },
    State.NUM_DECIMAL: {',': State.P_NEXT, '}': State.P_CLOSE},
}

escape_chars = '"\\/bfnrt'

allow_char_temp = (
    string.ascii_letters + string.punctuation + ' ' + string.digits
).replace('\\', '')
allowed_in_string = allow_char_temp.replace('"', '')


def next_state(
    state: State,
    char: str,
    generated_so_far: str,
    options: list[str],
    param_type: str | None,
    func_name: str | None = None,
) -> State | None:
    """Works out the next state for one character.

    Checks the character against the current state and returns the state
    it leads to. Returns None if the character isn't allowed here.

    Args:
        state: The state we're in right now
        char: The character we want to add
        generated_so_far: All the text generated before this character
        options: The names we can pick from (function or parameter names)
        param_type: The type of the current parameter, like "string"
        func_name: The chosen function's name, needed once we're on
            its parameters

    Returns:
        The next state, or None if the character breaks the rules
    """

    # the name typed so far, which is the text after the last quote
    last_quote = generated_so_far.rfind('"')
    name_so_far = generated_so_far[last_quote + 1:]

    # get current state
    if char in TRANSITIONS.get(state, {}):
        return TRANSITIONS[state][char]

    if state == State.F_START and '{"name": "' == (generated_so_far + char):
        return State.F_NAME

    if state == State.F_START and (
        '{"name": "'.startswith(generated_so_far + char)
    ):
        return State.F_START

    if state == State.F_NAME and any(
        opts.startswith(name_so_far + char) for opts in options
    ):
        return State.F_NAME

    if state == State.F_NAME and char == '"' and name_so_far in options:
        return State.F_PARAM

    if state == State.F_PARAM:
        # skip past the function name, so we only look at the text after it
        if func_name is not None:
            find_func = generated_so_far.find(func_name + '"')
            after_func_name = generated_so_far[find_func + len(func_name)
                                               + 1:]
        else:
            return None

    if (
        state == State.F_PARAM
        and ', "parameters": {' == (after_func_name + char)
    ):
        return State.P_OPEN

    if state == State.F_PARAM and (
        ', "parameters": {'.startswith(after_func_name + char)
    ):
        return State.F_PARAM

    if (
        state in (State.NUM_START, State.NUM_NEGATIVE, State.NUM_INT)
        and char in string.digits
    ):
        return State.NUM_INT

    if (
        state in (State.NUM_DECIMAL_POINT, State.NUM_DECIMAL)
        and char in string.digits
    ):
        return State.NUM_DECIMAL

    # only this function's parameter name are allowed
    if state == State.P_NAME and any(
        opts.startswith(name_so_far + char) for opts in options
    ):
        return State.P_NAME

    if state == State.P_NAME and char == '"' and name_so_far in options:
        return State.P_COLON

    # the value type comes from the func_def, not the model
    if state == State.P_COLON and char == ':' and param_type == "string":
        return State.STR_OPEN

    if state == State.P_COLON and char == ':' and param_type == "number":
        return State.NUM_START

    # not reachable yet, since backslash isn't in the allowed characters
    if state == State.IN_ESCAPE and char in escape_chars:
        return State.IN_STRING

    if state == State.IN_STRING and char == '\\':
        return State.IN_ESCAPE

    if state == State.IN_STRING and char in allowed_in_string:
        return State.IN_STRING

    return None


def walk_token(
    state: State,
    token_text: str,
    generated_so_far: str,
    options: list[str],
    param_type: str | None,
    func_name: str | None = None,
) -> tuple[State, State] | None:
    """Runs a whole token through next_state, one character at a time.

    Stops as soon as one character is rejected. A colon has to be the
    last character of the token.

    Args:
        state: The state before the token
        token_text: The text of the token
        generated_so_far: All the text generated before this token
        options: The names we can pick from
        param_type: The type of the current parameter
        func_name: The chosen function's name, if we have one

    Returns:
        A tuple of (the state after the token, the state before its
        last character), or None if any character is rejected
    """

    prev_state = state

    for i, c in enumerate(token_text):
        prev_state = state
        new_state = next_state(
            state, c, generated_so_far, options, param_type, func_name
        )
        if new_state is None:
            return None
        state = new_state
        if prev_state == State.P_COLON:
            if c == ':' and i != len(token_text) - 1:
                return None
        generated_so_far += c

    return state, prev_state


def allowed_tokens(
    state: State,
    vocab: dict[str, int],
    generated_so_far: str,
    options: list[str],
    param_type: str | None,
    func_name: str | None = None,
) -> set[int]:
    """Finds every token in the vocab that can come next.

    Runs each token through walk_token and keeps the ones that pass.

    Args:
        state: The state we're in right now
        vocab: Maps token text to token ID
        generated_so_far: All the text generated so far
        options: The names we can pick from
        param_type: The type of the current parameter
        func_name: The chosen function's name, if we have one

    Returns:
        The set of token IDs we're allowed to pick
    """

    valid_tokens = set()

    for text, token_id in vocab.items():
        result = walk_token(
            state, text, generated_so_far, options, param_type, func_name
        )
        if result is not None:
            valid_tokens.add(token_id)

    return valid_tokens


def extract_value_after_pattern(
    text: str,
    pattern: str,
    stop_chars: list[str],
    first: bool = True,
) -> str | None:
    """Gets a value from the text that comes after a pattern.

    Finds where the pattern shows up in the text, then grabs everything
    until it hits one of the stop characters.

    Args:
        text: The JSON text we're working with
        pattern: What we're looking for (like '"name": "')
        stop_chars: Characters that tell us when to stop (like ['"', ',', '}'])
        first: if True, use the first occurrence of the pattern,
            otherwise the last one

    Returns:
        The value we extracted, or None if the pattern isn't in the text.
        If there's no stop character, we return everything up to the end.
    """

    if pattern in text:
        # find where the value starts (right after the pattern)
        # if first, get the first position, otherwise, the last
        if first:
            start_pos = text.find(pattern) + len(pattern)
        else:
            start_pos = text.rfind(pattern) + len(pattern)

        # default to the end of the text if no stop character is found
        # find the earliest stop character position
        end_pos = len(text)
        for c in stop_chars:
            pos = text.find(c, start_pos)
            if pos != -1:
                if pos < end_pos:
                    end_pos = pos

        # extract the value (from start to the first stop character)
        extracted = text[start_pos:end_pos]

        return extracted

    return None


def is_json_complete(text: str) -> bool:
    """Checks if the generated JSON text is complete by counting braces.

    Keeps track of opening and closing braces to see if we've closed all
    the JSON objects we opened. Returns True only if we opened at least one
    brace and closed all of them.

    Args:
        text: The JSON text generated so far

    Returns:
        True if the JSON is complete (all braces are balanced), False otherwise
    """
    depth = 0
    opened = False

    for c in text:
        if '{' == c:
            depth += 1
            opened = True

        elif '}' == c:
            depth -= 1

    # if it was opened, check depth, otherwise always false
    if opened:
        return depth == 0
    else:
        return False

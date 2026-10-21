
from enum import Enum, auto
import string


class State(Enum):
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


TRANSITIONS: dict[State, dict[str, State]] = {
    State.P_OPEN: {'"': State.P_NAME},
    State.P_NEXT: {' ': State.P_NEXT, '"': State.P_NAME},
    State.P_AFTER_VALUE: {',': State.P_NEXT, '}': State.P_CLOSE},
    State.P_CLOSE: {'}': State.DONE},
    State.STR_OPEN: {'"': State.IN_STRING, " ": State.STR_OPEN},
    State.IN_STRING: {'"': State.P_AFTER_VALUE},
    State.NUM_START: {'-': State.NUM_NEGATIVE, " ": State.NUM_START},
    State.NUM_INT: {'.': State.NUM_DECIMAL_POINT, ',': State.P_NEXT, '}': State.P_CLOSE},
    State.NUM_DECIMAL: {',': State.P_NEXT, '}': State.P_CLOSE},
}

escape_chars = '"\\/bfnrt'

allow_char_temp = (string.ascii_letters + string.punctuation + ' ' + string.digits).replace('\\', '')
allowed_in_string = allow_char_temp.replace('"', '')


def next_state(state: State, char: str, generated_so_far: str, options: list[str], param_type: str, func_name: str | None = None) -> State | None:

    last_quote = generated_so_far.rfind('"')
    typed = generated_so_far[last_quote + 1:]

    if char in TRANSITIONS.get(state, {}):
        return TRANSITIONS[state][char]

    if state == State.F_START and '{"name": "' == (generated_so_far + char):
        return State.F_NAME
    if state == State.F_START and '{"name": "'.startswith(generated_so_far + char):
        return State.F_START

    if state == State.F_NAME and any(opts.startswith(typed + char) for opts in options):
        return State.F_NAME
    if state == State.F_NAME and char == '"' and typed in options:
        return State.F_PARAM

    if state == State.F_PARAM:
        if func_name is not None:
            find_func = generated_so_far.find(func_name + '"')
            typed_func = generated_so_far[find_func + len(func_name) + 1:]
        else:
            return None

    if state == State.F_PARAM and ', "parameters": {' == (typed_func + char):
        return State.P_OPEN
    if state == State.F_PARAM and ', "parameters": {'.startswith(typed_func + char):
        return State.F_PARAM

    if state in (State.NUM_START, State.NUM_NEGATIVE, State.NUM_INT) and char in string.digits:
        return State.NUM_INT
    if state in (State.NUM_DECIMAL_POINT, State.NUM_DECIMAL) and char in string.digits:
        return State.NUM_DECIMAL

    if state == State.P_NAME and any(opts.startswith(typed + char) for opts in options):
        return State.P_NAME
    if state == State.P_NAME and char == '"' and typed in options:
        return State.P_COLON

    if state == State.P_COLON and char == ':' and param_type == "string":
        return State.STR_OPEN
    if state == State.P_COLON and char == ':' and param_type == "number":
        return State.NUM_START

    if state == State.IN_ESCAPE and char in escape_chars:
        return State.IN_STRING

    if state == State.IN_STRING and char == '\\':
        return State.IN_ESCAPE

    if state == State.IN_STRING and char in allowed_in_string:
        return State.IN_STRING

    return None


def walk_token(state: State, token_text: str, generated_so_far: str, options: list[str], param_type: str, func_name: str | None = None) -> tuple[State, State] | None:

    prev_state = state

    for i, c in enumerate(token_text):
        prev_state = state
        state = next_state(state, c, generated_so_far, options, param_type, func_name)
        if state is None:
            return None
        if prev_state == State.P_COLON:
            if c == ':' and i != len(token_text) - 1:
                return None
        generated_so_far += c

    return state, prev_state


def allowed_tokens(state: State, vocab: dict[str, int], generated_so_far: str, options: list[str], param_type: str, func_name: str | None = None) -> set[int]:

    valid_tokens = set()

    for text, token_id in vocab.items():
        result = walk_token(state, text, generated_so_far, options, param_type, func_name)
        if result is not None:
            valid_tokens.add(token_id)

    return valid_tokens


def extract_value_after_pattern(text: str, pattern: str, stop_chars: list[str], first: bool = True) -> str | None:
    """Gets a value from the text that comes after a pattern.

    Finds where the pattern shows up in the text, then grabs everything
    until it hits one of the stop characters.

    Args:
        text: The JSON text we're working with
        pattern: What we're looking for (like '"name": "')
        stop_chars: Characters that tell us when to stop (like ['"', ',', '}'])
        first: if True, use the first occurrence of the pattern, otherwise the last one

    Returns:
        The value we extracted, or None if the pattern isn't in the text.
        If there's no stop character, we return everything up to the end.
    """
    extracted = ""

    if pattern in text:
        # find where the value starts (right after the pattern)
        # if first, get the first position, otherwise, the last
        if first:
            start_pos = text.find(pattern) + len(pattern)
        else:
            start_pos = text.rfind(pattern) + len(pattern)

        # start with infinity so any real position will be smaller
        # find the earliest stop character position
        end_pos = float('inf')
        found = False
        for c in stop_chars:
            pos = text.find(c, start_pos)
            if pos != -1:
                found = True
                if pos < end_pos:
                    end_pos = pos

        if not found:
            end_pos = len(text)
            return text[start_pos:end_pos]

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

    # check every character in text
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

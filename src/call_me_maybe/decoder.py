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
    BOOL_OPEN = auto()
    IN_BOOL = auto()
    DONE = auto()


# fixed characters only. digits and names are checked in next_state
TRANSITIONS: dict[State, dict[str, State]] = {
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
    State.BOOL_OPEN: {" ": State.BOOL_OPEN}
}

escape_chars = '"\\/bfnrt'

# letters, punctuation, space and digits.
# the quote is removed from allowed_in_string, since it
# would end the string value early
vocab_chars = string.ascii_letters + string.punctuation + ' ' + string.digits
allowed_in_string = vocab_chars.replace('"', '')


# key is (state, text, options, param_type). each state fills only the slots
# it reads, the others stay None
CACHE_TYPE = dict[tuple[State, str | None,
                        tuple[str, ...] | None, str | None], set[int]]


def find_after_name(generated_so_far: str, func_name: str | None
                    ) -> str | None:
    """Gets the text typed after the function name.

    Only the part after the name is checked for `, "parameters": {`.

    Args:
        generated_so_far: All the text generated so far
        func_name: The chosen function's name, or None if we don't have one yet

    Returns:
        The text after the name,
        or None if there's no name or it isn't in the text
    """
    if func_name is not None:
        name_start = generated_so_far.find(func_name + '"')
        if name_start < 0:
            return None
        after_name_start = generated_so_far[name_start + len(func_name) + 1:]
        return after_name_start
    return None


def text_after_last_quote(generated_so_far: str) -> str:
    """Gets the text typed after the last quote.

    If there's no quote yet, it gives back the whole text.

    Args:
        generated_so_far: All the text generated so far

    Returns:
        The text after the last quote, or the whole text if there isn't one
    """
    last_quote = generated_so_far.rfind('"')
    name_so_far = generated_so_far[last_quote + 1:]

    return name_so_far


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
    name_so_far = text_after_last_quote(generated_so_far)

    # no parameters left, so a comma can't start another one
    if (
        not options and char == ','
        and state in
        (State.P_AFTER_VALUE, State.NUM_INT, State.NUM_DECIMAL)
    ):
        return None

    # parameters still left, so the object can't be closed yet
    if (
            options and char == '}'
            and state in
            (State.P_AFTER_VALUE, State.NUM_INT, State.NUM_DECIMAL)
    ):
        return None

    # zero parameters check
    if (
        not options and char == '}'
        and state == State.P_OPEN
    ):
        return State.P_CLOSE

    if (
        options and char == '"'
        and state == State.P_OPEN
    ):
        return State.P_NAME

    if state == State.NUM_INT and char in (',', '}') and param_type == "number":
        return None
    if state == State.NUM_INT and char == '.' and param_type == "integer":
        return None

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
        after_func_name = find_after_name(generated_so_far, func_name)
        if after_func_name is not None:
            if ', "parameters": {' == (after_func_name + char):
                return State.P_OPEN
            if ', "parameters": {'.startswith(after_func_name + char):
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

    # only this function's parameter names are allowed. A name is removed
    # from options once it's used, so it can't be picked again
    if state == State.P_NAME and any(
        opts.startswith(name_so_far + char) for opts in options
    ):
        return State.P_NAME

    if state == State.P_NAME and char == '"' and name_so_far in options:
        return State.P_COLON

    # the value type comes from the func_def, not the model
    if state == State.P_COLON and char == ':' and param_type == "string":
        return State.STR_OPEN

    if state == State.P_COLON and char == ':' and param_type in ("number", "integer"):
        return State.NUM_START

    if state == State.P_COLON and char == ':' and param_type == "boolean":
        return State.BOOL_OPEN

    if state == State.IN_ESCAPE and char in escape_chars:
        return State.IN_STRING

    if state == State.IN_STRING and char == '\\':
        return State.IN_ESCAPE

    if state == State.IN_STRING and char in allowed_in_string:
        return State.IN_STRING

    # the letters typed after the colon, with the spaces removed
    b_find = name_so_far.rfind(':')

    # true and false are checked letter by letter
    if (
        state == State.BOOL_OPEN
        and 't' == (name_so_far[b_find + 1:].strip() + char)
    ):
        return State.IN_BOOL

    if (
        state == State.BOOL_OPEN
        and 'f' == (name_so_far[b_find + 1:].strip() + char)
    ):
        return State.IN_BOOL

    if (
        state == State.IN_BOOL
        and 'true' == (name_so_far[b_find + 1:].strip() + char)
    ):
        return State.P_AFTER_VALUE

    if (
        state == State.IN_BOOL
        and ('true'.startswith(name_so_far[b_find + 1:].strip() + char))
    ):
        return State.IN_BOOL

    if (
        state == State.IN_BOOL
        and 'false' == (name_so_far[b_find + 1:].strip() + char)
    ):
        return State.P_AFTER_VALUE

    if (
        state == State.IN_BOOL
        and ('false'.startswith(name_so_far[b_find + 1:].strip() + char))
    ):
        return State.IN_BOOL

    return None


def walk_token(
    state: State,
    token_text: str,
    generated_so_far: str,
    options: list[str],
    param_type: str | None,
    func_name: str | None = None,
) -> State | None:
    """Runs a whole token through next_state, one character at a time.

    Stops as soon as one character is rejected. A colon has to be the
    last character of the token, and so does the closing quote of a
    parameter name.

    Args:
        state: The state before the token
        token_text: The text of the token
        generated_so_far: All the text generated before this token
        options: The names we can pick from
        param_type: The type of the current parameter
        func_name: The chosen function's name, if we have one

    Returns:
        The state after the token,
        or None if any character is rejected
    """

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
        if (
            prev_state == State.P_NAME
            and new_state == State.P_COLON
            and i != len(token_text) - 1
        ):
            return None

        generated_so_far += c

    return state


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


def get_allowed_tokens(
        cache: CACHE_TYPE,
        state: State,
        vocab: dict[str, int],
        generated_so_far: str,
        options: list[str],
        param_type: str | None,
        func_name: str | None = None
) -> set[int]:
    """Same as allowed_tokens, but
    reuses the answer if the key was seen before.

    The key only holds the inputs this state reads, so the same key always
    gives the same set of tokens. States whose tokens can run into a
    parameter name also keep the remaining names in the key.

    Args:
        cache: Saved results, keyed by state and the inputs it reads
        state: The state we're in right now
        vocab: Maps token text to token ID
        generated_so_far: All the text generated so far
        options: The names we can pick from
        param_type: The type of the current parameter
        func_name: The chosen function's name, if we have one

    Returns:
        The set of token IDs we're allowed to pick
    """
    text = None
    key_options = None
    key_type = None

    # states that don't read these inputs leave them as None
    if state == State.F_START:
        text = generated_so_far
    elif state in (State.F_NAME, State.P_NAME):
        text = text_after_last_quote(generated_so_far)
        key_options = tuple(options)
    elif state == State.F_PARAM:
        text = find_after_name(generated_so_far, func_name)
        key_options = tuple(options)
    elif state == State.P_COLON:
        key_type = param_type
    elif state == State.IN_BOOL:
        b_find = generated_so_far.rfind(':')
        text = generated_so_far[b_find + 1:].strip()
    elif state in (State.NUM_START, State.NUM_NEGATIVE, State.NUM_DECIMAL_POINT):
        key_type = param_type
    elif state in (State.P_AFTER_VALUE, State.NUM_INT, State.NUM_DECIMAL):
        key_options = tuple(options)
        key_type = param_type
    elif state in (State.IN_STRING, State.P_NEXT, State.P_OPEN):
        key_options = tuple(options)

    key = (state, text, key_options, key_type)
    if key in cache:
        return cache[key]
    result = allowed_tokens(state, vocab, generated_so_far,
                            options, param_type, func_name
                            )
    cache[key] = result
    return result


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

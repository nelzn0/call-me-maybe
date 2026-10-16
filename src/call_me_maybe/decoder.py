
from llm_sdk import Small_LLM_Model
from call_me_maybe.models import FunctionDefinition
from enum import Enum, auto
from string import digits


class State(Enum):
    P_OPEN = auto()
    P_NEXT = auto()
    P_NAME = auto()
    P_COLON = auto()
    P_AFTER_VALUE = auto()
    P_CLOSE = auto()
    STR_OPEN = auto()
    IN_STRING = auto()
    NUM_START = auto()
    NUM_NEGATIVE = auto()
    NUM_INT = auto()
    NUM_DECIMAL_POINT = auto()
    NUM_DECIMAL = auto()
    DONE = auto()


TRANSITIONS: dict[State, dict[str, State]] = {
    State.P_OPEN: {'"': State.P_NAME},
    State.P_NEXT: {' ': State.P_NEXT, '"': State.P_NAME},
    State.P_NAME: {'"': State.P_COLON},
    State.P_COLON: {},
    State.P_AFTER_VALUE: {',': State.P_NEXT, '}': State.P_CLOSE},
    State.P_CLOSE: {'}': State.DONE},
    State.STR_OPEN: {'"': State.IN_STRING, " ": State.STR_OPEN},
    State.IN_STRING: {'"': State.P_AFTER_VALUE},
    State.NUM_START: {'-': State.NUM_NEGATIVE},
    State.NUM_NEGATIVE: {},
    State.NUM_INT: {'.': State.NUM_DECIMAL_POINT, ',': State.P_NEXT, '}': State.P_CLOSE},
    State.NUM_DECIMAL_POINT: {},
    State.NUM_DECIMAL: {',': State.P_NEXT, '}': State.P_CLOSE},
}


def next_state(state: State, char: str, generated_so_far: str, options: list[str]) -> State | None:

    if char in TRANSITIONS.get(state, {}):
        return TRANSITIONS[state][char]

    if state in (State.NUM_START, State.NUM_NEGATIVE, State.NUM_INT) and char in digits:
        return State.NUM_INT
    if state in (State.NUM_DECIMAL_POINT, State.NUM_DECIMAL) and char in digits:
        return State.NUM_DECIMAL

    if state == State.P_NAME and any(char for opts in options if opts.startswith(generated_so_far + char)):
        return State.P_NAME
    if state == State.P_NAME and char == '"' and generated_so_far in options:
        return State.P_COLON

    return None


def walk_token(state: State, token_text: str) -> State | None:
    if state is None:
        return None


def allowed_tokens(state: State, vocab: dict[str, int]) -> set[int]:
    pass


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


def get_valid_next_tokens(generated_so_far: str, options: list[str], llm_model: Small_LLM_Model) -> list[int]:
    """Gets the valid next tokens by checking what was generated so far,
    matching it with a list of choices, and converting them to token IDs.

    Args:
        generated_so_far: The partial text generated so far (like "fn_gr").
        options: A list of valid options to match against (like function names
                or parameter names). (example: "fn_gr" would match "fn_greet")
        llm_model: The LLM model used to encode characters into token IDs.

    Returns:
        A list of valid token IDs that can come next.
    """

    next_chars = []
    next_tokens = []

    # find which options match what's been generated so far
    matches = [option for option in options if option.startswith(generated_so_far)]

    # for each matching option, get the next character that should come
    for match in matches:
        if len(match) > len(generated_so_far):
            next_char = match[len(generated_so_far)]
            next_chars.append(next_char)
        else:
            # if its finished, add the closing '"'
            next_chars.append('"')

    # remove duplicate characters
    # (in case multiple options have same next char)
    next_chars = list(set(next_chars))

    # encode each valid character into its token IDs
    for c in next_chars:
        token_id = llm_model.encode(c)
        next_tokens += token_id[0].tolist()

    return next_tokens


def is_escaped_quote(text: str, pos: int) -> bool:
    """Checks if the quote at pos is escaped by an odd number of backslashes.

    Walks left from pos and flips a flag for each backslash.

    Args:
        text: The JSON text we're working with
        pos: The index of the quote we want to check

    Returns:
        True if the quote is escaped, False otherwise
    """
    is_backslash = False
    i = pos - 1

    while i >= 0 and text[i] == "\\":
        is_backslash = not is_backslash
        i -= 1

    return is_backslash


def is_in_string(text: str) -> bool:
    """Checks if the end of the text is inside a string.

    Counts the quotes that aren't escaped. An odd count means the last
    quote opened a string.

    Args:
        text: The JSON text generated so far

    Returns:
        True if we're inside a string, False otherwise
    """
    search_pos = 0
    quote_finder = text.find('"')
    quote_count = 0

    while quote_finder != -1:
        # escaped quotes don't open or close a string
        if not is_escaped_quote(text, quote_finder):
            quote_count += 1

        search_pos = quote_finder + 1
        quote_finder = text.find('"', search_pos)

    return quote_count % 2 > 0


def get_valid_tokens_for_current_state(text: str, func_defs: list[FunctionDefinition], llm_model: Small_LLM_Model, vocab: set[int]) -> set[int] | list[int] | None:
    """Figures out what tokens are valid right now based on where we are in the JSON.

    First we constrain the function name, then the parameter names for the
    chosen function, then the parameter values. Numbers and strings get their
    own rules, and string contents use the precomputed vocab list.

    Args:
        text: The JSON text we've generated so far
        func_defs: All the function definitions we can choose from
        llm_model: The LLM model we use to encode chars into token IDs
        vocab: The token IDs whose text only uses allowed characters

    Returns:
        A set (or list) of valid token IDs for the current state, or None if
        the chosen function isn't in func_defs.
    """

    next_tokens = []

    # get all available function names
    func_names = [func.name for func in func_defs]

    # extract the function name being generated (or the complete one)
    function_extract = extract_value_after_pattern(text, '{"name": "', ['"'])

    # STATE 0: nothing generated yet, force the output to begin with {"name": "
    if function_extract is None:
        return get_valid_next_tokens(text, ['{"name": "'], llm_model)

    # find where the function name value starts
    name_start_pos = text.find('{"name": "')
    name_value_start_pos = name_start_pos + len('{"name": "')

    # check if the function name is complete (has closing quote)
    name_finished = text.find('"', name_value_start_pos)

    if name_finished != -1:
        # STATE 2: function name is done, now constrain parameter names

        # find which function was chosen
        chosen_func = None
        for func in func_defs:
            if func.name == function_extract:
                chosen_func = func
                break

        if not chosen_func:
            return None

        else:
            # inside the parameters object: names, number values and string values
            if '"parameters": {"' in text:
                # get the parameter names for the chosen function
                param_names = list(chosen_func.parameters.keys())

                # later parameters start with ', "', the first one with '"parameters": {"'
                start = text.index('"parameters": {"') + len('"parameters": {"')

                if ',' in text[start:]:
                    param_pattern = ', "'
                else:
                    param_pattern = '"parameters": {"'

                # extract the parameter name being generated
                # False = last occurrence, since the parameter being typed is the newest one
                param_extract = extract_value_after_pattern(text, param_pattern, ['"'], False)

                # quote_pos is the last quote; outside a string we skip the closing quote
                if is_in_string(text):
                    quote_pos = text.rfind('"')
                else:
                    quote_pos = text.rfind('"', 0, -1)

                if text.endswith('"') and param_extract != "" and not is_in_string(text) and quote_pos != -1 and text[quote_pos - 2] != ":":
                    next_tokens = llm_model.encode(':')[0].tolist()

                # after the ':': a string opens with '"', a number starts with a digit or '-'
                elif (text.endswith(':') or text.endswith(': ')) and not is_in_string(text):
                    if chosen_func.parameters[param_extract].type == "string":
                        if text.endswith(':'):
                            next_tokens.extend(llm_model.encode(' ')[0].tolist())
                            next_tokens.extend(llm_model.encode('"')[0].tolist())
                        else:
                            next_tokens.extend(llm_model.encode('"')[0].tolist())

                    elif chosen_func.parameters[param_extract].type == "number":
                        for n in "0123456789-":
                            next_tokens.extend(llm_model.encode(n)[0].tolist())

                # inside a string: the precomputed vocab list (all tokens whose text is allowed)
                # ':' two back means this is a value, not a parameter name
                elif is_in_string(text) and quote_pos != -1 and text[quote_pos - 2] == ":":
                    next_tokens = vocab

                # inside a number (outside a string): more digits, or ',' or } to end it
                elif text[-1].isdigit():
                    for n in "0123456789,}":
                        next_tokens.extend(llm_model.encode(n)[0].tolist())

                # after the comma (outside a string): allow the next parameter name to start
                elif text.endswith(",") and not is_in_string(text):
                    next_tokens.extend(llm_model.encode(' ')[0].tolist())
                    next_tokens.extend(llm_model.encode(' "')[0].tolist())
                    next_tokens.extend(llm_model.encode('"')[0].tolist())
                elif text.endswith(", ") and not is_in_string(text):
                    next_tokens = llm_model.encode('"')[0].tolist()
                # parameters object just closed: } is the only valid token
                elif text.endswith("}") and not is_in_string(text):
                    next_tokens = llm_model.encode('}')[0].tolist()
                # value just closed: ',' or '}' next
                elif text.endswith('"') and not is_in_string(text) and quote_pos != -1 and text[quote_pos - 2] == ":":
                    next_tokens.extend(llm_model.encode(',')[0].tolist())
                    next_tokens.extend(llm_model.encode('}')[0].tolist())
                else:
                    # keep only the valid parameter names
                    next_tokens = get_valid_next_tokens(param_extract, param_names, llm_model)

            # object just opened, quote next
            elif text.endswith('{'):
                next_tokens = llm_model.encode('"')[0].tolist()

            # STATE 2b: the parameters key is being typed, constrain it to "parameters"
            elif ', "' in text or ',"' in text:

                # the key is closed: allow the colon
                if text.endswith('"parameters"'):
                    next_tokens = llm_model.encode(':')[0].tolist()
                # after the colon: allow a space or the opening brace
                elif text.endswith('"parameters":'):
                    next_tokens.extend(llm_model.encode(' ')[0].tolist())
                    next_tokens.extend(llm_model.encode('{')[0].tolist())
                elif text.endswith('"parameters": '):
                    next_tokens = llm_model.encode('{')[0].tolist()
                else:

                    param_key_extract = extract_value_after_pattern(text, ', "', ['"'])

                    if param_key_extract is None:
                        param_key_extract = extract_value_after_pattern(text, ',"', ['"'])

                    next_tokens = get_valid_next_tokens(param_key_extract, ["parameters"], llm_model)

            else:
                # name done, outside parameters: comma is the only valid token
                if text.endswith(","):
                    next_tokens.extend(llm_model.encode(' ')[0].tolist())
                    next_tokens.extend(llm_model.encode(' "')[0].tolist())
                    next_tokens.extend(llm_model.encode('"')[0].tolist())
                elif text.endswith(", "):
                    next_tokens = llm_model.encode('"')[0].tolist()
                else:
                    next_tokens = llm_model.encode(',')[0].tolist()

    else:
        # STATE 1: still generating function name

        next_tokens = get_valid_next_tokens(function_extract, func_names, llm_model)

    return next_tokens


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


from llm_sdk import Small_LLM_Model
from call_me_maybe.models import FunctionDefinition, TestPrompt


def extract_value_after_pattern(text: str, pattern: str, stop_chars: list[str]) -> str | None:
    """Gets a value from the text that comes after a pattern.

    Finds where the pattern shows up in the text, then grabs everything
    until it hits one of the stop characters.

    Args:
        text: The JSON text we're working with
        pattern: What we're looking for (like '"name": "')
        stop_chars: Characters that tell us when to stop (like ['"', ',', '}'])

    Returns:
        The value we extracted if we found the pattern, or None if we couldn't
        find it or hit a stop character.
    """
    extracted = ""

    if pattern in text:
        # find where the value starts (right after the pattern)
        start_pos = text.index(pattern) + len(pattern)

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
            return None

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


def get_valid_tokens_for_current_state(text: str, func_defs: list[FunctionDefinition], llm_model: Small_LLM_Model) -> list[int] | None:
    """Figures out what tokens are valid right now based on where we are in the JSON.

    Works like a state machine - first we constrain the function name, then once that's
    done, we switch to constraining the parameter names for whatever function was chosen.

    Args:
        text: The JSON text we've generated so far
        func_defs: All the function definitions we can choose from
        llm_model: The LLM model we use to encode chars into token IDs

    Returns:
        A list of valid token IDs for the current state, or None if we haven't
        started generating anything yet or something's missing.
    """

    next_tokens = []

    # get all available function names
    func_names = [func.name for func in func_defs]

    # extract the function name being generated (or the complete one)
    function_extract = extract_value_after_pattern(text, '{"name": "', ['"'])

    # if function name hasn't started yet, return None
    if function_extract is None:
        return None

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
            # get the parameter names for the chosen function
            param_names = list(chosen_func.parameters.keys())

            # extract the parameter name being generated
            param_extract = extract_value_after_pattern(text, '"parameters": {', ['},'])

            if param_extract is None:
                return None

            # constrain to valid parameter names
            next_tokens = get_valid_next_tokens(param_extract, param_names, llm_model)

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

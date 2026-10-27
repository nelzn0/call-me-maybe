import json
import sys
import numpy as np
from call_me_maybe.models import FunctionDefinition, TestPrompt
from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model
import argparse
import os
from typing import Any
from collections.abc import Callable
from call_me_maybe.decoder import (
    State, allowed_tokens, walk_token,
    extract_value_after_pattern, is_json_complete, allow_char_temp
)


MAX_TOKENS = 200


def load_json_data[T: BaseModel](
    file_path: str, model_class: type[T]
) -> list[T]:
    """Loads JSON data from a file and validates it against a pydantic model.

    Reads the JSON file, checks if it's valid, and makes sure each item matches
    the expected format for the given model. If something's wrong with the file
    or any item, it raises an error with details about what went wrong.

    Args:
        file_path: Path to the JSON file to load
        model_class: The pydantic model class to validate each item against

    Returns:
        A list of validated model objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or any item doesn't match the model
    """

    json_data: list[T] = []
    data_errors: list[str] = []

    # opening and validating the json file
    try:
        with open(file_path) as f:
            try:
                out = json.load(f)
                for i in out:
                    try:
                        data = model_class.model_validate(i)
                        json_data.append(data)
                    except ValidationError as e:
                        data_errors.append(f"Error: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Error: Invalid JSON in file {file_path}: {e}"
                )
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {file_path}")

    # if there's errors
    if data_errors:
        raise ValueError('\n'.join(data_errors))
    else:
        return json_data


def get_vocabulary(
    allowed_chars: str, llm_model: Small_LLM_Model
) -> dict[str, int]:
    """Builds a dict of token text to token ID.

    Reads the vocabulary file from the model, turns the Ġ space marker into
    a real space, and keeps every token where all characters are allowed.

    Args:
        allowed_chars: The characters a token's text is allowed to use
        llm_model: The LLM model, used to find the vocabulary file

    Returns:
        The token text as the key, and its ID as the value
    """

    vocab = llm_model.get_path_to_vocab_file()

    with open(vocab) as f:

        vocab_dict = json.load(f)

        valid_chars = {}

        for k in vocab_dict:
            converted_text = k.replace("Ġ", " ")
            valid = True
            for c in converted_text:
                if c not in allowed_chars:
                    valid = False
            if valid:
                valid_chars[converted_text] = vocab_dict[k]

    return valid_chars


def find_function(
    func_defs: list[FunctionDefinition], text: str
) -> FunctionDefinition | None:
    """Finds the function definition with the given name.

    Args:
        func_defs: All the function definitions we loaded
        text: The function name to look for

    Returns:
        The matching definition, or None if no function has that name
    """

    for f in func_defs:
        if f.name == text:
            return f
    return None


def build_functions_text(func_defs: list[FunctionDefinition]) -> str:
    """Turns the function definitions into text for the prompt.

    Each function gets its name, description, and parameter types, so the
    model knows what it can pick from.

    Args:
        func_defs: The function definitions to list

    Returns:
        The text block with every function in it
    """

    functions = ""
    for f in func_defs:
        para = {}
        for param_name, param_type in f.parameters.items():
            para[param_name] = param_type.type

        functions += f"""

Function name: {f.name}
Description: {f.description}
Parameters: {json.dumps(para)}

"""

    return functions


def build_prompt(functions_text: str, user_prompt: str) -> str:
    """Puts the function list and the user's question into the prompt.

    Args:
        functions_text: The text block from build_functions_text
        user_prompt: The question the user asked

    Returns:
        The full prompt we send to the model
    """

    prompt = f"""You have access to the following functions:

Available functions:

{functions_text}

Given the user's question, respond with a JSON object specifying which
function to call and with what parameters.

User question: {user_prompt}

Respond ONLY with a JSON object in this format using the appropriate \
function to the user's question:
{{"name": "function_name", "parameters": {{...}}}}

Response: """

    return prompt


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """Flips the vocab so we can look up text from an ID.

    Args:
        vocab: Maps token text to token ID

    Returns:
        Maps token ID back to token text
    """

    vocab_invert = {token_id: text for text, token_id in vocab.items()}

    return vocab_invert


def mask_logits(logits: list[float], valid_tokens: set[int]) -> list[float]:
    """Sets the logit of every token we can't use to -inf.

    That way argmax can only pick a valid token.

    Args:
        logits: The scores from the model, one per token
        valid_tokens: The IDs we're allowed to pick

    Returns:
        The same logits list, with the invalid ones set to -inf
    """

    for j in range(len(logits)):
        if j not in valid_tokens:
            logits[j] = -float('inf')

    return logits


def write_result(path: str, results: list[dict[str, Any]]) -> None:
    """Writes the results to a JSON file.

    Makes the folder first if it doesn't exist yet.

    Args:
        path: Where to save the file
        results: The entries to write
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def pick_token(
    logits: list[float],
    valid_tokens: set[int],
    vocab_invert: dict[int, str],
) -> tuple[int, str]:
    """Takes the logits and valid tokens, masks the invalid logits
    with mask_logits and picks the highest one

    Args:
        logits: the list of logits
        valid_tokens: the valid tokens so far
        vocab_invert: the inverted vocabulary

    Returns:
        the token ID and text
    """

    logits = mask_logits(logits, valid_tokens)

    best_token_id = int(np.argmax(logits))

    chosen_text = vocab_invert[best_token_id]

    return best_token_id, chosen_text


def capture_function_name(
    generated_text: str,
    chosen_text: str,
    func_defs: list[FunctionDefinition],
) -> FunctionDefinition | None:
    """Gets the function name from the text and finds its definition.

    Args:
        generated_text: The text generated before this token
        chosen_text: The token we just picked
        func_defs: The function definitions to search

    Returns:
        The matching function, or None if we can't find the name
    """

    func_name = extract_value_after_pattern(
        generated_text + chosen_text, '{"name": "', ['"']
    )
    if func_name is None:
        return None

    chosen_func = find_function(func_defs, func_name)
    return chosen_func


def capture_param_name(
    generated_text: str,
    chosen_text: str,
    name_start_state: State | None,
) -> str | None:
    """Gets the name of the parameter we're on right now.

    The first parameter comes after '"parameters": {"'. Later ones come
    after ', "', so we take the last one.

    Args:
        generated_text: The text generated before this token
        chosen_text: The token we just picked
        name_start_state: The state where the parameter name started

    Returns:
        The parameter name, or None if we can't find it
    """

    if name_start_state == State.P_OPEN:
        param_name = extract_value_after_pattern(
            generated_text + chosen_text, '"parameters": {"', ['"']
        )
    else:
        param_name = extract_value_after_pattern(
            generated_text + chosen_text, ', "', ['"'], False
        )

    return param_name


def generate_call(
    llm_model: Small_LLM_Model,
    vocab: dict[str, int],
    vocab_invert: dict[int, str],
    func_defs: list[FunctionDefinition],
    prompt: str,
) -> dict[str, Any]:
    """Runs the token loop for one prompt and returns the parsed JSON.

    Args:
        llm_model: The model we're generating with
        vocab: Maps token text to token ID
        vocab_invert: Maps token ID back to token text
        func_defs: The function definitions to pick from
        prompt: The full prompt, from build_prompt

    Returns:
        The JSON object the model built, as a dict

    Raises:
        ValueError: If no valid token is left, the state doesn't match,
            or the JSON isn't finished after MAX_TOKENS
    """

    encoded = llm_model.encode(prompt)
    input_ids = encoded[0].tolist()
    options = [func.name for func in func_defs]
    output_ids: list[int] = []
    chosen_func = None
    func_name = None
    state = State.F_START
    param_type = None
    name_start_state = None

    for _ in range(MAX_TOKENS):
        full_context = input_ids + output_ids
        logits = llm_model.get_logits_from_input_ids(full_context)

        generated_text = llm_model.decode(output_ids)

        # valid tokens depend on the JSON generated so far
        valid_tokens = allowed_tokens(
            state, vocab, generated_text, options, param_type, func_name
        )

        if not valid_tokens:
            raise ValueError("No valid tokens found")

        best_token_id, chosen_text = pick_token(
            logits, valid_tokens, vocab_invert
        )

        walk_result = walk_token(
            state, chosen_text, generated_text, options, param_type, func_name
        )
        if walk_result is not None:
            new_state, prev_state = walk_result
            # remember where the parameter name starts,
            # capture_param_name needs it
            if new_state == State.P_NAME and state != State.P_NAME:
                name_start_state = prev_state
            # the function name is done, so look up its definition
            # and its parameters
            if state == State.F_NAME and new_state == State.F_PARAM:
                chosen_func = capture_function_name(
                    generated_text, chosen_text, func_defs
                )
                if chosen_func is None:
                    raise ValueError("Unable to find chosen function")
                func_name = chosen_func.name
                options = list(chosen_func.parameters.keys())
            # the parameter name is done, so look up its type
            # in the functions definition
            if state == State.P_NAME and new_state == State.P_COLON:
                param_name = capture_param_name(
                    generated_text, chosen_text, name_start_state
                )
                if param_name is None:
                    raise ValueError(
                        "Unable to find chosen function parameter"
                    )
                if chosen_func is None:
                    raise ValueError("Unable to find chosen function")
                param_type = chosen_func.parameters[param_name].type
            state = new_state
        else:
            raise ValueError("No new states found")

        output_ids.append(best_token_id)

        generated_text = llm_model.decode(output_ids)

        # stop once the braces are balanced
        if is_json_complete(generated_text):
            parsed_result: dict[str, Any] = json.loads(generated_text)
            return parsed_result

    raise ValueError("Failed to generate call")


def generate(
    args: argparse.Namespace,
    on_result: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Generates a function call for each prompt and writes them to a file.

    Loads the functions and prompts, builds the vocabulary once, then runs
    generate_call for each prompt. A prompt that fails is printed and skipped.

    Args:
        args: The parsed command-line arguments (file paths)
        on_result: Optional function called after each successful prompt
            with its position, the total number of prompts, and the result,
            for the CLI

    Returns:
        The results that were written to the output file
    """

    func_defs = load_json_data(args.functions_definition, FunctionDefinition)

    inputs = load_json_data(args.input, TestPrompt)

    try:
        llm_model = Small_LLM_Model()

        # builds the vocab list once, before the prompt loop
        vocab = get_vocabulary(allow_char_temp, llm_model)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Could not load the model: {e}")

    vocab_invert = invert_vocab(vocab)

    functions = build_functions_text(func_defs)

    final_result = []
    total = len(inputs)

    for index, current_input in enumerate(inputs, start=1):
        # one bad prompt shouldn't stop the rest
        try:
            prompt = build_prompt(functions, current_input.prompt)
            entry = {}
            # prompt goes first so it's the first key in the output
            entry["prompt"] = current_input.prompt
            parsed_result = generate_call(
                llm_model, vocab, vocab_invert, func_defs, prompt
            )
            entry.update(parsed_result)
            final_result.append(entry)
            # the caller decides how to show each result
            if on_result is not None:
                print("")
                on_result(index, total, entry)
        except ValueError as e:
            print(f"Prompt: '{current_input.prompt}' - Failed because {e}",
                  file=sys.stderr)

    write_result(args.output, final_result)
    return final_result

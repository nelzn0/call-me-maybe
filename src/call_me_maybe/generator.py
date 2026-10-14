import json
import numpy as np
from call_me_maybe.models import FunctionDefinition, TestPrompt
from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model
from call_me_maybe.decoder import get_valid_next_tokens, get_valid_tokens_for_current_state, extract_value_after_pattern, is_json_complete
import argparse
import os
import string


def load_json_data(file_path: str, model_class: type[BaseModel]) -> list[BaseModel]:
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

    json_data = []
    data_errors = []

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
                raise ValueError(f"Error: Invalid JSON in file {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {file_path}")

    # if there's errors
    if data_errors:
        raise ValueError('\n'.join(data_errors))
    else:
        return json_data


def get_vocabulary(allowed_chars: str, llm_model: Small_LLM_Model) -> list[int]:
    """Builds the list of token IDs whose text only uses allowed characters.

    Reads the vocabulary file from the model, turns the Ġ space marker into
    a real space, and keeps every token where all characters are allowed.

    Args:
        allowed_chars: The characters a token's text is allowed to use
        llm_model: The LLM model, used to find the vocabulary file

    Returns:
        A list of token IDs that can be used inside a string value
    """

    vocab = llm_model.get_path_to_vocab_file()

    with open(vocab) as f:

        vocab_dict = json.load(f)

        valid_chars = []

        for k in vocab_dict:
            converted_text = k.replace("Ġ", " ")
            valid = True
            for c in converted_text:
                if c not in allowed_chars:
                    valid = False
            if valid:
                valid_chars.append(vocab_dict[k])

    return valid_chars


def generate(args: argparse.Namespace) -> None:
    """Generates a function call for each prompt and writes the results to a file.

    Builds the vocabulary once, then for each prompt masks the logits so only
    valid tokens can be picked, until the JSON object is complete.

    Args:
        args: The parsed command-line arguments (file paths)
    """
    func_defs = load_json_data(args.functions_definition, FunctionDefinition)

    inputs = load_json_data(args.input, TestPrompt)

    llm_model = Small_LLM_Model()

    allowed_chars = string.ascii_letters + string.punctuation + ' ' + string.digits
    allow_char_temp = allowed_chars.replace('\\', '')

    # builds the vocab list once, before the prompt loop
    vocab = set(get_vocabulary(allow_char_temp, llm_model))

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

    final_result = []

    for current_input in inputs:

        prompt = f"""You have access to the following function:

        Available functions:

        {functions}

        Given the user's question, respond with a JSON object specifying which
        function to call and with what parameters.

        User question: {current_input.prompt}

        Respond ONLY with a JSON object in this format:
        {{"name": "function_name", "parameters": {{...}}}}

        Response: """

        print("starting generation")

        encoded = llm_model.encode(prompt)
        input_ids = encoded[0].tolist()
        output_ids = []
        max_tokens = 200

        for i in range(max_tokens):
            full_context = input_ids + output_ids
            logits = llm_model.get_logits_from_input_ids(full_context)

            generated_text = llm_model.decode(output_ids)

            # valid tokens depend on the JSON generated so far
            valid_tokens = get_valid_tokens_for_current_state(generated_text, func_defs, llm_model, vocab)

            if valid_tokens is not None:
                # block invalid tokens
                for j in range(len(logits)):
                    if j not in valid_tokens:
                        logits[j] = -float('inf')

            best_token_id = np.argmax(logits)
            output_ids.append(best_token_id)

            generated_text = llm_model.decode(output_ids)
            print(generated_text)

            if is_json_complete(generated_text):
                entry = {}
                entry["prompt"] = current_input.prompt
                parsed_result = json.loads(generated_text)
                entry.update(parsed_result)
                final_result.append(entry)
                break

    if os.path.dirname(args.output) != "":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(final_result, f, indent=2)

    print("FInal result")
    print(final_result)

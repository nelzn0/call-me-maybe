from llm_sdk import Small_LLM_Model
from call_me_maybe.models import FunctionDefinition, TestPrompt
import argparse
import json
from pydantic import BaseModel, ValidationError
import numpy as np


def load_functions(file_path: str) -> list[FunctionDefinition]:
    functions = []
    function_errors = []
    try:
        with open(file_path) as f:
            try:
                out = json.load(f)
                for i in out:
                    try:
                        function = FunctionDefinition.model_validate(i)
                        functions.append(function)
                    except ValidationError as e:
                        function_errors.append(f"Error: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error: Invalid JSON in file {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {file_path}")

    if function_errors:
        raise ValueError('\n'.join(function_errors))
    else:
        return functions


def load_input(file_path: str) -> list[TestPrompt]:
    prompts = []
    prompt_errors = []
    try:
        with open(file_path) as f:
            try:
                out = json.load(f)
                for i in out:
                    try:
                        prompt = TestPrompt.model_validate(i)
                        prompts.append(prompt)
                    except ValidationError as e:
                        prompt_errors.append(f"Error: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error: Invalid JSON in file {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {file_path}")

    if prompt_errors:
        raise ValueError('\n'.join(prompt_errors))
    else:
        return prompts


def is_json_complete(text: str) -> bool:
    depth = 0
    opened = False
    for c in text:
        if '{' == c:
            depth += 1
            opened = True

        elif '}' == c:
            depth -= 1

    if opened:
        return depth == 0
    else:
        return False


def get_valid_tokens_for_current_state(text: str, func_defs: list[FunctionDefinition], llm_model: Small_LLM_Model) -> list[int] | None:

    next_tokens = []

    # 1: Check if we're currently generating a function name,
    # generate if text contains '{"name": '"
    if '{"name": "' in text:
        # 2: Extract the partial function name, find where it starts after name:
        start = text.index('{"name": "') + len('{"name": "')

        # Get everything after the start position, partial name generated so far
        current_name = text[start:]

        if '"' in current_name:
            return None

        # list of all function names in func_defs objects
        function_names = [func.name for func in func_defs]

        # find matches
        matches = [name for name in function_names if name.startswith(current_name)]

        # 4: Find the next valid character for each match
        # for each match, look at character that comes right after current_name

        next_chars = []

        for match in matches:
            if len(match) > len(current_name):
                # is there a char after the current position
                next_char = match[len(current_name)]
                next_chars.append(next_char)
            else:
                # if its finished, add the closing '"'
                next_chars.append('"')

        # remove duplicates
        next_chars = list(set(next_chars))

        # 5: convert chars to token IDs
        for c in next_chars:
            token_id = llm_model.encode(c)
            next_tokens += token_id[0].tolist()

        return next_tokens

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Constraining")

    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")

    args = parser.parse_args()

    func_defs = load_functions(args.functions_definition)

    inputs = load_input(args.input)

    llm_model = Small_LLM_Model()

    para = {}
    functions = ""
    for f in func_defs:
        for param_name, param_type in f.parameters.items():
            para[param_name] = param_type.type

        functions += f"""

        Function name: {f.name}
        Description: {f.description}
        Parameters: {json.dumps(para)}

        """

    input1 = inputs[1]

    prompt = f"""You have access to the following function:

    Avaliable functions:

    {functions}

    Given the user's question, respond with a JSON object specifying which
    function to call an with what parameters.

    User question: {input1.prompt}

    Respond ONLY with a JSON object in this format:
    {{"name": "function_name", "parameters": {{...}}}}

    Response: """

    print("starting generation")

    encoded = llm_model.encode(prompt)
    input_ids = encoded[0].tolist()
    output_ids = []
    max_tokens = 30

    for i in range(max_tokens):

        print(f"iteration {i}")

        full_context = input_ids + output_ids
        logits = llm_model.get_logits_from_input_ids(full_context)

        generated_text = llm_model.decode(output_ids)

        print(f"Generated text: {generated_text}")

        valid_tokens = get_valid_tokens_for_current_state(generated_text, func_defs, llm_model)

        print(f"Valid tokens: {valid_tokens}")

        if valid_tokens is not None:
            # block invalid tokens
            for j in range(len(logits)):
                if j not in valid_tokens:
                    logits[j] = -float('inf')

        best_token_id = np.argmax(logits)
        output_ids.append(best_token_id)

        if is_json_complete(generated_text):
            break

    print("FInal result")
    print(generated_text)


if __name__ == "__main__":
    main()

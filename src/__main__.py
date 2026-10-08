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


def get_valid_tokens_for_current_state(text: str, func_defs: list[FunctionDefinition], llm_model: Small_LLM_Model) -> list[int]:


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Constraining")

    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")

    args = parser.parse_args()

    func_defs = load_functions(args.functions_definition)

    inputs = load_input(args.input)

    llm_model = Small_LLM_Model()

    func1 = func_defs[1]
    input1 = inputs[1]

    prompt = f"""You have access to the following function:

    Function: {func1.name}
    Description: {func1.description}
    Parameters: {func1.parameters}

    Given the user's question, respond with a JSON object specifying which
    function to call an with what parameters.

    User question: {input1.prompt}

    Respond ONLY with a JSON object in this format:
    {{"name": "function_name", "parameters": {{...}}}}

Response: """

    encoded = llm_model.encode(prompt)
    input_ids = encoded[0].tolist()
    output_ids = []

    while True:
        full_context = input_ids + output_ids
        logits = llm_model.get_logits_from_input_ids(full_context)

        valid_tokens =

        for i in range(len(logits)):
            if i not in valid_tokens:
                logits[i] = -float('inf')

        best_token_id = np.argmax(logits)
        output_ids.append(best_token_id)

        generated_text = llm_model.decode(output_ids)
        print(generated_text)

        if is_json_complete(generated_text):
            break


if __name__ == "__main__":
    main()

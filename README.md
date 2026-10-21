*This project has been created as part of the 42 curriculum by nda-roch.*

## Description

Call Me Maybe is a function calling system that translates natural language prompts into structured JSON function calls using constrained decoding. Given a question like "What is the sum of 40 and 2?", the system uses a small language model (Qwen3-0.6B) to output a valid JSON object specifying which function to call and with what parameters.

The project demonstrates how constrained decoding can make small language models achieve reliable structured output generation, solving the problem of LLMs that naturally generate unstructured text.

## Instructions

### Prerequisites

- Python 3.12+
- uv package manager

### Installation

```bash
uv sync
```

This installs numpy, pydantic, accelerate, and llm-sdk. llm-sdk brings in torch, transformers, and huggingface-hub.

### Running the Program

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

**Default paths:**
- Functions definition: `data/input/functions_definition.json`
- Input prompts: `data/input/function_calling_tests.json`
- Output results: `data/output/function_calling_results.json`

### Example

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Resources

### References

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - OpenAI GPT-3 paper
- [JSON Schema](https://json-schema.org/) - JSON validation specification
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation library
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - LLM framework

### AI Usage

AI tools were used as a learning aid throughout this project, following the 42 curriculum guidelines:

- **Code assistance**: AI helped debug token encoding/decoding issues and understand tensor operations from the LLM SDK
- **Design guidance**: Discussed state machine architecture for constrained decoding
- **Documentation**: AI assisted in writing docstrings and comments, with manual review and editing
- **Debugging**: Helped trace through token generation loops and identify state machine logic errors

All AI-generated code was reviewed, tested, and modified as needed. The core algorithm design and implementation were developed through understanding the concepts and writing the solution manually, with AI serving as a supplementary resource for clarification and error diagnosis.

## Algorithm Explanation

### Constrained Decoding Approach

The system uses a state machine to track the current position in the JSON output and constrain token selection at each step:

1. **State Detection**: Analyze the generated text to determine which JSON field is being written
2. **Token Filtering**: Based on the current state, identify which tokens are valid
3. **Logit Masking**: Set invalid token logits to negative infinity before selection
4. **Token Selection**: Choose the highest-probability token from the remaining valid options

### Current Implementation

The output is built one token at a time. Before each token is chosen, the
decoder looks at the text generated so far and decides which characters are
allowed next. Every token that isn't allowed gets its logit set to `-inf`, so
`argmax` can only pick a valid one.

The state is stored in an explicit variable and updated after every token by `walk_token()`. Each state allows only the following:

1. **Start:** nothing matches `{"name": "` yet, so only the characters of
   `{"name": "` are allowed, one at a time.
2. **Function name:** the characters of the function names that match the
   text so far. When a name is complete, its closing `"` is allowed.
3. **After the name:** a comma is allowed, then a space, the quote, or both.
4. **Parameters key:** the letters of `parameters`, then its closing quote.
5. **After the key:** a colon, then a space or `{`, then `{`.
6. **Parameter names:** the letters of the valid parameter names for the
   chosen function. Once a name is closed, a colon is allowed.
7. **Parameter values:** the parameter's type comes from the schema. `number` allows digits and `-` after the colon, then more digits, `,`, or `}`. `string` allows `"` after the colon to open the value. Inside the string, the allowed tokens come from the precomputed vocabulary list. After the closing `"`, only `,` or `}` is allowed.

The first parameter name is found after `"parameters": {"`. Later parameter
names are found after `, "`, using the last occurrence.

Before generation, `get_vocabulary()` builds a dict of token text to token ID, keeping only tokens whose text uses allowed characters, once per run. The `Ġ` marker becomes a space before the check.

### Token Extraction

The `extract_value_after_pattern()` function:
- Searches for a pattern in the text
- Finds the start position after the pattern
- Scans for the earliest stop character (from a list of possibilities)
- Returns the text between start and stop positions

This handles cases where we need to extract partial values during generation (like a function name being built token by token).

### Token Validation

The `get_valid_next_tokens()` function:
- Takes the partial text generated so far
- Compares it against valid options (function names or parameter names)
- Finds which options match the current prefix
- Returns the token IDs that could continue any matching option

This ensures the model can only generate tokens that lead to valid JSON structure and schema compliance.

### Escapes and Strings

A quote inside a value is only a boundary if it isn't escaped. `is_escaped_quote()`
walks left from a quote and counts the backslashes directly in front of it. An odd
count means the quote is escaped. `is_in_string()` counts the unescaped quotes in
the text, and an odd total means the text ends inside a string.


## Design Decisions

### Modular Architecture

The code is organized into four modules:
- `models.py`: Pydantic models for data validation (FunctionDefinition, TestPrompt, ParamType)
- `decoder.py`: Constrained decoding logic and state machine
- `generator.py`: Main generation loop and data loading
- `cli.py`: Command-line interface and argument parsing

This separation makes each component testable and maintainable.

### Generic Helper Functions

Created reusable functions like `extract_value_after_pattern()` and `get_valid_next_tokens()` that work for both function names and parameter names. This reduces code duplication and makes the state machine easier to extend.

### Pydantic for Validation

Used Pydantic models to validate input JSON files at load time. This catches malformed data early and provides clear error messages, meeting the subject's requirement for robust error handling.

### State Machine Pattern

The constrained decoding is being refactored into a state machine to make the logic explicit and maintainable. `decoder.py` contains a `State` enum (including the outer `F_START`, `F_NAME`, and `F_PARAM` states), a transition table, and `next_state()`, which covers the outer object, function name, parameter names, colon, string, and number states. `walk_token()` runs one candidate token through `next_state()`, and `allowed_tokens()` collects the IDs of the vocabulary tokens that pass. `generator.py` has `find_function()`, which looks up the chosen function. `generate()` uses `allowed_tokens()` and `walk_token()` with an explicit state variable. It passes the chosen function name to both, and uses `vocab_invert` (ID to text) to get the text of each chosen token. The old helpers in `decoder.py` are still present and will be removed once the refactor runs end to end. [TODO: the current run fails with a `KeyError` when looking up a parameter name; end-to-end output is not verified yet.]

## Known Limitations

- Function selection is not reliable for every prompt (for example, "Reverse the string" and "square root" get `fn_add_numbers`). The final choice is the model's.
- Escape sequences are not supported. Backslashes are removed from string content.
- The state machine in `decoder.py` plans to support escape sequences other than `\u`, which is deliberately rejected because the test prompts do not need it.

## Performance Analysis

[TODO: Add performance metrics after completing the implementation]

### Accuracy
[TODO: Measure correct function selection rate across test prompts]

### Speed
[TODO: Measure tokens per second and total processing time]

### Reliability
[TODO: Measure percentage of outputs that are valid JSON and schema-compliant]

## Challenges Faced

### Token Encoding Mismatch

**Problem**: Initially tried to compare function names as strings, but the LLM generates tokens, not characters. A function name like `"fn_greet"` might be encoded as multiple tokens `[8822, 1889, 3744]`.

**Solution**: Implemented token-level comparison by encoding each candidate option and comparing token sequences, ensuring we constrain at the right granularity.

### Parameter Type Constraints

**Problem**: Different parameters have different types, and the first version allowed number tokens after every colon, so string values never opened.

**Solution**: The decoder reads the parameter's type from the schema. The `number` lane allows digits and `-`. The `string` lane allows `"` to open the value, then uses the vocabulary list.

### Escapes and Quotes

**Problem**: A `"` can open a value, close a value, or be part of the content of a value, and a backslash can turn a quote into content. Counting quotes alone gives the wrong answer when escaped quotes are present.

**Solution**: `is_escaped_quote()` walks left from a quote and checks whether an odd number of backslashes precedes it. `is_in_string()` counts only the unescaped quotes, so an odd total means the text ends inside a string. Backslashes are currently removed from string content, so escape sequences are not supported yet.

### Multi-Character Tokens

**Problem**: The masks were built one character at a time, but tokens like `hunter` span several characters, so the model could only produce single letters in strings (`"s"` instead of `"shrek"`).

**Solution**: Build the allowed token list from the vocabulary, keeping every token whose text uses only allowed characters.

## Testing Strategy

- Helper checks: `is_escaped_quote()` on `"a\"b"`, `"a\\"b"`, and `"a\\\"b"` returns `True`, `False`, and `True` at the quote positions. `is_in_string()` on `"parameters": {"name": "` returns `True`, and on `"a\"b"` returns `False`.
- Single-prompt run: `one_prompt.json` produced a valid result for `What is the sum of 2 and 3?` before the refactor. [TODO: re-verify after the refactor run completes.]
- Full run: string values such as `"shrek"` and `"john"` are complete. Known gap: function choice is wrong for several prompts.
- Validation of output JSON structure using Python's json module.
- Edge case testing (TO-DO)

## Example Usage

Output for the prompt "What is the sum of 2 and 3?" (from `data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

Output for the prompt "Greet shrek" and "Greet john" (from `data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": "john"
    }
  }
]
```
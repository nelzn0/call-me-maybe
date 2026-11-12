*This project has been created as part of the 42 curriculum by nda-roch.*

## Description

Call Me Maybe is a function calling system that translates natural language prompts into structured JSON function calls using constrained decoding. Given a question like "What is the sum of 40 and 2?", the system uses a small language model (Qwen3-0.6B) to output a valid JSON object specifying which function to call and with what parameters.

The project demonstrates how constrained decoding can make small language models achieve reliable structured output generation, solving the problem of LLMs that naturally generate unstructured text.

## Instructions

### Prerequisites

- Python 3.10+
- uv package manager

### Installation

```bash
uv sync
```

This installs numpy, pydantic, accelerate, and llm-sdk. llm-sdk brings in torch, transformers, and huggingface-hub.

The Makefile provides these rules: `make install`, `make run`, `make debug`, `make clean`, `make lint`, and `make lint-strict`.

### Running the Program

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

**Default paths:**
- Functions definition: `data/input/functions_definition.json`
- Input prompts: `data/input/function_calling_tests.json`
- Output results: `data/output/function_calling_results.json`

The program prints a title banner, then status lines for the functions and prompts. After each prompt finishes, it prints the prompt and its JSON result, and it prints a summary at the end. Errors go to stderr. The program exits with code 1 for file or output errors and with code 130 on Ctrl+C. Colors are used only when output goes to a terminal, and they are turned off when `NO_COLOR` is set.

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
   chosen function. Once a name is closed, a colon is allowed, and that name is removed from the list so it can't be used again. The closing
   quote must be the last character of its token, so a token like `":` is
   rejected and the colon comes in a later token.
7. **Parameter values:** the parameter's type comes from the schema. `number` allows spaces, digits, and `-` after the colon. After a digit, it allows more digits, one `.` followed by digits, `,`, or `}`. `string` allows spaces, then `"` after the colon to open the value. Inside the string, the allowed tokens come from the precomputed vocabulary list. After the closing `"`, the value is done. `boolean` allows spaces after the colon, then `t` or `f`. The letters have to spell `true` or `false`, and the full word ends the value.
8. **After a value:** a `,` is allowed only while parameter names are left, and a `}` is allowed only after all of them are used. The model can't start an extra parameter or close the object early.

A parameter name is read when its closing quote is walked. `generate_call()` takes the text up to the closing quote, which includes the chosen token minus its last character, and then the text after the last quote, using `text_after_last_quote()`. It looks up that name in the schema and uses its type. A name that is not in the function's parameters raises a `ValueError`. This keeps a token such as `b"` from losing its letter.

Before generation, `get_vocabulary()` builds a dict of token text to token ID, keeping only tokens whose text uses allowed characters, once per run. The `Ġ` marker becomes a space before the check.

### State Diagram

Overview of the main path:

```
+--------------+    +---------------+    +----------------+    +---------+
| Outer object |--->| Function name |--->| Parameter name |--->|  Colon  |
+--------------+    +---------------+    +----------------+    +---------+
                                                                    |
                                                +-------------------+
                                                v                   v
                                         +--------------+   +--------------+
                                         | String value |   | Number value |
                                         +--------------+   +--------------+
                                         after a value:
                                           ',' --> back to Parameter name
                                           '}' --> Done
```

Each diagram shows one part of the state machine. `(stay)` means the state loops back to itself on that input. Each arrow label is the character (or forced piece of text) that moves the decoder to the next state.

**Header** (outer object, function name, parameters key):

```
 +----------+
 | F_START  |<-- prefix of {"name": "  (stay)
 +----------+
     |
     | {"name": "
     v
 +----------+
 |  F_NAME  |<-- letters of a function name  (stay)
 +----------+
     |
     | closing quote, name is valid
     v
 +----------+
 |  F_PARAM |<-- prefix of , "parameters": {  (stay)
 +----------+
     |
     | , "parameters": {
     v
 +----------+
 |  P_OPEN  |
 +----------+
     |
     | '"'
     v
   (parameter name)
```

**Parameter name** (the key of each parameter):

```
   (from ',' after a value)
 +----------+
 |  P_NEXT  |<-- space  (stay)
 +----------+
     |
     | '"'
     v
 +----------+
 |  P_NAME  |<-- letters of a parameter name  (stay)
 +----------+
     |
     | closing quote, name is valid
     v
 +----------+
 |  P_COLON |
 +----------+
     |
     | ':'
     +-- string type --> String value
     +-- number type --> Number value
```

**String value**:

```
 +----------+
 | STR_OPEN |<-- space  (stay)
 +----------+
     |
     | '"'
     v
 +----------+
 |IN_STRING |<-- allowed character  (stay)
 +----------+
     |
     | '"'
     v
 +--------------+
 |P_AFTER_VALUE |
 +--------------+
     |
     | ','   --> P_NEXT
     | '}'   --> P_CLOSE
```

A backslash inside a string would move to `IN_ESCAPE`, but it is not reachable yet because backslash is removed from the vocabulary.

**Number value**:

```
 after ':'
     |
     v
 +-----------+   '-'   +--------------+
 | NUM_START |-------->| NUM_NEGATIVE |
 +-----------+         +--------------+
     |    ^                   |
     |    '-- space (stay)    |
     |                        |
     | digit                  | digit
     |                        |
     v                        v
 +------------------------------------+
 |              NUM_INT               |<-- digit (stay)
 +------------------------------------+
     |                 ',' --> P_NEXT
     | '.'             '}' --> P_CLOSE
     v
 +-------------------+  digit   +--------------+
 | NUM_DECIMAL_POINT |--------->| NUM_DECIMAL  |<-- digit (stay)
 +-------------------+          +--------------+
                                    |
                                    | ',' --> P_NEXT
                                    | '}' --> P_CLOSE
```

**Boolean value**:

```
  after ':'
      |
      v
  +----------+
  | BOOL_OPEN|<-- space  (stay)
  +----------+
      |
      | 't' or 'f'
      v
  +----------+
  | IN_BOOL  |<-- letters that still spell true or false  (stay)
  +----------+
      |
      | last letter of true or false
      v
  +--------------+
  |P_AFTER_VALUE |
  +--------------+
```

**Closing**:

```
 +----------+
 |  P_CLOSE |
 +----------+
     |
     | '}'
     v
 +----------+
 |   DONE   |
 +----------+
```

### Token Extraction

The `extract_value_after_pattern()` function:
- Searches for a pattern in the text
- Finds the start position after the pattern
- Scans for the earliest stop character (from a list of possibilities)
- Returns the text between start and stop positions

This handles cases where we need to extract partial values during generation (like a function name being built token by token).

### Token Validation

The `allowed_tokens()` function:
- Loops over the vocabulary
- Runs each token's characters through `walk_token()` from the current state
- Returns the IDs of the tokens where every character moves the state machine forward

This ensures the model can only generate tokens that lead to valid JSON structure and schema compliance.

### Forced Steps

`generate_call()` computes the valid tokens before it asks the model for logits. If exactly one token is valid, that token is chosen directly, and the model is not called for that step. The chosen token still goes through `walk_token()`, is appended to the output, and the new state is checked for `DONE`, the same as a token picked by the model. Since the mask leaves only one token, the model's scores could not change the choice.

### Allowed-Token Cache

The `get_allowed_tokens()` function wraps `allowed_tokens()` with a cache:
- The cache is a dict created once per run in `generate()` and passed into each `generate_call()`.
- The key is a tuple of four slots: the state, a text value, the options as a tuple, and the parameter type. Each state fills only the slots its branch in `next_state()` reads, and leaves the others as `None`.
- The cache's type is named `CACHE_TYPE` in `decoder.py`, and `generator.py` uses the same name.
- `F_START` uses the full text generated so far, so identical text gives the same key in every prompt.
- `F_NAME` and `P_NAME` use the text typed since the last quote and the options. `F_PARAM` uses the text after the function name, found by `find_after_name()`, and the options.
- `P_COLON` uses the parameter type. `IN_BOOL` uses the letters typed after the colon, with the spaces removed.
- `IN_STRING`, `P_NEXT`, `P_OPEN`, `P_AFTER_VALUE`, `NUM_INT`, and `NUM_DECIMAL` use the remaining parameter names. A token can run into a parameter name, and `,` or `}` depends on whether names are left.
- Other states use only the state.
- On a hit, the stored set is returned. On a miss, `allowed_tokens()` computes the set, which is stored and returned.

### Escapes and Strings

Inside a string, `next_state()` has an escape path: a `\` moves to `IN_ESCAPE`, and the next character must be one of `"\/bfnrt`. Backslash is not in the vocabulary used for string values, so this path is not reachable yet.


## Design Decisions

### Modular Architecture

The code is organized into four modules:
- `models.py`: Pydantic models for data validation (FunctionDefinition, TestPrompt, ParamType)
- `decoder.py`: Constrained decoding logic and state machine
- `generator.py`: Main generation loop and data loading
- `cli.py`: Command-line interface, argument parsing, title banner, progress output, and error handling

This separation makes each component testable and maintainable.

### Generic Helper Functions

Created reusable functions like `extract_value_after_pattern()` and `allowed_tokens()` that work for both function names and parameter names. `text_after_last_quote()` and `find_after_name()` are shared by `next_state()` and the cache key, so both read the same text. This reduces code duplication and makes the state machine easier to extend.

### Pydantic for Validation

Used Pydantic models to validate input JSON files at load time. This catches malformed data early and provides clear error messages, meeting the subject's requirement for robust error handling.

### State Machine Pattern

The constrained decoding is a state machine. `decoder.py` contains a `State` enum (including the outer `F_START`, `F_NAME`, and `F_PARAM` states), a transition table, and `next_state()`, which covers the outer object, function name, parameter names, colon, string, and number states. `walk_token()` runs one candidate token through `next_state()`, and `allowed_tokens()` collects the IDs of the vocabulary tokens that pass. `generator.py` has `find_function()`, which looks up the chosen function, `capture_function_name()`, which reads the function name from the generated text, and `pick_token()`, which masks the logits with `mask_logits()`, picks the highest one, and returns its ID and text. `build_functions_text()`, `build_prompt()`, `invert_vocab()`, and `write_result()` handle the prompt and the output file. `generate_call()` runs the token loop for one prompt with an explicit state variable, passing the chosen function name to `get_allowed_tokens()` and `walk_token()`, and returns the parsed JSON. If exactly one token is valid, `generate_call()` skips the model for that step (see Forced Steps). `cli.py` loads the files and passes them to `generate()`. `generate()` builds the vocabulary once, calls `generate_call()` for each prompt, and saves the results. It takes an optional `on_result` callback, which it calls with the prompt's position, the total count, and the result after each successful prompt, and it returns the list of results. `cli.py` uses the callback to print each result. `walk_token()` returns the state after the token, or None if a character is rejected. `walk_token()` also rejects a token in which the closing quote of a parameter name is followed by more characters.

### Parameter Rules

Each parameter name can be used once per call. The name is taken out of `options` when its closing quote is done, so the model can't repeat it. Commas and closing braces are checked against the names that are left: a `,` needs at least one name left, and a `}` needs none. Without these rules, the model could keep adding parameters until it hit the token limit, or close the object before all the required arguments were written.

### Error Handling

Each prompt is handled on its own. If `generate_call()` raises a `ValueError` (no valid token, no matching function or parameter, or the JSON is not finished after `MAX_TOKENS`), `generate()` prints the prompt and the reason, then moves on. `generate()` catches any other error too, so one unexpected error only fails its own prompt. Errors that get past `generate()` are caught in `cli.py`, which prints one line and exits with code 1. The output can therefore have fewer entries than the input.

`cli.py` handles missing or invalid input files and output write errors. It prints one error line to stderr and exits with code 1.

## Known Limitations

- Function selection and argument extraction were checked on the 11 prompts in `data/input/function_calling_tests.json`. The output matches the earlier hand-checked run. Other prompts are untested, and the final choice is the model's.
- Only `number`, `string`, and `boolean` parameter types are supported. Other types stop generation with an error.
- `boolean` was tested with a small temporary functions file and two prompts, not in the main test file.
- Functions with no parameters are not supported.
- String values cannot contain `"` or `\`, and characters outside ASCII are not in the vocabulary used for string values.
- Escape sequences are not supported. Backslashes are removed from string content.
- The state machine in `decoder.py` plans to support escape sequences other than `\u`. `\u` is not accepted because `u` is not in the list of escape characters, and the test prompts do not need it.
- `{` and `}` are allowed inside string values. A `}` inside a value doesn't end the JSON early, because the output only stops after the closing `}` of the whole object. A value like `{x}` was tested with one prompt and came out correctly.
- Any exception raised while loading the model or the vocabulary is caught in `generate()` and reported as a `ValueError`.
- A top-level JSON value that is not an array in the input files is rejected with a clear `ValueError`.

### Number Format

Numbers are written the way the model produced them. If the model picks `2`, the output has `2`, and if it picks `2.0`, the output has `2.0`. I didn't convert everything to floats, because a float can't hold every big integer exactly, so something like `12345678901234567890` would get rounded or written in scientific notation. Both `2` and `2.0` are valid JSON numbers and match the `number` type in `functions_definition.json`. The subject's example shows `2.0`, but that's an example, not a required format.

## Performance Analysis

The timing below was measured before the parameter-name and cache changes. A new timing still needs to be recorded.

### Accuracy
All 11 prompts in `data/input/function_calling_tests.json` produce a function name and parameters from the schema, and by inspection the chosen functions match the prompts. There is no expected-output file to score against, so the 11 outputs were compared with the earlier run, and they matched exactly. This is a small test set, so it doesn't show accuracy on other prompts.

### Speed
Measured with `time make run` on a GTX 1650, including model loading. The latest run, with the forced-step skip, took 4:14.38 for the 11 prompts, under the subject's 5-minute target. Earlier versions took 5:13 to 6:02. Those earlier runs generated 319 tokens at about 0.9 to 1.3 seconds per token. `nvidia-smi` showed the GPU at 99% utilization, so the forward pass over the full prompt is the main cost. [TODO: repeat the timing over more runs, and count the forced steps in a full run]

### Reliability
All 11 outputs were checked by hand against the function schema (function names and parameter names). [TODO: validate the outputs with a script using Python's json module]. This is measured on the test file only.

## Challenges Faced

### Token Encoding Mismatch

**Problem**: Initially tried to compare function names as strings, but the LLM generates tokens, not characters. A function name like `"fn_greet"` might be encoded as multiple tokens `[8822, 1889, 3744]`.

**Solution**: Implemented token-level comparison by encoding each candidate option and comparing token sequences, ensuring we constrain at the right granularity.

### Parameter Type Constraints

**Problem**: Different parameters have different types, and the first version allowed number tokens after every colon, so string values never opened.

**Solution**: The decoder reads the parameter's type from the schema. The `number` lane allows digits and `-`. The `string` lane allows `"` to open the value, then uses the vocabulary list.

### Escapes and Quotes

**Problem**: A `"` can open a value, close a value, or be part of the content of a value, and a backslash can turn a quote into content. Counting quotes alone gives the wrong answer when escaped quotes are present.

**Solution**: Backslashes are currently removed from the vocabulary, so string values cannot contain `\`. `next_state()` has an escape path for `\`, but it is not reachable until backslash tokens are allowed. Escape sequences are not supported yet.

### Multi-Character Tokens

**Problem**: The masks were built one character at a time, but tokens like `hunter` span several characters, so the model could only produce single letters in strings (`"s"` instead of `"shrek"`).

**Solution**: Build the allowed token list from the vocabulary, keeping every token whose text uses only allowed characters.

### Boolean Values

**Problem**: `true` and `false` have no quotes, so the closing-quote rule used for strings doesn't work, and the decoder has to check the word letter by letter.

**Solution**: After the colon, the decoder compares the letters typed so far with `true` and `false`, ignoring the spaces. When the full word is typed, the value is done, and `,` or `}` can follow.

### Repeated Parameter Names

**Problem**: In a boolean test, the model kept writing `"on": true, "on": false, ...` because nothing stopped it from reusing a name. The object never closed, and the prompt failed at the token limit. A full run also dropped `regex` and `replacement` from the regex prompts, because a set saved for an earlier prompt was reused.

**Solution**: A name is removed from `options` once its closing quote is done. `,` and `}` are checked against the names that are left, and the states that can reach a name keep those names in their cache key.

## Testing Strategy

- Helper checks: `build_functions_text()`, `build_prompt()`, `invert_vocab()`, `mask_logits()`, `find_function()`, `capture_function_name()`, and `write_result()` were checked by hand against the real input files. These are not yet written as unit tests. [TODO: add unit tests]
- Full run: the 11-prompt test file was re-run after the prompt changes, and the output is identical to the earlier run in `data/output/function_calling_results_before.json`. The run after the forced-step skip was also compared with `diff` and was identical. The run after the parameter-name and cache changes was compared the same way, and it was identical again. The vowel prompt uses the regex `([aeiouAEIOU])`.
- Boolean: a temporary functions file with one `boolean` parameter, and the prompts "Turn the light on" and "Turn the light off". Both came out as JSON booleans (`true` and `false`).
- Validation of output JSON structure using Python's json module. [TODO: write the validation script]
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
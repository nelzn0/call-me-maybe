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

The program prints each prompt and its JSON result as it finishes. Errors go to stderr, and the program exits with code 1 if a file fails or any prompt fails. The output file gets one entry for every prompt, including failed ones, so the answers stay in the same order as the tests.

### Example

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Algorithm Explanation

### Constrained Decoding Approach

The system uses a state machine to track the current position in the JSON output and constrain token selection at each step:

1. **State Detection**: Analyze the generated text to determine which JSON field is being written
2. **Token Filtering**: Identify which tokens are valid in the current state
3. **Logit Masking**: Set invalid token logits to negative infinity
4. **Token Selection**: Choose the highest-probability valid token

### Current Implementation

The output is built one token at a time. Before each token is chosen, the
decoder looks at the text generated so far and decides which characters are
allowed next. Every token that isn't allowed gets its logit set to `-inf`, so
`argmax` can only pick a valid one.

The state is stored in an explicit variable and updated after every token by `walk_token()`. Each state allows only the following:

1. **Start:** the text `{"name": "` is forced. The model isn't called for it, and the state moves straight to the function name.
2. **Function name:** the characters of the function names that match the
   text so far. When a name is complete, its closing `"` is allowed.
3. **After the name:** the text `, "parameters": {` is forced once the name's closing quote is done.
4. **Parameters key:** nothing to choose here, since the forced text above already covers `"parameters": {`.
5. **After the key:** a colon, then a space or `{`, then `{`.
6. **Parameter names:** the letters of the valid parameter names for the
   chosen function. Once a name is closed, a colon is allowed, and that name is removed from the list so it can't be used again. The closing quote must be the last character of its token.
7. **Parameter values:** the type comes from the schema. `integer` allows digits and `-`, but no `.`. `number` allows digits, `-`, and one `.`, and it needs digits after the `.`. `string` opens with `"` and uses the precomputed vocabulary list. `boolean` must spell `true` or `false`.
8. **After a value:** a `,` is allowed only while parameter names are left, and a `}` only after all of them are used.

### Forced Text and Shorter Prompt

The fixed parts of the JSON, `{"name": "` and `, "parameters": {`, never change. Asking the model for them only costs time, so `force_text()` adds them directly. The text is encoded to token IDs and walked through `walk_token()`, so the state moves the same way it would if the model had picked those tokens. If the state machine rejects the text, `force_text()` raises a `ValueError`.

The model re-reads the whole prompt for each token it writes. Once the function name is picked, the prompt is rebuilt with only that function, and the other function definitions are no longer sent. The output built so far is kept, so the JSON carries on from where it was.

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

In `P_OPEN`, a `}` is allowed only when the function has no parameters, and it goes straight to `P_CLOSE`. A `"` is allowed only when there are parameters.

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

A backslash inside a string moves to `IN_ESCAPE`, and the next character must be one of `"\/bfnrt`.

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

For `number`, `NUM_INT` doesn't accept `,` or `}`, so the value has to go through `NUM_DECIMAL` first. For `integer`, `NUM_INT` rejects `.`.

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

### Token Validation

The `allowed_tokens()` function:
- Loops over the vocabulary
- Runs each token's characters through `walk_token()` from the current state
- Returns the IDs of the tokens where every character moves the state machine forward

This ensures the model can only generate tokens that lead to valid JSON structure and schema compliance.

### Escapes and Strings

Inside a string, `next_state()` has an escape path: a `\` moves to `IN_ESCAPE`, and the next character must be one of `"\/bfnrt`. The regex `\d+` is written in the JSON as `"\\d+"`, which parses to `\d+`. Tokens such as `\"` work because `walk_token()` checks each character.


## Design Decisions

### Modular Architecture

The code is split into four modules: `models.py` (Pydantic models), `decoder.py` (state machine), `generator.py` (generation loop and data loading), and `cli.py` (arguments, output, and errors). This keeps each part testable and maintainable.

### Pydantic for Validation

Used Pydantic models to validate input JSON files at load time. This catches malformed data early and provides clear error messages, meeting the subject's requirement for robust error handling.

### Parameter Rules

Each parameter name can be used once per call. The name is taken out of `options` when its closing quote is done, so the model can't repeat it. Commas and closing braces are checked against the names that are left: a `,` needs at least one name left, and a `}` needs none. Without these rules, the model could keep adding parameters until it hit the token limit, or close the object before all the required arguments were written.

Each parameter's type from the schema also decides which characters are allowed. `integer` takes digits and `-`, and a `.` is rejected. `number` takes digits, `-`, and a `.`, and it needs at least one digit after the `.`. `string` and `boolean` follow the rules in the Current Implementation section.

### Error Handling

Each prompt is handled on its own. If `generate_call()` raises a `ValueError` (no valid token, no matching function or parameter, or the JSON is not finished after `MAX_TOKENS`), `generate()` prints the prompt and the reason, then moves on. `generate()` catches any other error too, so one unexpected error only fails its own prompt. Errors that get past `generate()` are caught in `cli.py`, which prints one line and exits with code 1.

A failed prompt still gets an entry, written as `{"prompt": ..., "name": "", "parameters": {}}`. `cli.py` counts successes by the entries that have a name, so these placeholders are not counted as successes.

`cli.py` handles missing or invalid input files and output write errors. It prints one error line to stderr and exits with code 1.

### Number Format

The type decides the shape of a number. An `integer` parameter is written as digits only (`2`), and a `number` parameter needs a decimal part (`3.0`). A plain `3` is rejected for a `number` parameter, so the value always reads back as a float.

I still don't convert the values to Python floats in the decoder, because a float can't hold every big integer exactly. The text is kept as the model wrote it, and the type rules above decide which text is valid.

The cache key includes the parameter type. Without it, a result cached for an `integer` could be reused for a `number`, and the two types now allow different characters.

## Performance Analysis

### Accuracy
All 11 prompts in `data/input/function_calling_tests.json` produce a function name and parameters from the schema, and by inspection the chosen functions match the prompts. There is no expected-output file to score against, so the outputs were compared with the earlier run, and they matched exactly. This is a small test set, so it doesn't show accuracy on other prompts.

### Speed
Measured on a GTX 1650, including model loading. The 11 prompts take about 4:10 to 4:20 with `make run`, under the subject's 5-minute target. Earlier versions took 5:13 to 6:02. After forcing the fixed JSON parts and shortening the prompt after the function name, one run took about 2:33, and the output matched the earlier run exactly. `nvidia-smi` showed the GPU at 99% utilization, so the forward pass over the full prompt is the main cost.

### Reliability
All 11 outputs were checked by hand against the function schema, and the same checks were run automatically with a validation script. All 11 outputs pass. This is measured on the test file only.

## Challenges Faced

### Token Encoding Mismatch

**Problem**: Initially tried to compare function names as strings, but the LLM generates tokens, not characters. A function name like `"fn_greet"` might be encoded as multiple tokens `[8822, 1889, 3744]`.

**Solution**: Implemented token-level comparison by encoding each candidate option and comparing token sequences, ensuring we constrain at the right granularity.

### Parameter Type Constraints

**Problem**: Different parameters have different types, and the first version allowed number tokens after every colon, so string values never opened.

**Solution**: The decoder reads the parameter's type from the schema. The `number` lane allows digits and `-`. The `string` lane allows `"` to open the value, then uses the vocabulary list.

### Escapes and Quotes

**Problem**: A `"` can open a value, close a value, or be part of the content of a value, and a backslash can turn a quote into content. Counting quotes alone gives the wrong answer when escaped quotes are present.

**Solution**: Backslash is kept in the vocabulary, and `next_state()` has an escape path for `\`. Only `IN_STRING` accepts a backslash, so it can't appear in other states. Each character of a token goes through `next_state()`, so `\"` and `\\` work inside a single token.

The decoder could only allow `\"`. It couldn't make the model write it. In a test with `Say "hello" to {name}`, the model dropped the escapes, so the quotes were not valid inside the string. To fix this, the system prompt has a few-shot example with an escaped quote: `Say \"hi\" to Bob` becomes a `fn_format_template` call. This shows the model to write `\"` inside a string.

The example is written in general form, but it reuses the test's function name, `fn_format_template`, so it is tuned to this test file. If the function names change, the example should change with them.

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

The test files used during development are not included in the repository. The checks below were run locally.

- Unit tests (no model needed): the state machine (empty input, large numbers, wrong types, special characters, escapes, zero parameters, braces in values), the helpers (vocabulary building, prompt building, logit masking, token picking, function lookup), and input loading (wrong types, empty files, invalid JSON, missing files).
- Full run: the 11-prompt test file was re-run after each change, and the output was identical to the baseline run.
- Output validation: a script checked that the output is a JSON array, that each entry has exactly `prompt`, `name`, and `parameters`, that the function exists, that the parameters match the schema exactly, that each value has the declared type, and that there is one entry per input prompt.
- Boolean: a temporary functions file with one `boolean` parameter and the prompts "Turn the light on" and "Turn the light off" gave `true` and `false`.
- Zero-parameter functions: a temporary functions file with `fn_get_time` and the prompt "What time is it?" gave `"parameters": {}`.
- Edge cases: 12 prompts in a temporary edge-case set produced schema-valid output. Results by category:
  - Empty input: the empty prompt `""` produced `fn_get_time` (a guess, since nothing was asked). `Reverse the string ''` produced `"s": ""`.
  - Large numbers: `123456789012345` and `987654321098765` were written exactly.
  - Wrong types: `Add 'two' and 'three'` produced `"a": 2, "b": 3`, so the model turned the words into numbers.
  - Ambiguous prompt: `Do the thing with 5` produced `fn_add_numbers` with `a: 5, b: 5`.
  - Special characters: `O'Brien-Smith!@#$%^&*()` produced `"O'Brien-Smith!"`, so the model dropped the trailing symbols.
  - Zero parameters: `What time is it?` produced `"parameters": {}`.
  - Booleans: `on` and `off` produced `true` and `false`.
  - Braces: `{x}` is kept inside the string value.
  - Escapes: `\\d+` in the regex prompt was written correctly, but `Reverse the string 'say "hi" now'` still gives `"say "`, because the decoder allows a quote to close the string early.
- Non-ASCII: four prompts with non-ASCII characters (`Greet José`, `Reverse the string 'café'`, `Greet 你好`, `Greet 😀`) were run. All four exited with code 0, so the failure is silent. The values are wrong, as shown above.
- Failed prompts: a temporary functions file with one `fn_make_list` function, which has an `array` parameter. The decoder doesn't support that type, so the prompt "Make a list of apples and pears" failed with `ValueError: No valid tokens found`. The other two prompts in the same run, "Greet shrek" and "Greet john", succeeded. The output file had two entries, and the exit code was 1.

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

## Resources

### References

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - OpenAI GPT-3 paper
- [JSON Schema](https://json-schema.org/) - JSON validation specification
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation library
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - LLM framework

### AI Usage

AI was used as a tutor and assistant throughout this project, following the 42 curriculum guidelines. No AI-generated code is in the project. All code was written and reviewed by me.

- **Concept explanations**: AI explained tokenizers, logits, and constrained decoding, and helped me understand tensor operations from the LLM SDK
- **Design discussion**: Talked through state machine architecture for constrained decoding. The design choices are mine.
- **Debugging guidance**: AI helped me trace token encoding and decoding issues, token generation loops, and state machine logic errors. I made the fixes myself.
- **Documentation**: AI helped with the wording of documentation, including docstrings, comments, and this README. I reviewed and edited all of it.

The core algorithm design and implementation were developed through understanding the concepts and writing the solution manually.

import argparse
import json
import sys
from typing import Any

from call_me_maybe.generator import generate, load_json_data
from call_me_maybe.models import FunctionDefinition, TestPrompt
from call_me_maybe.colors import colorize, RED, GREEN, YELLOW, BLUE, CYAN


BANNER = """ ██████╗ █████╗ ██╗     ██╗         ███╗   ███╗███████╗
██╔════╝██╔══██╗██║     ██║         ████╗ ████║██╔════╝
██║     ███████║██║     ██║         ██╔████╔██║█████╗
██║     ██╔══██║██║     ██║         ██║╚██╔╝██║██╔══╝
╚██████╗██║  ██║███████╗███████╗    ██║ ╚═╝ ██║███████╗
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝

███╗   ███╗ █████╗ ██╗   ██╗██████╗ ███████╗
████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝
██╔████╔██║███████║ ╚████╔╝ ██████╔╝█████╗
██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██╗██╔══╝
██║ ╚═╝ ██║██║  ██║   ██║   ██████╔╝███████╗
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚══════╝"""


def show_result(index: int, total: int, entry: dict[str, Any]) -> None:
    """Prints one finished prompt and its JSON result.

    Args:
        index: The position of this prompt in the list
        total: How many prompts there are in total
        entry: The prompt and its result, as a dict
    """

    result = {k: v for k, v in entry.items() if k != "prompt"}
    print(colorize(f"[{index}/{total}] {entry['prompt']}", CYAN))
    print(f"  {json.dumps(result)}")


def main() -> None:
    """Reads the command-line arguments and runs the whole program."""

    # clear terminal
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")

    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling Constraining"
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="JSON file with the available functions",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="JSON file with the prompts to process",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="JSON file to write the results to",
    )

    args = parser.parse_args()

    print("")

    print(colorize(BANNER, BLUE))

    print("")

    try:
        print(colorize("Getting functions...", CYAN))
        functions = load_json_data(
            args.functions_definition, FunctionDefinition
        )
        print(f"  {len(functions)} functions found")

        print("")

        print(colorize("Getting inputs...", CYAN))
        prompts = load_json_data(args.input, TestPrompt)
        print(f"  {len(prompts)} prompts found")

        print("")

        print(colorize("Start generation", YELLOW))

        print("")

        results = generate(functions, prompts, args.output, show_result)

        failed = sum(1 for r in results if not r.get("name"))
        success = len(prompts) - failed

        print("")

        print(colorize(
            f"\nGeneration complete! {success}/{len(prompts)} "
            "prompts succeeded.", RED if failed else GREEN
        ))

        print("")

        print(f"Output written to: {args.output}")

        if failed:
            print(f"Error: {failed} prompt(s) have failed.", file=sys.stderr)
            sys.exit(1)

    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

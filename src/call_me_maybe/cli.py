import argparse
import json
import os
import sys
from typing import Any

from call_me_maybe.generator import generate, load_json_data
from call_me_maybe.models import FunctionDefinition, TestPrompt

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


def colorize(text: str, code: str) -> str:
    """Adds ANSI color codes, but only when printing to a terminal."""

    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def show_result(index: int, total: int, entry: dict[str, Any]) -> None:
    """Prints one finished prompt and its JSON result.

    Args:
        index: The position of this prompt in the list
        total: How many prompts there are in total
        entry: The prompt and its result, as a dict
    """

    result = {k: v for k, v in entry.items() if k != "prompt"}
    print(colorize(f"[{index}/{total}] {entry['prompt']}", "36"))
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

    print(colorize(BANNER, "34"))

    print("")

    try:
        print(colorize("Getting functions...", "33"))
        functions = load_json_data(
            args.functions_definition, FunctionDefinition
        )
        print(f"  {len(functions)} functions found")

        print("")

        print(colorize("Getting inputs...", "33"))
        prompts = load_json_data(args.input, TestPrompt)
        print(f"  {len(prompts)} prompts found")

        print("")

        print(colorize("Start generation", "33"))

        print("")

        results = generate(args, show_result)

        print("")

        print(colorize(
            f"\nGeneration complete! {len(results)}/{len(prompts)} "
            "prompts succeeded.", "32"
        ))

        print("")

        print(f"Output written to: {args.output}")

    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

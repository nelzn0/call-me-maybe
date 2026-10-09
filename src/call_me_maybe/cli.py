
import argparse
from call_me_maybe.generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Constraining")

    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")

    args = parser.parse_args()

    generate(args)

from llm_sdk import Small_LLM_Model
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Constraining")

    parser.add_argument("--function_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")

    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Function definition: {args.function_definition}")


if __name__ == "__main__":
    main()

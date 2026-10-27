"""The pydantic models that check the input files"""

from pydantic import BaseModel


class ParamType(BaseModel):
    """The type of one parameter, like "number" or "string"."""
    type: str


class FunctionDefinition(BaseModel):
    """one function from functions_definition.json"""
    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class TestPrompt(BaseModel):
    """one prompt from function_calling_tests.json"""
    prompt: str

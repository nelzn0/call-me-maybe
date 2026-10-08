from pydantic import BaseModel


class ParamType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class TestPrompt(BaseModel):
    prompt: str

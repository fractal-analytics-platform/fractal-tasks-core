from pydantic import BaseModel


class CustomModel(BaseModel):
    """
    Short description

    Attributes:
        x: Description of `x`
    """

    x: int

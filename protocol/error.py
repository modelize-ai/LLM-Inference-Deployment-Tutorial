from typing import Optional

from pydantic import BaseModel, Field


class Error(BaseModel):
    type: str = Field(default=...)
    detail: Optional[str] = Field(default=None)

    def __repr__(self):
        return f"{self.type}: {self.detail}"

    def __str__(self):
        return self.__repr__()


__all__ = [
    "Error"
]

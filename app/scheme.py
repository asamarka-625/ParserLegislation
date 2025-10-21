# Внешние зависимости
from typing import Annotated
from datetime import datetime
from pydantic import BaseModel, Field


class DataSite(BaseModel):
    name: Annotated[str, Field(strict=True, strip_whitespace=True)]
    publication_number: Annotated[int, Field(ge=0)]
    publication_date: datetime
    link_pdf: Annotated[str, Field(strict=True, strip_whitespace=True)]

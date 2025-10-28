# Внешние зависимости
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import sqlalchemy as sa
import sqlalchemy.orm as so
from sqlalchemy.ext.asyncio import AsyncAttrs


# Базовая модель
class Base(AsyncAttrs, so.DeclarativeBase):
    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# Модель органов власти
class Authority(Base):
    __tablename__ = 'authorities'

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    name: so.Mapped[str] = so.mapped_column(
        sa.String(256),
        unique=True,
        index=True,
        nullable=False
    )
    uuid_authority: so.Mapped[UUID] = so.mapped_column(
        sa.UUID,
        unique=True,
        index=True,
        nullable=False
    )

    legislations: so.Mapped[List["DataLegislation"]] = so.relationship(
        "DataLegislation",
        back_populates="authority",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Authority(id={self.id}, name='{self.name}')>"


# Модель данных законодательства
class DataLegislation(Base):
    __tablename__ = 'data_legislation'

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    name: so.Mapped[str] = so.mapped_column(
        sa.String(4096),
        unique=True,
        index=True,
        nullable=False
    )
    publication_number: so.Mapped[str] = so.mapped_column(
        sa.String(512),
        unique=True,
        index=True,
        nullable=False
    )
    publication_date: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime,
        index=True,
        nullable=False
    )
    link_pdf: so.Mapped[str] = so.mapped_column(
        sa.String(512),
        nullable=False
    )
    binary_pdf: so.Mapped[Optional[bytes]] = so.mapped_column(
        sa.LargeBinary,
        nullable=True
    )
    text: so.Mapped[Optional[str]] = so.mapped_column(
        sa.Text,
        nullable=True
    )
    law_number: so.Mapped[str] = so.mapped_column(
        sa.String(16),
        index=True,
        nullable=True
    )

    authority_id: so.Mapped[int] = so.mapped_column(
        sa.Integer, 
        sa.ForeignKey('authorities.id'),
        index=True,
        nullable=False
    )
    
    authority: so.Mapped["Authority"] = so.relationship(
        "Authority",
        back_populates="legislations"
    )

    def __repr__(self):
        return f"<DataLegislation(id={self.id}, name='{self.name}')>"
# Внешние зависимости
from typing import Sequence, Optional
from uuid import UUID
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
# Внутренние модули
from app.config import get_config
from app.database import connection
from app.models import Authority, DataLegislation
from app.scheme import DataSite


config = get_config()


# Добавляем новый орган власти
@connection
async def sql_add_new_authority(
        authority: str,
        uuid_authority: UUID,
        session: AsyncSession
) -> None:
    try:
        new_authority = Authority(
            name=authority,
            uuid_authority=uuid_authority
        )
        session.add(new_authority)
        await session.commit()

    except SQLAlchemyError as e:
        config.logger.error(f"Database error add new authority: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error add new authority: {e}")


# Получаем органы власти, которые еще не заполнены
@connection
async def sql_get_authorities_by_more_id(
        current_id: int,
        session: AsyncSession
) -> Sequence[Authority]:
    try:
        authorities_results = await session.execute(
            sa.select(Authority)
            .where(Authority.id > current_id)
        )
        authorities = authorities_results.scalars().all()
        return authorities

    except SQLAlchemyError as e:
        config.logger.error(f"Database error read authorities by more id: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error read authorities by more id: {e}")


# Добавляем новое законодательство
@connection
async def sql_add_new_legislation(
        authority_id: int,
        data: DataSite,
        session: AsyncSession
) -> None:
    try:
        new_legislation = DataLegislation(
            authority_id=authority_id,
            **data.model_dump()
        )
        session.add(new_legislation)
        await session.commit()

    except SQLAlchemyError as e:
        config.logger.error(f"Database error add new legislation: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error add new legislation: {e}")


# Выводим все законы, у которых нет байткода PDF файла
@connection
async def sql_get_legislation_by_not_binary_pdf(
        session: AsyncSession
) -> Sequence[DataLegislation]:
    try:
        legislation_results = await session.execute(
            sa.select(DataLegislation)
            .where(DataLegislation.binary_pdf == None)
            .order_by(DataLegislation.id)
        )

        legislation = legislation_results.scalars().all()
        return legislation

    except SQLAlchemyError as e:
        config.logger.error(f"Database error read legislation with none binary_pdf: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error read legislation with none binary_pdf: {e}")


# Записываем бинарный код PDF файла
@connection
async def sql_update_binary_pdf(
        publication_number: str,
        content: bytes,
        session: AsyncSession
) -> None:
    try:
        legislation_results = await session.execute(
            sa.select(DataLegislation)
            .where(DataLegislation.publication_number == publication_number)
        )

        legislation = legislation_results.scalar_one()
        legislation.binary_pdf = content
        await session.commit()

    except NoResultFound:
        config.logger.error(f"Legislation not found by publication_number: {publication_number}")

    except SQLAlchemyError as e:
        config.logger.error(f"Database error update binary_pdf: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error update binary_pdf: {e}")


# Выводим все законы, у которых нет текста, но есть binary_pdf
@connection
async def sql_get_legislation_by_have_binary_and_not_text(
        session: AsyncSession
) -> Sequence[DataLegislation]:
    try:
        legislation_results = await session.execute(
            sa.select(DataLegislation)
            .where(
                DataLegislation.binary_pdf != None,
                DataLegislation.text == None,
                DataLegislation.law_number == None
            )
            .order_by(DataLegislation.id)
        )

        legislation = legislation_results.scalars().all()
        return legislation

    except SQLAlchemyError as e:
        config.logger.error(f"Database error read legislation with binary_pdf and none text: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error read legislation with binary_pdf and none text: {e}")


# Записываем текст PDf файла
@connection
async def sql_update_text(
        publication_number: str,
        content: str,
        session: AsyncSession
) -> None:
    try:
        legislation_results = await session.execute(
            sa.select(DataLegislation)
            .where(DataLegislation.publication_number == publication_number)
        )

        legislation = legislation_results.scalar_one()
        legislation.text = content
        await session.commit()

    except NoResultFound:
        config.logger.error(f"Legislation not found by publication_number: {publication_number}")

    except SQLAlchemyError as e:
        config.logger.error(f"Database error update text: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error update text: {e}")


# Выводим текст PDF файла
@connection
async def sql_get_text_by_id(
        legislation_id: int,
        session: AsyncSession
) -> Optional[str]:
    try:
        legislation_results = await session.execute(
            sa.select(DataLegislation.text)
            .where(DataLegislation.id == legislation_id)
        )

        text = legislation_results.scalar_one()
        return text

    except NoResultFound:
        config.logger.error(f"Legislation not found by legislation_id: {legislation_id}")

    except SQLAlchemyError as e:
        config.logger.error(f"Database error read text by legislation_id {legislation_id}: {e}")

    except Exception as e:
        config.logger.error(f"Unexpected error read text by legislation_id {legislation_id}: {e}")

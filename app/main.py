# Внешние зависимости
from typing import Optional
import asyncio
# Внутренние модули
from app.database import setup_database
from app.parser import Parser
from app.parser_pdf import ParserPDF
from app.crud import (sql_add_new_legislation, sql_get_authorities_by_more_id,
                      sql_get_legislation_by_not_binary_pdf, sql_update_binary_pdf)
from app.config import get_config


config = get_config()


async def worker_parser_data(
    current_id: int
):
    await setup_database()

    authorities = await sql_get_authorities_by_more_id(current_id=current_id)

    for authority in authorities:
        config.logger.info(f"Начинаем работать с ({authority.id}) {authority.name}")

        parser = Parser(uuid_authority=authority.uuid_authority)
        results = await parser.async_run()

        config.logger.info(f"Всего данных получено: {len(results)}")
        for i, result in enumerate(results):
            config.logger.info(f"Записываем данные в таблицу. Итерация: {i+1}/{len(results)}")
            await sql_add_new_legislation(
                authority_id=authority.id,
                data=result
            )


async def worker_parser_pdf():
    all_legislation = await sql_get_legislation_by_not_binary_pdf()

    parser = ParserPDF()
    contents_binary = await parser.async_run(
        list_legislation=list(all_legislation[:100])
    )

    for i, publication_number in enumerate(contents_binary):
        config.logger.info(f"Обновляем данные в таблице. Итерация: {i + 1}/{len(contents_binary)}")
        await sql_update_binary_pdf(
            publication_number=publication_number,
            content=contents_binary[publication_number]
        )


def main(
        current_id: Optional[int] = None
):
    """
    if Optional is not None:
        asyncio.run(worker_parser_data(
            current_id=current_id
        ))
    """

    asyncio.run(worker_parser_pdf())
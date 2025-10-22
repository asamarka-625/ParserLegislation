# Внешние зависимости
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor
import os
import asyncio
# Внутренние модули
from app.database import setup_database
from app.parser import Parser
from app.parser_pdf import ParserPDF
from app.crud import (sql_add_new_legislation, sql_get_authorities_by_more_id,
                      sql_get_legislation_by_not_binary_pdf, sql_update_binary_pdf,
                      sql_get_legislation_by_have_binary_and_not_text, sql_update_text)
from app.config import get_config
from app.models import DataLegislation


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
    await setup_database()

    all_legislation = await sql_get_legislation_by_not_binary_pdf()
    batch_size = 300

    for batch_sart in range(0, len(all_legislation), batch_size):
        config.logger.info(f"Берем партию {batch_size} для запросов {batch_sart}/{len(all_legislation)}")

        batch_end = batch_sart + batch_size
        parser = ParserPDF()
        contents_binary = await parser.async_run(
            list_legislation=list(all_legislation[batch_sart:batch_end])
        )

        for i, data in enumerate(contents_binary):
            config.logger.info(f"Обновляем binary_pdf в таблице. Итерация: {i + 1}/{len(contents_binary)}")
            await sql_update_binary_pdf(
                publication_number=data[0],
                content=data[1]
            )


async def converter_multiprocess(all_legislation: Sequence[DataLegislation]):
    """Асинхронная функция с многопроцессорной обработкой"""

    # Синхронная обертка для асинхронной функции
    def process_batch_sync(batch_data):
        """Синхронная функция для обработки батча в процессе"""
        # Создаем новый event loop для каждого процесса
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_batch(batch_data))
        finally:
            loop.close()

    async def process_batch(batch_data):
        """Обработка одного батча данных"""
        convert = ParserPDF()
        contents_text = []

        for legislation in batch_data:
            text = convert.extract_text_from_pdf_bytes(legislation.binary_pdf)
            contents_text.append((legislation.publication_number, text))

        # Сохраняем результаты в базу
        for i, data in enumerate(contents_text):
            config.logger.info(f"Обновляем text в таблице. Итерация: {i + 1}/{len(contents_text)}")
            await sql_update_text(
                publication_number=data[0],
                content=data[1]
            )

        return len(contents_text)

    # Разбиваем данные на батчи для параллельной обработки
    cpu_count = os.cpu_count() or 4
    batch_size = max(1, len(all_legislation) // cpu_count)

    batches = []
    for i in range(0, len(all_legislation), batch_size):
        batch_end = i + batch_size
        batches.append(all_legislation[i:batch_end])

    config.logger.info(f"Запуск обработки {len(all_legislation)} документов в {len(batches)} процессах")

    # Запускаем многопроцессорную обработку
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        loop = asyncio.get_event_loop()

        # Создаем задачи для каждого батча
        tasks = [
            loop.run_in_executor(executor, process_batch_sync, batch)
            for batch in batches
        ]

        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем результаты
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                config.logger.error(f"Ошибка в батче {i}: {result}")
            else:
                successful += result

        config.logger.info(f"Успешно обработано: {successful} документов")


async def worker_convert_binary_to_text():
    """Основная рабочая функция"""
    await setup_database()

    all_legislation = await sql_get_legislation_by_have_binary_and_not_text()

    if not all_legislation:
        config.logger.info("Нет документов для обработки")
        return

    config.logger.info(f"Найдено {len(all_legislation)} документов для обработки")

    # Обрабатываем все документы с использованием многопроцессорности
    await converter_multiprocess(all_legislation)








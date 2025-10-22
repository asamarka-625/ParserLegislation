# Внешние зависимости
from concurrent.futures import ProcessPoolExecutor
import asyncio
# Внутренние модули
from app.database import setup_database
from app.parser import Parser
from app.parser_pdf import ParserPDF
from app.crud import (sql_add_new_legislation, sql_get_authorities_by_more_id,
                      sql_get_legislation_by_not_binary_pdf, sql_update_binary_pdf,
                      sql_get_legislation_by_have_binary_and_not_text, sql_update_text)
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
    await setup_database()

    while True:
        all_legislation = await sql_get_legislation_by_not_binary_pdf()
        batch_size = 300

        if len(all_legislation) == 0:
            break

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


# Функция для процесса
def process_doc(legislation):
    try:
        # Важно: импорты внутри функции для работы в multiprocessing
        from app.parser_pdf import ParserPDF

        parser = ParserPDF()
        text = parser.extract_text_from_pdf_bytes(legislation.binary_pdf)
        return (legislation.publication_number, text, None)

    except Exception as e:
        return (legislation.publication_number, None, str(e))


async def converter_absolute_simplest(all_legislation):
    """Самый простой вариант"""

    # Запускаем в 4 процессах
    with ProcessPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()

        # Запускаем все документы параллельно
        futures = [
            loop.run_in_executor(executor, process_doc, leg)
            for leg in all_legislation
        ]

        # Ждем все результаты
        results = await asyncio.gather(*futures)

        # Сохраняем в БД
        for pub_num, text, error in results:
            if text:
                await sql_update_text(publication_number=pub_num, content=text)
            elif error:
                config.logger.error(f"Ошибка {pub_num}: {error}")


async def worker_convert_binary_to_text_batch():
    docs = await sql_get_legislation_by_have_binary_and_not_text()

    if not docs:
        config.logger.info("Нет документов для обработки")
        return

    config.logger.info(f"Найдено {len(docs)} документов для обработки")

    # Обрабатываем батчами по 10 документов
    batch_size = 10
    total_batches = (len(docs) + batch_size - 1) // batch_size  # Округление вверх

    for batch_num in range(0, len(docs), batch_size):
        batch_end = batch_num + batch_size
        current_batch = docs[batch_num:batch_end]
        current_batch_num = (batch_num // batch_size) + 1

        config.logger.info(f"Обработка батча {current_batch_num}/{total_batches} ({len(current_batch)} документов)")

        await converter_absolute_simplest(current_batch)

        config.logger.info(f"Батч {current_batch_num} завершен")

    config.logger.info("Все документы обработаны")






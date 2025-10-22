# Внешние зависимости
import time
import os
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor
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


# Глобальные переменные для процессов
parser_instance = None


def worker_initializer():
    """Инициализация воркера (выполняется один раз при создании процесса)"""
    global parser_instance
    from app.parser_pdf import ParserPDF
    parser_instance = ParserPDF()
    print(f"🔄 Воркер {os.getpid()} инициализирован")


def process_doc_optimized(legislation):
    """Оптимизированная версия с глобальным парсером"""
    try:
        start_time = time.time()
        pid = os.getpid()

        # Используем глобальный парсер (переиспользуем в рамках процесса)
        text = parser_instance.extract_text_from_pdf_bytes(legislation.binary_pdf)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✅ Процесс {pid} завершил {legislation.publication_number} за {processing_time:.2f} сек")

        return (legislation.publication_number, text, None)

    except Exception as e:
        print(f"❌ Процесс {os.getpid()} ошибка в {legislation.publication_number}: {e}")
        return (legislation.publication_number, None, str(e))


async def converter_optimized(all_legislation):
    """Оптимизированная версия с пулом воркеров"""

    if not all_legislation:
        config.logger.info("Нет документов для обработки")
        return

    total_docs = len(all_legislation)
    # Используем минимум от количества CPU и количества документов
    max_workers = min(4, len(all_legislation), os.cpu_count() or 1)

    config.logger.info(f"🎯 Запуск обработки {total_docs} документов в {max_workers} процессах")

    start_time = time.time()

    # Создаем пул с инициализацией воркеров
    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_initializer
    ) as executor:
        loop = asyncio.get_event_loop()

        # Запускаем все задачи параллельно
        futures = [
            loop.run_in_executor(executor, process_doc_optimized, leg)
            for leg in all_legislation
        ]

        # Обрабатываем результаты по мере готовности
        successful = 0
        failed = 0
        processed = 0

        for future in asyncio.as_completed(futures):
            pub_num, text, error = await future

            if text:
                await sql_update_text(publication_number=pub_num, content=text)
                successful += 1
            else:
                config.logger.error(f"Ошибка {pub_num}: {error}")
                failed += 1

            processed += 1
            if processed % 10 == 0 or processed == total_docs:
                config.logger.info(f"Прогресс: {processed}/{total_docs}")

    total_time = time.time() - start_time
    config.logger.info(
        f"✅ Обработка завершена за {total_time:.2f} сек. "
        f"Успешно: {successful}, Ошибок: {failed}, "
        f"Скорость: {total_docs / total_time:.2f} док/сек"
    )


async def worker_convert_binary_to_text_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, default=1)
    parser.add_argument('--total-workers', type=int, default=1)
    args = parser.parse_args()

    # Каждый воркер обрабатывает свою часть данных
    all_docs = await sql_get_legislation_by_have_binary_and_not_text()

    if not all_docs:
        config.logger.info("Нет документов для обработки")
        return

    # Улучшенное распределение документов между воркерами
    total_workers = args.total_workers
    batch_size = len(all_docs) // total_workers

    # Определяем диапазон для этого воркера
    start_index = (args.worker_id - 1) * batch_size
    end_index = args.worker_id * batch_size if args.worker_id < total_workers else len(all_docs)

    docs_for_this_worker = all_docs[start_index:end_index]

    config.logger.info(
        f"Воркер {args.worker_id} обрабатывает {len(docs_for_this_worker)} документов из {len(all_docs)}")

    await converter_optimized(docs_for_this_worker)




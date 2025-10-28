# Внешние зависимости
import asyncio
import multiprocessing as mp
import hashlib
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


def get_worker_for_document(doc, worker_id):
    """Определяет, должен ли этот воркер обрабатывать документ"""
    # Создаем стабильный хэш на основе publication_number
    doc_hash = hashlib.md5(doc.publication_number.encode()).hexdigest()
    doc_int = int(doc_hash, 16)
    assigned_worker = (doc_int % config.TOTAL_WORKERS) + 1
    return assigned_worker == worker_id


async def process_single_document(
        pdf_parser: ParserPDF,
        doc: DataLegislation,
        worker_id: int
):
    """Асинхронная обработка одного документа"""
    try:
        text = pdf_parser.extract_text_from_pdf_bytes(doc.binary_pdf)

        if text:
            # Сохраняем в БД
            await sql_update_text(
                publication_number=doc.publication_number,
                content=text
            )
            return True
        return False

    except Exception as e:
        config.logger.error(f"Worker {worker_id}: ошибка обработки {doc.publication_number}: {e}")
        return False


def worker_process(worker_id, documents):
    """СИНХРОННАЯ обработка документов в отдельном процессе"""
    # Создаем новый event loop для процесса
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pdf_parser = ParserPDF()

    # Главная синхронная функция
    processed = 0
    errors = 0

    for i, doc in enumerate(documents):
        try:
            config.logger.info(f"""
            Документ: [{doc.publication_number}] Worker {worker_id}: прогресс {processed + errors}/{len(documents)}
            """)

            # Запускаем асинхронную обработку
            success = loop.run_until_complete(process_single_document(
                pdf_parser=pdf_parser,
                doc=doc,
                worker_id=worker_id
            ))

            if success:
                processed += 1
            else:
                errors += 1

            # Логируем прогресс
            if (processed + errors) % 10 == 0:
                config.logger.info(f"Worker {worker_id}: прогресс {processed + errors}/{len(documents)}")

        except Exception as e:
            config.logger.error(f"Worker {worker_id}: критическая ошибка {doc.publication_number}: {e}")
            errors += 1

    loop.close()
    config.logger.info(f"Worker {worker_id} завершил: {processed} успешно, {errors} ошибок")
    return processed, errors


async def worker_convert_binary_to_text_batch():
    # Получаем все документы
    all_documents = await sql_get_legislation_by_have_binary_and_not_text()

    if not all_documents:
        config.logger.info("Нет документов для обработки")
        return

    config.logger.info(f"Найдено {len(all_documents)} документов для обработки")

    worker_tasks = {i: [] for i in range(1, config.TOTAL_WORKERS + 1)}

    for doc in all_documents:
        for worker_id in range(1, config.TOTAL_WORKERS + 1):
            if get_worker_for_document(doc, worker_id):
                worker_tasks[worker_id].append(doc)
                break

    # Запускаем процессы
    with mp.get_context("spawn").Pool(processes=config.TOTAL_WORKERS) as pool:
        # Создаем задачи для каждого воркера
        tasks = []
        for worker_id in range(1, config.TOTAL_WORKERS + 1):
            if worker_tasks[worker_id]:
                task = pool.apply_async(
                    worker_process,
                    (worker_id, worker_tasks[worker_id])
                )
                tasks.append(task)

        # Собираем результаты
        total_processed = 0
        total_errors = 0

        for worker_id, task in tasks:
            try:
                processed, errors = task.get(timeout=3600 * 5)  # Таймаут 5 часов
                total_processed += processed
                total_errors += errors
                config.logger.info(f"Worker {worker_id} завершил: {processed} успешно, {errors} ошибок")

            except mp.TimeoutError:
                config.logger.error(f"Worker {worker_id}: таймаут выполнения")

            except Exception as e:
                config.logger.error(f"Ошибка в процессе {worker_id}: {e}")

        config.logger.info(f"Итог: {total_processed} успешно, {total_errors} ошибок")



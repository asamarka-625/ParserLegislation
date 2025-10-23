# Внешние зависимости
import argparse
import hashlib
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


def get_worker_for_document(doc, total_workers, worker_id):
    """Определяет, должен ли этот воркер обрабатывать документ"""
    # Создаем стабильный хэш на основе publication_number
    doc_hash = hashlib.md5(doc.publication_number.encode()).hexdigest()
    doc_int = int(doc_hash, 16)
    assigned_worker = (doc_int % total_workers) + 1
    return assigned_worker == worker_id


async def worker_convert_binary_to_text_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, default=1)
    parser.add_argument('--total-workers', type=int, default=1)
    args = parser.parse_args()

    # Каждый воркер обрабатывает свою часть данных
    all_legislation = await sql_get_legislation_by_have_binary_and_not_text()

    if not all_legislation:
        config.logger.info("Нет документов для обработки")
        return

    legislation_for_this_worker = [
        doc for doc in all_legislation
        if get_worker_for_document(doc, args.total_workers, args.worker_id)
    ]

    config.logger.info(
        f"Воркер {args.worker_id} обрабатывает {len(legislation_for_this_worker)} документов из {len(all_legislation)}")

    # СОЗДАЕМ ОДИН ЭКЗЕМПЛЯР ПАРСЕРА для переиспользования
    pdf_parser = ParserPDF()
    processed = 0
    errors = 0

    for legislation in legislation_for_this_worker:
        try:
            text = pdf_parser.extract_text_from_pdf_bytes(legislation.binary_pdf)

            if text:
                await sql_update_text(
                    publication_number=legislation.publication_number,
                    content=text
                )
                processed += 1
            else:
                config.logger.warning(f"Пустой текст для {legislation.publication_number}")
                errors += 1

            # Логируем прогресс
            if (processed + errors) % 10 == 0:
                config.logger.info(f"Воркер {args.worker_id}: прогресс {processed + errors}/{len(legislation_for_this_worker)}")

        except Exception as e:
            config.logger.error(f"Ошибка обработки {legislation.publication_number}: {e}")
            errors += 1

    config.logger.info(f"Воркер {args.worker_id} завершил: {processed} успешно, {errors} ошибок")




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


def process_documents_batch(documents_batch):
    """Обработка батча документов в отдельном процессе"""
    try:
        convert = ParserPDF()
        results = []

        for doc_data in documents_batch:
            try:
                text = convert.extract_text_from_pdf_bytes(doc_data['binary_pdf'])
                results.append({
                    'publication_number': doc_data['publication_number'],
                    'content': text,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'publication_number': doc_data['publication_number'],
                    'content': None,
                    'success': False,
                    'error': str(e)
                })

        return results
    except Exception as e:
        # Возвращаем ошибки для всего батча
        return [{
            'publication_number': 'batch_error',
            'content': None,
            'success': False,
            'error': f"Batch processing error: {str(e)}"
        }]


async def converter_multiprocess_batch(all_legislation: Sequence[DataLegislation]):
    """Версия с обработкой целых батчей в процессах"""

    # Подготавливаем данные
    legislation_data_list = [
        {
            'publication_number': leg.publication_number,
            'binary_pdf': leg.binary_pdf
        }
        for leg in all_legislation
    ]

    # Разбиваем на батчи
    cpu_count = os.cpu_count() or 4
    batch_size = max(10, len(legislation_data_list) // cpu_count)

    batches = []
    for i in range(0, len(legislation_data_list), batch_size):
        batch_end = i + batch_size
        batches.append(legislation_data_list[i:batch_end])

    config.logger.info(f"Запуск обработки {len(all_legislation)} документов в {len(batches)} батчах")

    # Обрабатываем батчи параллельно
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        loop = asyncio.get_event_loop()

        # Запускаем обработку каждого батча
        tasks = [
            loop.run_in_executor(executor, process_documents_batch, batch)
            for batch in batches
        ]

        # Собираем результаты
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем все результаты
        total_successful = 0
        total_failed = 0

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                config.logger.error(f"Ошибка в батче {i}: {result}")
                total_failed += len(batches[i]) if i < len(batches) else 1
                continue

            # Сохраняем успешные результаты
            for doc_result in result:
                if doc_result['success']:
                    await sql_update_text(
                        publication_number=doc_result['publication_number'],
                        content=doc_result['content']
                    )
                    total_successful += 1
                else:
                    config.logger.error(
                        f"Ошибка обработки {doc_result['publication_number']}: {doc_result.get('error')}")
                    total_failed += 1

        config.logger.info(f"Обработка завершена. Успешно: {total_successful}, Ошибок: {total_failed}")


async def worker_convert_binary_to_text_batch():
    await setup_database()
    all_legislation = await sql_get_legislation_by_have_binary_and_not_text()

    if not all_legislation:
        config.logger.info("Нет документов для обработки")
        return

    await converter_multiprocess_batch(all_legislation)







# Внешние зависимости
import psutil
import os
import resource
from typing import Optional, List, Tuple
import asyncio
from PIL import Image
from pdf2image import convert_from_bytes
from fake_useragent import UserAgent
import httpx
from paddleocr import PaddleOCR
import torch
import numpy as np
# Внутренние модули
from app.config import get_config
from app.models import DataLegislation


config = get_config()


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    config.logger.info(f"=== ИСПОЛЬЗОВАНИЕ ПАМЯТИ ===")
    config.logger.info(f"RSS (физическая): {memory_info.rss / 1024 / 1024:.2f} MB")
    config.logger.info(f"VMS (виртуальная): {memory_info.vms / 1024 / 1024:.2f} MB")
    config.logger.info(f"Пиковое RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB")

    # Общая память системы
    system_memory = psutil.virtual_memory()
    config.logger.info(f"Всего памяти: {system_memory.total / 1024 / 1024:.2f} MB")
    config.logger.info(f"Использовано: {system_memory.used / 1024 / 1024:.2f} MB")
    config.logger.info(f"Свободно: {system_memory.available / 1024 / 1024:.2f} MB")


class ParserPDF:
    def __init__(self):
        self.url_base = f"http://publication.pravo.gov.ru/file/pdf?eoNumber="
        self.ua = UserAgent()
        self.headers = {
            "Accept": "text / html, application / xhtml + xml, application / xml;q = 0.9, * / *;q = 0.8",
            "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
            "Host": "publication.pravo.gov.ru",
            "Referer": "http://publication.pravo.gov.ru/documents/monthly",
            "User-Agent": self.ua.random
        }

    def init_gpu(self):
        """Инициализация PaddleOCR с GPU"""
        try:
            use_gpu = torch.cuda.is_available()
            self.ocr = PaddleOCR(
                lang='ru',
                use_angle_cls=False,  # определение ориентации текста
                # det=True,  # детекция текста (включено по умолчанию)
                # rec=True,  # распознавание текста (включено по умолчанию)
                # cls=True,  # классификация ориентации

                # Пути к моделям (опционально)
                # det_model_dir='path/to/det/model',
                # rec_model_dir='path/to/rec/model',
                # cls_model_dir='path/to/cls/model',

                # Производительность
                use_gpu=True,  # использовать GPU если доступно

                # Параметры детекции
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                det_db_unclip_ratio=1.5,

                # Параметры распознавания
                rec_batch_num=6,
                drop_score=0.5  # минимальный порог уверенности
            )
            config.logger.info(f"PaddleOCR инициализирован, GPU: {use_gpu}")

            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                config.logger.info(f"Используется GPU: {gpu_name}")

        except Exception as e:
            config.logger.error(f"Ошибка инициализации PaddleOCR: {e}")
            self.ocr = None

    async def get_async_response(
            self,
            url: str,
            publication_number: str,
            client: httpx.AsyncClient
    ) -> bytes:
        config.logger.info(f"Делаем запрос на: {url}")
        try:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            pdf_content = response.content

            config.logger.info(f"PDF ({publication_number}) успешно скачан, размер: {len(pdf_content)} байт")
            return pdf_content

        except httpx.ReadTimeout:
            config.logger.error(f"ReadTimeout для {url}")
            raise

        except httpx.ConnectTimeout:
            config.logger.error(f"ConnectTimeout для {url}")
            raise

        except httpx.HTTPStatusError as err:
            config.logger.error(f"HTTPError {err.response.status_code} для {url}")
            raise

        except Exception as err:
            config.logger.error(f"Неожиданная ошибка для {url}: {type(err).__name__}: {err}")
            raise

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        try:
            # Проверяем что OCR инициализирован
            if self.ocr is None:
                config.logger.error("PaddleOCR не инициализирован")
                return None

            ocr_text = self._extract_text_with_ocr_from_bytes(pdf_bytes)
            return ocr_text

        except Exception as e:
            config.logger.error(f"Ошибка при извлечении текста: {e}")
            return None

    def _extract_text_with_ocr_from_bytes(self, pdf_bytes: bytes) -> str:
        """Извлечение текста с помощью OCR из PDF байтов"""
        try:
            config.logger.info("Конвертация PDF в изображения...")

            # Конвертируем PDF байты в изображения
            images = convert_from_bytes(
                pdf_bytes,
                dpi=300,
                fmt='JPEG'
            )

            all_text = ""

            for i, image in enumerate(images):
                config.logger.info(f"Обработка страницы {i + 1}/{len(images)}...")

                # Обрабатываем изображение
                page_text = self._process_image_async(image, i)
                all_text += f"\n--- Страница {i + 1} ---\n{page_text}"

            return all_text

        except Exception as e:
            config.logger.error(f"Ошибка OCR: {e}")
            return ""

    def _process_image_async(self, image: Image.Image, page_num: int) -> str:
        """Обработка одного изображения"""
        try:
            # Предобработка изображения
            processed_image = self._preprocess_image(image)

            # OCR распознавание
            page_text = self._perform_ocr_paddle(processed_image, page_num)
            return page_text

        except Exception as e:
            config.logger.error(f"Ошибка обработки страницы {page_num + 1}: {e}")
            return ""

    @staticmethod
    def _preprocess_image(image: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения распознавания"""
        # PaddleOCR лучше работает с RGB, а не grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _perform_ocr_paddle(self, image: Image.Image, page_num: int) -> str:
        """PaddleOCR с GPU ускорением"""
        try:
            # Конвертируем PIL Image в numpy array
            image_np = np.array(image)

            # ✅ Проверяем что изображение не пустое
            if image_np.size == 0:
                config.logger.warning(f"Пустое изображение на странице {page_num + 1}")
                return ""

            # Распознавание текста
            result = self.ocr.ocr(image_np, cls=True)

            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                return '\n'.join(texts)
            else:
                config.logger.info(f"Текст не найден на странице {page_num + 1}")
                return ""

        except Exception as e:
            config.logger.error(f"PaddleOCR ошибка на странице {page_num + 1}: {e}")
            return ""


    async def async_run(
            self,
            list_legislation: List[DataLegislation]
    ) -> List[Tuple[str, bytes]]:
        # Создаем клиент для каждой пачки
        timeout = httpx.Timeout(
            connect=30.0,  # Таймаут на подключение
            read=60.0,  # Таймаут на чтение
            write=30.0,  # Таймаут на запрос
            pool=100.0  # Таймаут на получение из пула
        )
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        batch_size = 45
        results = []

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for batch_start in range(0, len(list_legislation), batch_size):
                batch_end = batch_start + batch_size
                current_batch = list_legislation[batch_start:batch_end]

                config.logger.info(f"Обрабатываем батч {batch_start}-{batch_end} из {len(list_legislation)}")

                tasks = []
                for legislation in current_batch:
                    task = self.get_async_response(
                        url=f"{self.url_base}{legislation.publication_number}",
                        publication_number=legislation.publication_number,
                        client=client
                    )
                    tasks.append(task)

                # Добавляем timeout для всего батча
                try:
                    batch_contents = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=60.0  # 60 секунд на батч
                    )
                except asyncio.TimeoutError:
                    config.logger.error(f"Таймаут батча {batch_start}-{batch_end}")
                    continue

                # Обрабатываем результаты пачки
                successful = 0
                failed = 0

                for legislation, content in zip(current_batch, batch_contents):
                    if isinstance(content, Exception):
                        config.logger.error(
                            f"Не удалась загрузить PDF файл (publication_number: {legislation.publication_number}): {content}"
                        )
                        failed += 1
                        continue

                    if not content:
                        config.logger.error(
                            f"Пустой контент для PDF (publication_number: {legislation.publication_number})"
                        )
                        failed += 1
                        continue

                    results.append((legislation.publication_number, content))
                    successful += 1

                config.logger.info(f"Батч {batch_start}-{batch_end} завершен: {successful} успешно, {failed} ошибок")
                get_memory_usage()

        return results


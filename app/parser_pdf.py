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
import easyocr
import torch
import numpy as np
import cv2
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
        """Инициализация EasyOCR с GPU"""
        try:
            use_gpu = torch.cuda.is_available()

            # Инициализация EasyOCR для русского языка
            self.reader = easyocr.Reader(
                ['ru', 'en'],  # русский и английский языки
                gpu=use_gpu,
                download_enabled=True
            )

            config.logger.info(f"EasyOCR инициализирован, GPU: {use_gpu}")

            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                config.logger.info(f"Используется GPU: {gpu_name}")

        except Exception as e:
            config.logger.error(f"Ошибка инициализации EasyOCR: {e}")
            self.reader = None

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Основной метод извлечения текста из PDF"""
        try:
            if self.reader is None:
                config.logger.error("EasyOCR не инициализирован")
                return None

            return self._extract_text_with_ocr_from_bytes(pdf_bytes)

        except Exception as e:
            config.logger.error(f"Ошибка при извлечении текста: {e}")
            return None

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

    def _extract_text_with_ocr_from_bytes(self, pdf_bytes: bytes) -> str:
        """Конвертация PDF и обработка каждой страницы"""
        try:
            config.logger.info("Конвертация PDF в изображения...")

            # Конвертация PDF в изображения с высоким DPI для сканов
            images = convert_from_bytes(
                pdf_bytes,
                dpi=300,  # Высокий DPI для качественных сканов
                fmt='JPEG'
            )

            all_text = []

            for i, image in enumerate(images):
                config.logger.info(f"Обработка страницы {i + 1}/{len(images)}...")

                # Оптимизация изображения для OCR
                processed_image = self._optimize_image_for_ocr(image)

                # Распознавание текста
                page_text = self._perform_ocr_easy(processed_image, i)
                if page_text.strip():
                    all_text.append(f"--- Страница {i + 1} ---\n{page_text}")

            return "\n\n".join(all_text)

        except Exception as e:
            config.logger.error(f"Ошибка OCR: {e}")
            return ""

    def _optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Оптимизация изображения для улучшения распознавания сканов"""
        try:
            # Конвертация в grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Увеличение разрешения для мелкого текста
            original_width, original_height = image.size

            # Увеличиваем изображение если оно слишком маленькое для скана
            if max(original_width, original_height) < 2000:
                scale_factor = 2000 / max(original_width, original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Улучшение контраста и резкости
            img_np = np.array(image)

            # Контрастное растяжение
            min_val = np.percentile(img_np, 2)
            max_val = np.percentile(img_np, 98)
            img_contrast = np.clip((img_np - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)

            # Легкое шумоподавление
            img_denoised = cv2.medianBlur(img_contrast, 3)

            # Увеличение резкости
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img_sharp = cv2.filter2D(img_denoised, -1, kernel)

            return Image.fromarray(img_sharp)

        except Exception as e:
            config.logger.error(f"Ошибка оптимизации изображения: {e}")
            return image

    def _perform_ocr_easy(self, image: Image.Image, page_num: int) -> str:
        """Распознавание текста с помощью EasyOCR"""
        try:
            # Конвертируем в RGB для EasyOCR
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)

            # Распознавание текста с параметрами для сканов
            results = self.reader.readtext(
                image_np,
                batch_size=4,  # Оптимально для GPU
                paragraph=True,  # Группировка в параграфы
                detail=1,
                contrast_ths=0.3,  # Порог контраста
                adjust_contrast=0.7,  # Автоконтраст
                width_ths=0.8  # Ширина для объединения текста
            )

            return self._parse_easyocr_results(results, page_num)

        except Exception as e:
            config.logger.error(f"EasyOCR ошибка на странице {page_num + 1}: {e}")
            return ""

    def _parse_easyocr_results(self, results, page_num: int) -> str:
        """Обработка и фильтрация результатов EasyOCR"""
        if not results:
            config.logger.info(f"Текст не найден на странице {page_num + 1}")
            return ""

        try:
            valid_lines = []
            total_confidence = 0

            for (bbox, text, confidence) in results:
                # Фильтрация по уверенности и длине текста
                if confidence >= 0.6 and len(text.strip()) >= 2:
                    valid_lines.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    total_confidence += confidence

            if valid_lines:
                # Восстановление структуры документа
                final_text = self._reconstruct_document_structure(valid_lines)

                avg_confidence = total_confidence / len(valid_lines)
                config.logger.info(
                    f"Страница {page_num + 1}: {len(valid_lines)} строк, уверенность: {avg_confidence:.3f}")

                return final_text
            else:
                config.logger.info(f"На странице {page_num + 1} не найдено качественного текста")
                return ""

        except Exception as e:
            config.logger.error(f"Ошибка парсинга результатов EasyOCR: {e}")
            return ""

    def _reconstruct_document_structure(self, lines_data: List[dict]) -> str:
        """Восстановление структуры документа с учетом layout"""
        if not lines_data:
            return ""

        try:
            # Сортируем строки по вертикальной позиции
            lines_data.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))

            paragraphs = []
            current_block = []
            previous_bottom = None

            for line in lines_data:
                bbox = line['bbox']
                current_top = bbox[0][1]

                # Определяем новый блок если большой вертикальный отступ
                if previous_bottom is not None and current_top - previous_bottom > 25:
                    if current_block:
                        paragraph_text = ' '.join(current_block)
                        paragraphs.append(paragraph_text)
                        current_block = []

                current_block.append(line['text'])
                previous_bottom = bbox[2][1]  # Нижняя координата

            # Добавляем последний блок
            if current_block:
                paragraph_text = ' '.join(current_block)
                paragraphs.append(paragraph_text)

            return '\n\n'.join(paragraphs)

        except Exception as e:
            config.logger.error(f"Ошибка восстановления структуры: {e}")
            # Возвращаем простой объединенный текст в случае ошибки
            return ' '.join(item['text'] for item in lines_data)

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


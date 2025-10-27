# Внешние зависимости
import re
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

    def init_gpu(self):
        """Инициализация EasyOCR с GPU"""
        try:
            use_gpu = torch.cuda.is_available()

            # Инициализация EasyOCR для русского языка с оптимизацией для GPU
            self.reader = easyocr.Reader(
                ['ru', 'en'],  # русский и английский языки
                gpu=use_gpu,
                download_enabled=True,
                model_storage_directory=None,
                user_network_directory=None
            )

            config.logger.info(f"EasyOCR инициализирован, GPU: {use_gpu}")

            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                config.logger.info(f"Используется GPU: {gpu_name}")
                # Оптимизация для CUDA
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

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

    def _extract_text_with_ocr_from_bytes(self, pdf_bytes: bytes) -> str:
        """Конвертация PDF и обработка каждой страницы с оптимизацией для GPU"""
        try:
            config.logger.info("Конвертация PDF в изображения...")

            # Конвертация PDF в изображения с оптимизированными параметрами
            images = convert_from_bytes(
                pdf_bytes,
                dpi=250,  # Снизил DPI для скорости, качество остается хорошим
                fmt='JPEG',
                thread_count=4,  # Увеличил количество потоков
                use_pdftocairo=True,  # Более быстрый рендерер
                strict=False
            )

            all_text = []

            # Обрабатываем все изображения последовательно для максимальной скорости GPU
            for i, image in enumerate(images):
                config.logger.info(f"Обработка страницы {i + 1}/{len(images)}...")

                # Быстрая оптимизация изображения
                processed_image = self._fast_optimize_image(image)

                # Распознавание текста с увеличенным batch_size
                page_text = self._perform_ocr_easy_fast(processed_image, i)
                if page_text.strip():
                    all_text.append(f"--- Страница {i + 1} ---\n{page_text}")

            return "\n\n".join(all_text)

        except Exception as e:
            config.logger.error(f"Ошибка OCR: {e}")
            return ""

    def _fast_optimize_image(self, image: Image.Image) -> Image.Image:
        """Быстрая оптимизация изображения для GPU"""
        try:
            # Быстрая конвертация в grayscale
            if image.mode != 'L':
                image = image.convert('L')

            original_width, original_height = image.size

            # Быстрое увеличение разрешения только если действительно нужно
            if max(original_width, original_height) < 1600:  # Снизил порог
                scale_factor = min(2000 / max(original_width, original_height), 2.0)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Быстрая обработка через OpenCV
            img_np = np.array(image)

            # Быстрое улучшение контраста
            img_contrast = cv2.convertScaleAbs(img_np, alpha=1.2, beta=0)

            # Быстрое шумоподавление
            img_denoised = cv2.medianBlur(img_contrast, 1)  # Уменьшил ядро

            return Image.fromarray(img_denoised)

        except Exception as e:
            config.logger.error(f"Ошибка оптимизации изображения: {e}")
            return image

    def _perform_ocr_easy_fast(self, image: Image.Image, page_num: int) -> str:
        """Быстрое распознавание текста с оптимизацией для GPU"""
        try:
            # Быстрая конвертация в RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)

            # Оптимизированные параметры для мощной видеокарты
            results = self.reader.readtext(
                image_np,
                batch_size=16,  # Увеличил для заполнения GPU
                paragraph=False,  # Отключил для скорости, обработаем сами
                detail=1,
                contrast_ths=0.1,  # Снизил пороги для большей чувствительности
                adjust_contrast=0.5,
                width_ths=0.7,
                decoder='greedy',  # Более быстрый декодер
                beamWidth=1,  # Минимальное значение для скорости
                min_size=2,  # Минимальный размер текста
                text_threshold=0.4,  # Более низкий порог
                link_threshold=0.4,
                mag_ratio=1.0  # Без увеличения
            )

            return self._parse_easyocr_results_fast(results, page_num)

        except Exception as e:
            config.logger.error(f"EasyOCR ошибка на странице {page_num + 1}: {e}")
            return ""

    def _parse_easyocr_results_fast(self, results, page_num: int) -> str:
        """Быстрая обработка результатов EasyOCR"""
        if not results:
            config.logger.info(f"Текст не найден на странице {page_num + 1}")
            return ""

        try:
            valid_lines = []
            total_confidence = 0

            for result in results:
                try:
                    if len(result) == 3:
                        bbox, text, confidence = result
                    elif len(result) == 2:
                        bbox, text = result
                        confidence = 0.7  # Понизил значение по умолчанию
                    else:
                        continue

                    text = str(text).strip() if text else ""

                    # Быстрая замена символов
                    text = self._fast_replace_symbols(text)

                    # Более низкий порог уверенности для скорости
                    if confidence >= 0.5 and len(text) >= 1:  # Снизил требования
                        valid_lines.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        total_confidence += confidence

                except Exception as e:
                    continue  # Пропускаем ошибки для скорости

            if valid_lines:
                # Быстрое восстановление структуры
                final_text = self._fast_reconstruct_structure(valid_lines)

                avg_confidence = total_confidence / len(valid_lines)
                config.logger.info(
                    f"Страница {page_num + 1}: {len(valid_lines)} строк, уверенность: {avg_confidence:.3f}")

                return final_text
            else:
                return ""

        except Exception as e:
            config.logger.error(f"Ошибка парсинга результатов EasyOCR: {e}")
            return ""

    @staticmethod
    def _fast_replace_symbols(text: str) -> str:
        """Быстрая замена символов Ng и N на №"""
        if not text:
            return text

        # Быстрые строковые замены вместо regex
        text = text.replace('Ng ', '№ ')
        text = text.replace('N ', '№ ')
        text = text.replace('Ng-', '№-')
        text = text.replace('N-', '№-')

        # Быстрая замена через цикл для комбинаций
        words = text.split()
        for i, word in enumerate(words):
            if word.upper() in ['NG', 'N'] and i + 1 < len(words) and words[i + 1].isdigit():
                words[i] = '№'
            elif word.startswith('Ng') and word[2:].isdigit():
                words[i] = '№ ' + word[2:]
            elif word.startswith('N') and word[1:].isdigit():
                words[i] = '№ ' + word[1:]

        return ' '.join(words)

    @staticmethod
    def _fast_reconstruct_structure(lines_data: List[dict]) -> str:
        """Быстрое восстановление структуры документа"""
        if not lines_data:
            return ""

        try:
            # Быстрая сортировка
            lines_data.sort(key=lambda x: x['bbox'][0][1])

            paragraphs = []
            current_block = []
            previous_bottom = None

            for line in lines_data:
                bbox = line['bbox']
                current_top = bbox[0][1]

                if previous_bottom is not None and current_top - previous_bottom > 20:  # Уменьшил отступ
                    if current_block:
                        paragraph_text = ' '.join(current_block)
                        paragraphs.append(paragraph_text)
                        current_block = []

                current_block.append(line['text'])
                previous_bottom = bbox[2][1]

            if current_block:
                paragraph_text = ' '.join(current_block)
                paragraphs.append(paragraph_text)

            return '\n\n'.join(paragraphs)

        except Exception as e:
            # Возвращаем простой текст в случае ошибки
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


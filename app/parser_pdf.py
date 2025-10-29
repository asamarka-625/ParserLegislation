# Внешние зависимости
import re
import psutil
import os
import resource
from typing import Optional, List, Tuple
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty
from multiprocessing import Manager
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

    def init_gpu(self, y_tolerance: int = 50):
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

            self.y_tolerance = y_tolerance

        except Exception as e:
            config.logger.error(f"Ошибка инициализации EasyOCR: {e}")
            self.reader = None

    def conveyor_extract_text_from_pdf_bytes(self, data, max_workers: int = 4):
        def preprocess_worker(input_queue, output_queue):
            try:
                while True:
                    pdf_bytes = input_queue.get()
                    if pdf_bytes is None:  # Сигнал остановки
                        output_queue.put(None)
                        break

                    images = convert_from_bytes(
                        pdf_bytes,
                        dpi=250,  # Снизил DPI для скорости, качество остается хорошим
                        fmt='JPEG',
                        thread_count=4,  # количество потоков
                        use_pdftocairo=True,  # Более быстрый рендерер
                        strict=False
                    )
                    for page_num, image in enumerate(images):
                        processed = self._fast_optimize_image(image)
                        output_queue.put({
                            'page_num': page_num,  # Номер страницы в PDF
                            'image': np.array(processed)
                        })
            except Exception as e:
                config.logger.error(f"Ошибка в preprocess_worker: {e}")
                output_queue.put(None)

        def ocr_worker(input_queue, output_queue):
            try:
                while True:
                    data_ = input_queue.get()
                    if data_ is None:
                        output_queue.put(None)
                        break

                    processed_image = data_['image']

                    results = self.reader.readtext(
                        processed_image,
                        batch_size=16,  # Сколько частиц изображения обрабатывать за один раз
                        paragraph=False, # Группировать слова в абзацы автоматически
                        detail=1, # Возвращать полную информацию (координаты + уверенность) detail=0 - только текст (быстрее)
                        contrast_ths=0.3,  # Порог контраста для обнаружения текста
                        adjust_contrast=0.5, # Насколько усиливать контраст изображения 0.5 - среднее усиление
                        width_ths=0.7, # Максимальная ширина для объединения слов в строку
                        decoder='greedy',  # Алгоритм преобразования нейросетевых данных в текст: greedy: Быстрый, но менее точный; beamsearch: Медленнее, но точнее
                        # beamWidth=2, Сколько вариантов текста рассматривать (только для beamsearch) При greedy: Игнорируется
                        min_size=10,  # Минимальный размер текста для распознавания (в пикселях)
                        text_threshold=0.6,  # Минимальная уверенность что это текст
                        link_threshold=0.5, # Уверенность для соединения символов в слова
                        mag_ratio=1.0,  # Без увеличения
                        slope_ths=0.1, # Допустимый наклон текста (в радианах) 0.1 - небольшой наклон разрешен
                        ycenter_ths=0.5, # Допуск по вертикали для объединения в строки
                        height_ths=0.5, # Допуск по высоте текста для объединения
                        add_margin=0.02 # Добавлять поля вокруг текста (2% от размера)
                    )
                    output_queue.put({
                        'page_num': data_['page_num'],
                        'results': results
                    })

            except Exception as e:
                config.logger.error(f"Ошибка в ocr_worker: {e}")
                output_queue.put(None)

        def reconstruct_worker(input_queue, output_queue):
            try:
                while True:
                    ocr_results = input_queue.get()
                    if ocr_results is None:
                        break

                    text = self.reconstruct_text(ocr_results['results'])
                    output_queue.put({
                        'page_num': ocr_results['page_num'],
                        'text': text
                    })

            except Exception as e:
                config.logger.error(f"Ошибка в reconstruct_worker: {e}")

        manager = Manager()
        raw_queue = manager.Queue()
        processed_queue = manager.Queue()
        ocr_queue = manager.Queue()
        result_queue = manager.Queue()

        # Запускаем воркеры
        with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            with ProcessPoolExecutor(max_workers=max_workers) as process_executor:
                # Заполняем начальную очередь
                for pdf_bytes in data:
                    raw_queue.put(pdf_bytes)

                # Добавляем сигналы остановки
                for _ in range(max_workers):
                    raw_queue.put(None)

                # Запускаем конвейер
                preprocess_futures = [process_executor.submit(preprocess_worker, raw_queue, processed_queue)
                                      for _ in range(max_workers)]
                ocr_futures = [thread_executor.submit(ocr_worker, processed_queue, ocr_queue)
                               for _ in range(max_workers)]
                reconstruct_futures = [process_executor.submit(reconstruct_worker, ocr_queue, result_queue)
                                       for _ in range(max_workers)]

                for future in preprocess_futures:
                    future.result()

                for future in ocr_futures:
                    future.result()

                for future in reconstruct_futures:
                    future.result()

        # Собираем ВСЕ результаты
        all_pages = []
        while True:
            try:
                # Ждем все результаты с таймаутом
                result = result_queue.get_nowait()
                if result:
                    all_pages.append(result)
            except Empty:
                break  # Больше нет результатов

        # Группируем и сортируем результаты
        return "\n".join(
            r["text"] for r in sorted(all_pages, key=lambda x: x["page_num"])
        )

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
                    all_text.append(page_text)

            return "\n".join(all_text)

        except Exception as e:
            config.logger.error(f"Ошибка OCR: {e}")
            return ""

    def _fast_optimize_image(self, image: Image.Image) -> Image.Image:
        """Быстрая оптимизация изображения для GPU"""
        try:
            # Быстрая конвертация в RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

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
            image_np = np.array(image)

            # Оптимизированные параметры для мощной видеокарты
            results = self.reader.readtext(
                image_np,
                batch_size=16,  # Сколько частиц изображения обрабатывать за один раз
                paragraph=False, # Группировать слова в абзацы автоматически
                detail=1, # Возвращать полную информацию (координаты + уверенность) detail=0 - только текст (быстрее)
                contrast_ths=0.3,  # Порог контраста для обнаружения текста
                adjust_contrast=0.5, # Насколько усиливать контраст изображения 0.5 - среднее усиление
                width_ths=0.7, # Максимальная ширина для объединения слов в строку
                decoder='greedy',  # Алгоритм преобразования нейросетевых данных в текст: greedy: Быстрый, но менее точный; beamsearch: Медленнее, но точнее
                # beamWidth=2, Сколько вариантов текста рассматривать (только для beamsearch) При greedy: Игнорируется
                min_size=10,  # Минимальный размер текста для распознавания (в пикселях)
                text_threshold=0.6,  # Минимальная уверенность что это текст
                link_threshold=0.5, # Уверенность для соединения символов в слова
                mag_ratio=1.0,  # Без увеличения
                slope_ths=0.1, # Допустимый наклон текста (в радианах) 0.1 - небольшой наклон разрешен
                ycenter_ths=0.5, # Допуск по вертикали для объединения в строки
                height_ths=0.5, # Допуск по высоте текста для объединения
                add_margin=0.02 # Добавлять поля вокруг текста (2% от размера)
            )

            return self.reconstruct_text(results)

        except Exception as e:
            config.logger.error(f"EasyOCR ошибка на странице {page_num + 1}: {e}")
            return ""

    def reconstruct_text(self, result) -> str:
        """
        Основной метод для восстановления текста из координат слов
        """
        # Группируем слова в строки
        lines = self.group_into_lines(result)

        reconstructed_lines = (
            ' '.join(word['text'] for word in sorted(line, key=lambda w: w['left']))
            for line in lines
        )

        return '\n'.join(reconstructed_lines)

    def group_into_lines(self, results):
        """Группирует слова в строки на основе Y-координат"""
        word_data = []

        for result in results:
            if len(result) >= 2:
                bbox, text = result[0], result[1]
                text = self._fast_replace_symbols(str(text).strip())

                # Получаем координаты bounding box
                points = np.array(bbox)

                left = points[0][0]
                top = points[0][1]
                right = points[2][0]
                bottom = points[2][1]

                word_data.append({
                    "text": text,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "avg_y": (top + bottom) / 2
                })

        sorted_words = sorted(word_data, key=lambda w: w["avg_y"])
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]["avg_y"]

        for word in sorted_words[1:]:
            # Если слово находится достаточно близко по Y, добавляем в текущую строку
            if abs(word["avg_y"] - current_y) <= self.y_tolerance:
                current_line.append(word)
            else:
                # Начинаем новую строку
                lines.append(current_line)
                current_line = [word]
                current_y = word["avg_y"]

        if current_line:
            lines.append(current_line)

        return lines

    @staticmethod
    def _fast_replace_symbols(text: str) -> str:
        """Быстрая замена символов Ng и N на №"""
        if not text:
            return text

        text = re.sub(r'[NJ№].?\s+', r'№ ', text)
        text = text.replace(' 0 ', 'о')
        return text

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


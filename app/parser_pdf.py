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
        """Инициализация PaddleOCR с GPU"""
        try:
            use_gpu = torch.cuda.is_available()
            self.ocr = PaddleOCR(
                lang='ru',
                use_angle_cls=True,  # определение ориентации текста
                # det=True,  # детекция текста (включено по умолчанию)
                # rec=True,  # распознавание текста (включено по умолчанию)
                # cls=True,  # классификация ориентации

                # Пути к моделям (опционально)
                # det_model_dir='path/to/det/model',
                # rec_model_dir='path/to/rec/model',
                # cls_model_dir='path/to/cls/model',

                # Производительность
                use_gpu=True,  # использовать GPU если доступно

                # Улучшенные параметры детекции
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,  # Увеличил для лучшего качества
                det_db_unclip_ratio=2.0,  # Увеличил для лучшего охвата текста

                # Улучшенные параметры распознавания
                rec_batch_num=8,  # Уменьшил для стабильности
                drop_score=0.7,  # Повысил порог уверенности

                # Дополнительные настройки
                det_limit_side_len=1920,  # Максимальный размер стороны для детекции
                det_limit_type='max',  # Ограничение по максимальной стороне
                rec_image_height=48,  # Высота изображения для распознавания

                # Улучшенные параметры для русского языка
                rec_char_dict_path=None,  # Использовать стандартный словарь
                cls_image_shape='3, 48, 192',  # Размер для классификации
                cls_batch_num=6,
                cls_thresh=0.9,
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
                dpi=400,
                fmt='JPEG',
                thread_count=2,
                grayscale=False,
                strict=False
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

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения распознавания"""
        # PaddleOCR лучше работает с RGB, а не grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        if width < 1200 or height < 1600:
            # Увеличиваем размер для мелкого текста
            new_width = max(width, 1200)
            new_height = max(height, 1600)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Увеличиваем контрастность
        image = self._enhance_image(image)

        return image

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Улучшение качества изображения для OCR"""
        # Конвертируем PIL в OpenCV
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Увеличиваем резкость
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)

        # Увеличиваем контрастность с CLAHE
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Дополнительное улучшение контраста
        img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=10)

        # Конвертируем обратно в PIL
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

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

            return self._parse_ocr_result(result, page_num)

        except Exception as e:
            config.logger.error(f"PaddleOCR ошибка на странице {page_num + 1}: {e}")
            return ""

    def _parse_ocr_result(self, result, page_num: int) -> str:
        """Парсинг и постобработка результатов OCR"""
        if not result or not result[0]:
            config.logger.info(f"Текст не найден на странице {page_num + 1}")
            return ""

        try:
            texts = []
            confidences = []

            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]

                    # Фильтруем по уверенности
                    if confidence >= 0.7:  # Используем тот же порог что в drop_score
                        texts.append(text)
                        confidences.append(confidence)

            if texts:
                # Объединяем текст с учетом структуры
                full_text = self._reconstruct_text(texts, result[0])
                avg_confidence = sum(confidences) / len(confidences)
                config.logger.info(
                    f"Страница {page_num + 1}: распознано {len(texts)} строк, средняя уверенность: {avg_confidence:.3f}")
                return full_text
            else:
                config.logger.info(f"На странице {page_num + 1} не найдено текста с достаточной уверенностью")
                return ""

        except Exception as e:
            config.logger.error(f"Ошибка парсинга результатов OCR: {e}")
            return ""

    def _reconstruct_text(self, texts: List[str], raw_result: List) -> str:
        """Восстановление структуры текста из результатов OCR"""
        try:
            # Сортируем строки по Y-координате (сверху вниз)
            lines_with_pos = []
            for line in raw_result:
                if line and len(line) >= 2:
                    points = line[0]
                    text = line[1][0]
                    # Берем среднюю Y-координату
                    y_coord = sum(point[1] for point in points) / len(points)
                    lines_with_pos.append((y_coord, text))

            # Сортируем по Y-координате
            lines_with_pos.sort(key=lambda x: x[0])

            # Объединяем текст
            reconstructed = []
            current_line = []
            last_y = None
            y_threshold = 25  # Порог для определения новой строки

            for y, text in lines_with_pos:
                if last_y is None or abs(y - last_y) > y_threshold:
                    if current_line:
                        reconstructed.append(' '.join(current_line))
                        current_line = []
                current_line.append(text)
                last_y = y

            if current_line:
                reconstructed.append(' '.join(current_line))

            return '\n'.join(reconstructed)

        except Exception as e:
            config.logger.error(f"Ошибка восстановления текста: {e}")
            return '\n'.join(texts)

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


# Внешние зависимости
import psutil
import os
import re
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
                det_db_box_thresh=0.6,  # Увеличил для лучшего качества
                det_db_unclip_ratio=1.8,  # Увеличил для лучшего охвата текста

                # Улучшенные параметры распознавания
                rec_batch_num=4,  # Уменьшил для стабильности
                drop_score=0.5,  # Повысил порог уверенности

                # Дополнительные настройки
                det_limit_side_len=2048,  # Максимальный размер стороны для детекции
                det_limit_type='max',  # Ограничение по максимальной стороне
                rec_image_height=64,  # Высота изображения для распознавания

                # Специфичные настройки для русского языка
                rec_char_type='ru',  # Явно указываем русский язык
                cls_batch_num=4,
                cls_thresh=0.6,

                # Дополнительные улучшения
                use_dilation=False,  # Расширение для детекции мелкого текста
                det_algorithm='DB',  # Явно указываем алгоритм детекции
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
                fmt='JPEG',
                thread_count=2,
                grayscale=True,
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
        if image.mode != 'L':
            image = image.convert('L')

        original_width, original_height = image.size

        target_min_size = 1200
        if min(original_width, original_height) < target_min_size:
            scale_factor = target_min_size / min(original_width, original_height)
            scale_factor = min(scale_factor, 2.5)  # Ограничиваем увеличение
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            config.logger.debug(f"Увеличен размер: {original_width}x{original_height} -> {new_width}x{new_height}")

        # ✅ Применяем улучшения для Ч/Б изображения
        return self._enhance_grayscale_image(image)

    def _enhance_grayscale_image(self, image: Image.Image) -> Image.Image:
        """Специализированное улучшение для Ч/Б изображений"""
        try:
            # Конвертируем PIL в numpy array
            img_np = np.array(image)

            # ✅ ПРОВЕРКА 1: Убедимся что изображение 2D (grayscale)
            if len(img_np.shape) > 2:
                # Если изображение цветное, конвертируем в grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                config.logger.warning("Изображение было цветным, сконвертировано в grayscale")

            # ✅ ПРОВЕРКА 2: Убедимся что изображение не пустое
            if img_np.size == 0:
                config.logger.warning("Пустое изображение, возвращаем оригинал")
                return image

            # ✅ ПРОВЕРКА 3: Проверим тип данных
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # 1. Убираем шум - используем НЕЧЕТНЫЙ размер ядра
            img_denoised = cv2.medianBlur(img_np, 3)  # 3 - нечетное число

            # 2. Адаптивная бинаризация с проверкой параметров
            block_size = 15
            # Блок должен быть нечетным и > 1
            if block_size % 2 == 0:
                block_size += 1

            img_binary = cv2.adaptiveThreshold(
                img_denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,  # Гарантированно нечетное
                5  # Смещение порога
            )

            # 3. Улучшаем резкость (только для бинарного изображения)
            kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
            img_sharpened = cv2.filter2D(img_binary, -1, kernel)

            # 4. Морфологические операции для очистки текста
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_cleaned = cv2.morphologyEx(img_sharpened, cv2.MORPH_CLOSE, kernel_morph)

            # 5. Убираем мелкие шумы - снова НЕЧЕТНЫЙ размер ядра
            img_final = cv2.medianBlur(img_cleaned, 3)  # Исправил на 3 (вместо 2)

            return Image.fromarray(img_final)

        except Exception as e:
            config.logger.error(f"Ошибка улучшения Ч/Б изображения: {e}")
            return image

    def _perform_ocr_paddle(self, image: Image.Image, page_num: int) -> str:
        """Улучшенное распознавание с постобработкой"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)

            if image_np.size == 0:
                config.logger.warning(f"Пустое изображение на странице {page_num + 1}")
                return ""

            # Распознавание с дополнительными параметрами
            result = self.ocr.ocr(image_np, cls=True)

            # Постобработка результатов
            return self._parse_and_correct_ocr_result(result, page_num)

        except Exception as e:
            config.logger.error(f"PaddleOCR ошибка на странице {page_num + 1}: {e}")
            return ""

    def _parse_and_correct_ocr_result(self, result, page_num: int) -> str:
        """Парсинг и коррекция результатов OCR для русского языка"""
        if not result or not result[0]:
            config.logger.info(f"Текст не найден на странице {page_num + 1}")
            return ""

        try:
            lines_with_data = []

            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]

                    # Применяем коррекцию для распространенных ошибок русского языка
                    corrected_text = self._correct_russian_ocr_errors(text)

                    # Более низкий порог уверенности с последующей коррекцией
                    if confidence >= 0.4:
                        lines_with_data.append({
                            'text': corrected_text,
                            'confidence': confidence,
                            'bbox': line[0]
                        })

            if lines_with_data:
                # Восстанавливаем структуру с учетом координат
                reconstructed_text = self._reconstruct_text_with_layout(lines_with_data)

                # Применяем дополнительную постобработку ко всему тексту
                final_text = self._postprocess_russian_text(reconstructed_text)

                avg_confidence = sum(item['confidence'] for item in lines_with_data) / len(lines_with_data)
                config.logger.info(
                    f"Страница {page_num + 1}: {len(lines_with_data)} строк, уверенность: {avg_confidence:.3f}")

                return final_text
            else:
                config.logger.info(f"На странице {page_num + 1} не найдено текста")
                return ""

        except Exception as e:
            config.logger.error(f"Ошибка парсинга результатов OCR: {e}")
            return ""

    def _correct_russian_ocr_errors(self, text: str) -> str:
        """Коррекция распространенных ошибок OCR для русского языка"""
        corrections = {
            # Частые символов
            '0': 'О', '1': 'І', '2': 'Z', '3': 'З',
            '4': 'Ч', '5': 'Б', '6': 'б', '7': 'Т',
            '8': 'В', '9': 'д',

            # Английские и русские буквы
            'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е',
            'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О',
            'P': 'Р', 'T': 'Т', 'X': 'Х', 'Y': 'У',
            'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о',
            'p': 'р', 'x': 'х', 'y': 'у',

            # Частые опечатки в словах
            'слъ': 'сле', 'тчк': 'тск', 'ьч': 'ьс'
        }

        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)

        return corrected_text

    def _postprocess_russian_text(self, text: str) -> str:
        """Постобработка всего текста для улучшения качества"""
        # Исправляем частые проблемы с пробелами
        text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Цифры разделенные пробелами
        text = re.sub(r'([а-яё])\s+([а-яё])', r'\1\2', text)  # Разорванные слова

        # Восстанавливаем заглавные буквы в начале предложений
        sentences = re.split(r'([.!?])\s+', text)
        processed_sentences = []

        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and len(sentence) > 1:
                    # Первую букву делаем заглавной
                    sentence = sentence[0].upper() + sentence[1:]
                processed_sentences.append(sentence)

                if i + 1 < len(sentences):
                    processed_sentences.append(sentences[i + 1])

        return ' '.join(processed_sentences)

    def _reconstruct_text_with_layout(self, lines_data: List[dict]) -> str:
        """Восстановление текста с учетом пространственного расположения"""
        try:
            # Группируем строки по Y-координатам
            lines_data.sort(key=lambda x: x['bbox'][0][1])  # Сортировка по Y

            reconstructed = []
            current_paragraph = []
            last_bottom = None

            for line in lines_data:
                bbox = line['bbox']
                top = bbox[0][1]

                if last_bottom is not None and top - last_bottom > 50:  # Новый параграф
                    if current_paragraph:
                        reconstructed.append(' '.join(current_paragraph))
                        current_paragraph = []

                current_paragraph.append(line['text'])
                last_bottom = bbox[2][1]  # Нижняя координата bbox

            if current_paragraph:
                reconstructed.append(' '.join(current_paragraph))

            return '\n'.join(reconstructed)

        except Exception as e:
            config.logger.error(f"Ошибка восстановления структуры: {e}")
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


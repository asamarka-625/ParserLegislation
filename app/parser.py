# Внешние зависимости
from typing import List
from datetime import datetime
import asyncio
from uuid import UUID
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import httpx
# Внутренние модули
from app.config import get_config
from app.scheme import DataSite


config = get_config()


class StopParser(Exception):
    def __init__(self, message):
        super().__init__(message)


class Parser:
    def __init__(self, uuid_authority: UUID):
        self.url_base = f"http://publication.pravo.gov.ru/Documents/search?pageSize=200&SignatoryAuthorityId={uuid_authority}&=&PublishDateSearchType=0&NumberSearchType=0&DocumentDateSearchType=0&JdRegSearchType=0&SortedBy=6&SortDestination=1"
        self.ua = UserAgent()
        self.headers = {
            "Accept": "text / html, application / xhtml + xml, application / xml;q = 0.9, * / *;q = 0.8",
            "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
            "Host": "publication.pravo.gov.ru",
            "Referer": "http://publication.pravo.gov.ru/documents/monthly",
            "User-Agent": self.ua.random
        }

    async def get_async_response(self, url: str, client: httpx.AsyncClient) -> bytes:
        config.logger.info(f"Делаем запрос на: {url}")
        """Используем переданный клиент для лучшей производительности"""
        try:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.content

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

    def get_response(self, url: str) -> bytes:
        config.logger.info(f"Делаем запрос на: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.content

        except requests.HTTPError as err:
            config.logger.error(f"HTTPError: {err}")

        except Exception as err:
            config.logger.error(f"Error: {err}")

    @staticmethod
    def get_data_from_html(content: bytes) -> List[DataSite]:
        config.logger.info("Парсим данные")
        soup = BeautifulSoup(content, 'html.parser')

        blocks = soup.find_all('div', class_='documents-table-row')
        if len(blocks) == 0:
            raise StopParser("Нет данных")

        results = []
        for block in blocks:
            try:
                name = block.find('a', class_='documents-item-name').text.strip()
                info_document = block.find('div', class_='infoindocumentlist').find_all('div')
                publication_number = int(info_document[0].find('span', class_='info-data').text.strip())
                publication_date = info_document[1].find('span', class_='info-data').text.strip()
                publication_date = datetime.strptime(publication_date, "%d.%m.%Y")
                link_pdf = info_document[2].find('a', class_='documents-item-file').get('href')

                results.append(DataSite(
                    name=name,
                    publication_number=publication_number,
                    publication_date=publication_date,
                    link_pdf=link_pdf
                ))

            except Exception as err:
                config.logger.error(f"Error: {err}")

        return results

    async def async_run(self) -> List[DataSite]:
        data = []
        batch_size = 10  # Обрабатываем по 5 страниц за раз
        batch_start = 1
        delay_between_batches = 1  # Задержка между пачками

        config.logger.info(f"Начинаем пакетный парсинг страниц")

        while True:
            try:
                batch_end = batch_start + batch_size - 1
                config.logger.info(f"Обрабатываем пачку страниц {batch_start}-{batch_end}")

                # Создаем клиент для каждой пачки
                timeout = httpx.Timeout(
                    connect=15.0,  # Таймаут на подключение
                    read=60.0,  # Таймаут на чтение
                    write=15.0,  # Таймаут на запрос
                    pool=100.0  # Таймаут на получение из пула
                )
                limits = httpx.Limits(max_connections=3, max_keepalive_connections=2)

                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    tasks = []
                    for page in range(batch_start, batch_end + 1):
                        url = f"{self.url_base}&index={page}"
                        task = self.get_async_response(url, client)
                        tasks.append(task)

                    batch_contents = await asyncio.gather(*tasks, return_exceptions=True)

                    # Обрабатываем результаты пачки
                    for i, content in enumerate(batch_contents):
                        page_num = batch_start + i
                        if isinstance(content, Exception):
                            config.logger.error(f"❌ Страница {page_num}: {type(content).__name__}")
                            continue

                        try:
                            results = self.get_data_from_html(content)
                            data.extend(results)
                            config.logger.info(f"✅ Страница {page_num}: {len(results)} записей")

                        except StopParser:
                            config.logger.info(f"Останавливаем парсинг страниц. На странице {page_num} пусто")
                            raise

                        except Exception as err:
                            config.logger.error(f"❌ Ошибка парсинга страницы {page_num}: {err}")

            except StopParser:
                config.logger.info(f"Выходим из цикла...")
                break

            else:
                batch_start += batch_size
                config.logger.info(f"Пауза {delay_between_batches} сек...")
                await asyncio.sleep(delay_between_batches)

        return data

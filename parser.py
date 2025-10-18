import requests
import json


class Parser:
    
    def __init__(
        self,
        file_cookies: str,
        file_data: str,
        file_ans: str
    ):
        self.file_cookies = file_cookies
        self.file_data = file_data
        self.file_ans = file_ans
        self.session = requests.Session()
        self.headers = {
            "Accept": "*/*, XMLHttpRequest/*",
            "Host": "demo.consultant.ru",
            "Referer": "https://demo.consultant.ru/cgi/online.cgi?req=query&cacheid=2F0224C944C329FEAD332958AA1E5921&NOQUERYLOG=1&ts=r62axzU8YTtiFqiL1&rnd=6f00eg",
            "Origin": "https://demo.consultant.ru",
            "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
        }
    
    @staticmethod
    def get_json(filename: str) -> dict:
        with open(f"{filename}.json", mode='r', encoding='utf-8') as file:
            return json.load(file)
    
    @staticmethod
    def write_json(filename: str, data: dict) -> None:
        with open(f"{filename}.json", mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            
    def init_session(self):
        cookies = self.get_json(self.file_cookies)
        self.session.cookies.update(cookies)
    
    def get_post_response(self, url: str) -> dict:
        try:
            data = self.get_json(self.file_data)
            response = self.session.post(url, data=data, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.HTTPError as err:
            print(f"HTTPError: {err}")
        
        except Exception as err:
            print(f"Error: {err}")
            
    def run(self, url: str) -> None:
        self.init_session()
        content = self.get_post_response(url)
        self.write_json(filename=self.file_ans, data=content)
        
if __name__ == "__main__":
    parser = Parser(
        file_cookies="cookies",
        file_data="data",
        file_ans="answer"
    )
    parser.run(url="https://demo.consultant.ru/cgi/online.cgi?rnd=6f00eg")
    
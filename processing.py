from app import worker_convert_binary_to_text_batch
import asyncio


if __name__ == "__main__":
    asyncio.run(worker_convert_binary_to_text_batch())
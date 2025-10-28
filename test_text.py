from app.crud import sql_get_text_by_id


if __name__ == "__main__":
    import asyncio
    asyncio.run(sql_get_text_by_id())
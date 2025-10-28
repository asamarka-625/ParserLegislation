from app.crud import sql_get_text_by_id


async def test_def():
    legislation_id = input("Введите id документа: ")
    text = await sql_get_text_by_id(legislation_id)
    print(text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_def())
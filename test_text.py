from app.crud import sql_get_text_by_id


if __name__ == "__main__":
    import asyncio

    leg_id = int(input("Напишите id: "))
    asyncio.run(sql_get_text_by_id(leg_id))
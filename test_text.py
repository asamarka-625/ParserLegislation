from app.crud import sql_get_text_by_id


def should_join_lines(line1, line2):
    """Определяет, должны ли две строки быть объединены"""
    if not line1 or not line2:
        return False

    # Если первая строка заканчивается знаком препинания - не объединяем
    if line1.strip() and line1[-1] in '.!?;:':
        return False

    # Если вторая строка начинается с заглавной буквы - вероятно новое предложение
    if line2.strip() and line2[0].isupper():
        return False

    # Если вторая строка короткая и может быть продолжением
    if len(line2.split()) <= 3:
        return True

    return True


def fix_text_from_bad_reconstruction(text: str) -> str:
    """Исправляет текст, восстановленный функцией _fast_reconstruct_structure"""
    lines = text.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        current_line = lines[i].strip()

        if not current_line:
            i += 1
            continue

        # Проверяем следующие строки на возможность объединения
        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip()

            if not next_line:
                break

            if should_join_lines(current_line, next_line):
                current_line += ' ' + next_line
                j += 1
            else:
                break

        fixed_lines.append(current_line)
        i = j

    # Восстанавливаем структуру абзацев
    result = []
    current_paragraph = []

    for line in fixed_lines:
        if line:
            current_paragraph.append(line)
        else:
            if current_paragraph:
                result.append(' '.join(current_paragraph))
                current_paragraph = []

    if current_paragraph:
        result.append(' '.join(current_paragraph))

    return '\n\n'.join(result)


def fix_line_word_order(line):
    """Исправляет порядок слов в одной строке"""
    words = line.split()
    if len(words) <= 3:
        return line

    # Эвристики для исправления порядка
    fixed_words = []
    i = 0

    while i < len(words):
        current_word = words[i]

        # Проверяем пары слов на возможную перестановку
        if i < len(words) - 1:
            next_word = words[i + 1]

            # Если текущее слово короткое, а следующее длинное - возможно переставлены
            if (len(current_word) <= 2 and len(next_word) > 3 and
                    not current_word[0].isupper() and not next_word[0].isupper()):
                fixed_words.append(next_word)
                fixed_words.append(current_word)
                i += 2
                continue

        fixed_words.append(current_word)
        i += 1

    return ' '.join(fixed_words)

# Дополнительная функция для исправления переставленных слов внутри строк
def fix_word_order_in_text(text: str) -> str:
    """Исправляет порядок слов внутри строк"""

    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        if line.strip():
            fixed_line = fix_line_word_order(line.strip())
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append('')

    return '\n'.join(fixed_lines)


# Основная функция для исправления всего текста
def completely_fix_ocr_text(text: str) -> str:
    """Полное исправление текста с проблемами восстановления структуры"""
    if not text:
        return text

    # Шаг 1: Исправляем структуру абзацев и объединяем разорванные строки
    text = fix_text_from_bad_reconstruction(text)

    # Шаг 2: Исправляем порядок слов внутри строк
    text = fix_word_order_in_text(text)

    return text


async def correction_text():
    leg_id = int(input("Напишите id: "))
    wrong_text = await sql_get_text_by_id(leg_id)

    correct_text = completely_fix_ocr_text(wrong_text)

    print(correct_text)


if __name__ == "__main__":
    import asyncio


    asyncio.run(correction_text())
import asyncio
import logging
import sys
import os
from os import getenv
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.rag_engine import query_rag

from dotenv import load_dotenv
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    sys.exit(1)

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    Обработчик команды /start
    """
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}! 👋\n\n"
        "Я — умный помощник по учебникам Зорича и Севастьянова.\n"
        "Задай мне вопрос по матанализу или теории вероятностей, и я найду ответ в книгах."
    )


@dp.message()
async def rag_handler(message: Message) -> None:
    """
    Обработчик любых текстовых сообщений
    """
    user_query = message.text

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        response_generator, sources = await asyncio.to_thread(query_rag, user_query)

        full_answer = ""
        for chunk in response_generator:
            full_answer += chunk

        final_text = full_answer

        if sources:
            source_list = "\n".join([f"• <i>{s}</i>" for s in sources])
            final_text += f"\n\n<b>Источники:</b>\n{source_list}"

        await message.answer(final_text, parse_mode=None)

    except Exception as e:
        await message.answer(f"Ошибка")


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    print("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен")
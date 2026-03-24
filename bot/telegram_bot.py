"""
Telegram бот для VLM моделей.

Отправьте боту изображение или текст — получите ответ от VLM.

Запуск:
    TELEGRAM_BOT_TOKEN=xxx python3 bot/telegram_bot.py

Или с кастомными настройками:
    TELEGRAM_BOT_TOKEN=xxx \
    VLLM_SERVER=http://localhost:8000/v1 \
    VLLM_MODEL=qwen3-vl-8b \
    python3 bot/telegram_bot.py

Зависимости:
    pip install python-telegram-bot openai
"""

import asyncio
import base64
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import AsyncVLMClient

# ============================================================
# Конфигурация
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
VLLM_SERVER = os.environ.get("VLLM_SERVER", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3-vl-8b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))

# Список разрешённых user_id (пустой = все разрешены)
ALLOWED_USERS = os.environ.get("ALLOWED_USERS", "")  # через запятую: "123,456"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_client() -> AsyncVLMClient:
    return AsyncVLMClient(
        base_url=VLLM_SERVER,
        model_name=VLLM_MODEL,
    )


def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS:
        return True
    allowed = [int(x.strip()) for x in ALLOWED_USERS.split(",") if x.strip()]
    return user_id in allowed


# ============================================================
# Обработчики команд
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "VLM Bot\n\n"
        "Отправьте мне:\n"
        "- Изображение (с подписью или без) — получите анализ\n"
        "- Текст — получите текстовый ответ\n\n"
        "Команды:\n"
        "/ocr — режим OCR (извлечение текста)\n"
        "/describe — режим описания\n"
        "/elements — список UI элементов\n"
        "/model — текущая модель\n"
        "/help — справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Как пользоваться:\n\n"
        "1. Отправьте фото — бот опишет что на нём\n"
        "2. Отправьте фото с подписью — бот ответит на ваш вопрос по фото\n"
        "3. Напишите /ocr и отправьте фото — бот извлечёт текст\n"
        "4. Отправьте текст без фото — бот ответит как обычный LLM\n\n"
        f"Текущая модель: {VLLM_MODEL}\n"
        f"Сервер: {VLLM_SERVER}"
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Model: {VLLM_MODEL}\n"
        f"Server: {VLLM_SERVER}\n"
        f"Max tokens: {MAX_TOKENS}\n"
        f"Temperature: {TEMPERATURE}"
    )


async def cmd_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "ocr"
    await update.message.reply_text(
        "OCR mode. Отправьте изображение — извлеку текст.\n"
        "Режим сбросится после обработки."
    )


async def cmd_describe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "describe"
    await update.message.reply_text(
        "Describe mode. Отправьте изображение — опишу подробно.\n"
        "Режим сбросится после обработки."
    )


async def cmd_elements(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "elements"
    await update.message.reply_text(
        "UI Elements mode. Отправьте скриншот — найду UI элементы.\n"
        "Режим сбросится после обработки."
    )


# ============================================================
# Обработка сообщений
# ============================================================

MODE_PROMPTS = {
    "ocr": "Extract ALL text from this image. Return only the text, preserving layout.",
    "describe": (
        "Describe this image in detail:\n"
        "1. What is shown?\n"
        "2. Key elements\n"
        "3. Text visible\n"
        "4. Notable details"
    ),
    "elements": (
        "List all interactive UI elements:\n"
        "- Buttons, fields, links, menus, icons\n"
        "Format as numbered list with locations."
    ),
}


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка фото."""
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    # Выбираем промпт
    mode = context.user_data.pop("mode", None)
    caption = update.message.caption or ""

    if mode and mode in MODE_PROMPTS:
        prompt = MODE_PROMPTS[mode]
    elif caption:
        prompt = caption
    else:
        prompt = "Describe this image. If there is text, also extract it."

    # Скачиваем фото (берём самое большое)
    photo = update.message.photo[-1]
    status_msg = await update.message.reply_text("Analyzing...")

    try:
        file = await context.bot.get_file(photo.file_id)
        buf = BytesIO()
        await file.download_to_memory(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = get_client()
        result = await client.chat(
            prompt=prompt,
            images=[b64],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        # Telegram лимит 4096 символов
        if len(result) > 4000:
            # Разбиваем на части
            parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
            await status_msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)
        else:
            await status_msg.edit_text(result)

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await status_msg.edit_text(f"Error: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка документов (картинки отправленные как файл)."""
    if not is_allowed(update.effective_user.id):
        return

    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Send an image file.")
        return

    mode = context.user_data.pop("mode", None)
    caption = update.message.caption or ""
    prompt = MODE_PROMPTS.get(mode, caption or "Describe this image.")

    status_msg = await update.message.reply_text("Analyzing...")

    try:
        file = await context.bot.get_file(doc.file_id)
        buf = BytesIO()
        await file.download_to_memory(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = get_client()
        result = await client.chat(
            prompt=prompt,
            images=[b64],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        if len(result) > 4000:
            parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
            await status_msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)
        else:
            await status_msg.edit_text(result)

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        await status_msg.edit_text(f"Error: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений (без изображений)."""
    if not is_allowed(update.effective_user.id):
        return

    prompt = update.message.text
    if not prompt:
        return

    status_msg = await update.message.reply_text("Thinking...")

    try:
        client = get_client()
        result = await client.chat(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        if len(result) > 4000:
            parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
            await status_msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)
        else:
            await status_msg.edit_text(result)

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        await status_msg.edit_text(f"Error: {e}")


# ============================================================
# Main
# ============================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("Usage: TELEGRAM_BOT_TOKEN=xxx python3 bot/telegram_bot.py")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("ocr", cmd_ocr))
    app.add_handler(CommandHandler("describe", cmd_describe))
    app.add_handler(CommandHandler("elements", cmd_elements))

    # Сообщения
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info(f"Starting bot with model {VLLM_MODEL} on {VLLM_SERVER}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

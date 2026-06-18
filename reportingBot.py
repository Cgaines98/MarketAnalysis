import logging
import os

import discord
from dotenv import load_dotenv

import stockDataCollector

load_dotenv()
logger = logging.getLogger(__name__)

TOKEN: str = os.getenv("BOT_TOKEN", "")
GUILD: str = os.getenv("DISCORD_GUILD", "")

client = discord.Client(intents=discord.Intents.all())
toolkit: dict[str, callable] = {"hist": stockDataCollector.get_stock_history}


@client.event
async def on_ready() -> None:
    logger.info("Bot connected")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user:
        return

    content: str = message.content
    logger.debug("Received message: %s", content)

    if content.startswith("c^ "):
        command = content[3:]
        handler = toolkit.get(command)
        if handler is None:
            await message.channel.send(f"Unknown command: `{command}`")
            return
        result = handler("TSLA", "1d", "1h")
        await message.channel.send(str(result.head()))


client.run(TOKEN)

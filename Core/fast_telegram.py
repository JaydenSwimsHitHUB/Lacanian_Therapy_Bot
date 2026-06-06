import asyncio
import time
import logging
from rasa.core.channels.telegram import TelegramInput

# Establish a direct, unbuffered line to the screen
logger = logging.getLogger(__name__)

class FastTelegramInput(TelegramInput):
    @classmethod
    def name(cls) -> str:
        return "fast_telegram"

    def blueprint(self, on_new_message):
        
        async def fast_on_new_message(user_message):
            trigger_time = time.time()
            message_id = id(user_message)
            logger.warning(f"[TRACKER-{message_id}] 1. Webhook triggered. Starting task.")
            
            async def safe_process():
                try:
                    logger.warning(f"[TRACKER-{message_id}] 2. Entering Rasa pipeline.")
                    start = time.time()
                    await on_new_message(user_message)
                    end = time.time()
                    logger.warning(f"[TRACKER-{message_id}] 3. Pipeline complete in {end - start:.2f}s.")
                except Exception as e:
                    logger.error(f"[TRACKER-{message_id}] ERROR: Pipeline crashed: {e}")

            # Fire the background task
            asyncio.create_task(safe_process())
            await asyncio.sleep(0) 
            
            finish_time = time.time()
            logger.warning(f"[TRACKER-{message_id}] 4. Handed back to server in {finish_time - trigger_time:.4f}s.")

        return super().blueprint(fast_on_new_message)
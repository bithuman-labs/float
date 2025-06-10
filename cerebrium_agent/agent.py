import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.plugins import bithuman, openai
from openai.types.beta.realtime.session import InputAudioTranscription

logger = logging.getLogger("bithuman-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

DEFAULT_AVATAR_IMAGE = os.path.join(os.path.dirname(__file__), "avatar.jpg")


lk_ws_url = os.getenv("LIVEKIT_URL_FLOAT_AGENT")
lk_api_key = os.getenv("LIVEKIT_API_KEY_FLOAT_AGENT")
lk_api_secret = os.getenv("LIVEKIT_API_SECRET_FLOAT_AGENT")
bithuman_api_url = os.getenv(
    "BITHUMAN_API_URL_FLOAT_AGENT",
    "https://api.cortex.cerebrium.ai/v4/p-5398b08f/float-agents-generator/launch?async=true",
)
bithuman_api_secret = os.getenv("BITHUMAN_API_SECRET_FLOAT_AGENT")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
        ),
    )

    avatar_image = ctx.room.local_participant.attributes.get(
        "avatar_image", DEFAULT_AVATAR_IMAGE
    )
    bithuman_avatar = bithuman.AvatarSession(
        mode="cloud",
        avatar_image=avatar_image,
        api_url=bithuman_api_url,
        api_secret=bithuman_api_secret,
    )
    await bithuman_avatar.start(
        session,
        room=ctx.room,
        livekit_url=lk_ws_url,
        livekit_api_key=lk_api_key,
        livekit_api_secret=lk_api_secret,
    )

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
            ws_url=lk_ws_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
        )
    )

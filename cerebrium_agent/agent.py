import logging
import os

os.environ["HF_HOME"] = "~/.cache/huggingface"

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
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai.types.beta.realtime.session import InputAudioTranscription

logger = logging.getLogger("bithuman-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

base_dir = os.getenv("BASE_DIR", "/persistent-storage/images")
DEFAULT_AVATAR_IMAGE = os.path.join(base_dir, "avatar.jpg")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
        ),
        turn_detection=MultilingualModel(),
    )

    avatar_image = ctx.room.local_participant.attributes.get(
        "avatar_image", DEFAULT_AVATAR_IMAGE
    )
    bithuman_avatar = bithuman.AvatarSession(
        mode="cloud",
        # avatar_image=avatar_image,
        avatar_id="A42NHA2628",
        api_url="https://api.cortex.cerebrium.ai/v4/p-5398b08f/cpu-avatar-service/launch?async=true",
        api_secret="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTUzOThiMDhmIiwiaWF0IjoxNzQ2MTIxMjIwLCJleHAiOjIwNjE2OTcyMjB9.CWNzYxEYU6_UOIXc55rgIwrZyliytKbWZH3A8Jxl-D1poEPBTJXEswr070h-J3kJafYlMgobK3ovvgsvKbKAXy20mTtobqL1u9HhUJXVgcDoE7IzJzaJcAlooShaC3BPbJkhh47bfxgJMq_bBkRzzoc52kZwtvnQVsqlIHbTyL5xXoqx_sIWW5KXR0B8B6w36xGVHrJLPE9qpYGqeTRnd4In3jn68ElCNTGhl2Wz_IoGHwXwlu81huCX_kWRL-crzU9ZBsxk7vqTQna8QFFVMjyG8NDDKx6fL8fFJG383JznmyAQGNFWObgsdvAa0ACEMbUqZHtDooH92OFnkB2ZNg",
    )
    await bithuman_avatar.start(
        session,
        room=ctx.room,
    )

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))

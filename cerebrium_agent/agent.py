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

base_dir = os.getenv("BASE_DIR", "/persistent-storage/images")
DEFAULT_AVATAR_IMAGE = os.path.join(base_dir, "avatar.jpg")

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

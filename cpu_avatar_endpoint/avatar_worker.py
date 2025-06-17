import asyncio
import logging
import os

from bithuman import AsyncBithuman
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import utils
from livekit.agents.cli.log import setup_logging
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)
from livekit.plugins.bithuman.avatar import BithumanGenerator

setup_logging("INFO", devmode=True, console=False)

load_dotenv()


ROOM = os.getenv("LIVEKIT_ROOM")
WS_URL = os.getenv("LIVEKIT_WS_URL")
TOKEN = os.getenv("LIVEKIT_TOKEN")
BITHUMAN_SECRET = os.getenv("BITHUMAN_API_SECRET")  # TODO: use user's secret?
BITHUMAN_IMX_PATH = os.getenv("BITHUMAN_IMX_PATH")

logger = logging.getLogger(f"avatar-{ROOM}")


def text_stream_handler(reader: rtc.TextStreamReader, participant: str):
    pass


async def start_avatar(room: rtc.Room, video_gen: BithumanGenerator) -> AvatarRunner:
    """Main application logic for the avatar worker"""
    on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
    if not on_behalf_of:
        raise ValueError("on_behalf_of is not set")

    room.register_text_stream_handler("lk.transcription", text_stream_handler)

    # Initialize and start worker
    output_width, output_height = video_gen.video_resolution
    avatar_options = AvatarOptions(
        video_width=output_width,
        video_height=output_height,
        video_fps=video_gen.video_fps,
        audio_sample_rate=video_gen.audio_sample_rate,
        audio_channels=1,
    )

    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room, frame_size_ms=10),
        video_gen=video_gen,
        options=avatar_options,
    )
    await asyncio.wait_for(runner.start(), timeout=120)
    return runner


async def main():
    logger.info(f"loading video gen model from {BITHUMAN_IMX_PATH}")

    runtime = await AsyncBithuman.create(
        api_secret=BITHUMAN_SECRET,
        model_path=BITHUMAN_IMX_PATH,
    )

    try:
        await runtime.start()

        video_gen = BithumanGenerator(runtime)

        stop_event = asyncio.Event()

        room = rtc.Room()
        await room.connect(WS_URL, TOKEN)
        on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)

        runner = await start_avatar(room, video_gen)
        close_runner_task: asyncio.Task[None] | None = None

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            nonlocal close_runner_task

            logger.info(f"Participant {participant.identity} disconnected")
            if participant.identity == on_behalf_of:
                logger.info("Agent disconnected, shutting down avatar worker...")
                if not close_runner_task:
                    close_runner_task = asyncio.create_task(runner.aclose())
                stop_event.set()

        @room.on("disconnected")
        def on_disconnected(reason: rtc.DisconnectReason):
            nonlocal close_runner_task

            logger.info(f"Room disconnected: {reason}")
            if not close_runner_task:
                close_runner_task = asyncio.create_task(runner.aclose())
            stop_event.set()

        # make sure the agent is connected, otherwise stop the worker
        try:
            await asyncio.wait_for(
                utils.wait_for_participant(room, identity=on_behalf_of), timeout=60
            )
        except asyncio.TimeoutError:
            logger.error("Agent not connected, shutting down avatar worker...")
            if not close_runner_task:
                close_runner_task = asyncio.create_task(runner.aclose())
            stop_event.set()
            return 1

        await stop_event.wait()
        return 0
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    return_code = asyncio.run(main())
    os._exit(return_code)

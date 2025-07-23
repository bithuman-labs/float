import argparse
import asyncio
import logging
import os
import sys

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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from generate import BaseOptions, FloatVideoGen  # noqa: E402

setup_logging("INFO", devmode=True, console=False)

load_dotenv()


ROOM = os.getenv("LIVEKIT_ROOM")
WS_URL = os.getenv("LIVEKIT_WS_URL")
TOKEN = os.getenv("LIVEKIT_TOKEN")
REF_IMAGE_PATH = os.getenv(
    "REF_IMAGE_PATH", os.path.join(THIS_DIR, "../assets/avatar-example.jpg")
)

logger = logging.getLogger(f"avatar-{ROOM}")


def text_stream_handler(reader: rtc.TextStreamReader, participant: str):
    pass


async def start_avatar(room: rtc.Room, video_gen: FloatVideoGen) -> AvatarRunner:
    """Main application logic for the avatar worker"""
    on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
    if not on_behalf_of:
        raise ValueError("on_behalf_of is not set")

    room.register_text_stream_handler("lk.transcription", text_stream_handler)

    # Initialize and start worker
    avatar_options = AvatarOptions(
        video_width=512,
        video_height=512,
        video_fps=25,
        audio_sample_rate=16000,
        audio_channels=1,
    )
    video_gen.start(ref_image=REF_IMAGE_PATH)
    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room, frame_size_ms=2000),
        video_gen=video_gen,
        options=avatar_options,
    )
    await asyncio.wait_for(runner.start(), timeout=120)
    return runner


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser: argparse.ArgumentParser):
        super().initialize(parser)
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="not using crop")

        return parser


async def main():
    logger.info("loading video gen model")
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    opt.seed = 15
    opt.a_cfg_scale = 2.0
    opt.e_cfg_scale = 1.0

    video_gen = FloatVideoGen(opt)
    logger.info("video gen model loaded")

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


if __name__ == "__main__":
    try:
        return_code = asyncio.run(main())
    except Exception:
        return_code = -1

    os._exit(return_code)

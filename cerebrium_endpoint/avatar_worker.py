import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from livekit import rtc
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

setup_logging("INFO", devmode=True, console=True)

load_dotenv()


ROOM = os.getenv("LIVEKIT_ROOM")
WS_URL = os.getenv("LIVEKIT_WS_URL")
TOKEN = os.getenv("LIVEKIT_TOKEN")
REF_IMAGE_PATH = os.getenv("REF_IMAGE_PATH", os.path.join(THIS_DIR, "../assets/avatar-example.png"))

logger = logging.getLogger(f"avatar-{ROOM}")


def text_stream_handler(reader: rtc.TextStreamReader, participant: str):
    pass


def shutdown(runner: AvatarRunner, stop: asyncio.Event):
    logger.info("Shutting down avatar worker...")
    asyncio.create_task(runner.aclose())
    stop.set()
    logger.info("Avatar worker shut down")


async def entrypoint(
    ws_url: str, token: str, video_gen: FloatVideoGen, stop_event: asyncio.Event
):
    """Main application logic for the avatar worker"""

    room = rtc.Room()
    await room.connect(ws_url, token)

    on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
    if not on_behalf_of:
        raise ValueError("on_behalf_of is not set")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant {participant.identity} disconnected")
        if participant.identity == on_behalf_of:
            shutdown(runner, stop_event)

    @room.on("disconnected")
    def on_disconnected(reason: rtc.DisconnectReason):
        logger.info(f"Room disconnected: {reason}")
        shutdown(runner, stop_event)

    room.register_text_stream_handler("lk.transcription", text_stream_handler)

    # Initialize and start worker
    avatar_options = AvatarOptions(
        video_width=512,
        video_height=512,
        video_fps=25,
        audio_sample_rate=16000,
        audio_channels=1,
    )
    video_gen.start(ref_image=REF_IMAGE_PATH)  # TODO: upload image
    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room, frame_size_ms=2000),
        video_gen=video_gen,
        options=avatar_options,
    )
    await asyncio.wait_for(runner.start(), timeout=120)


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
    await entrypoint(WS_URL, TOKEN, video_gen, stop_event)

    await stop_event.wait()


if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)

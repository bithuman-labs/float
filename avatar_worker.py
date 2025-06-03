import argparse
import json
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)

from generate import BaseOptions, FloatVideoGen

load_dotenv()

logger = logging.getLogger("avatar-example")

REF_IMAGE_PATH = "assets/Snipaste_2025-03-24_17-16-02.png"


def text_stream_handler(reader: rtc.TextStreamReader, participant: str):
    pass


async def entrypoint(ctx: JobContext):
    """Main application logic for the avatar worker"""

    metadata: dict = json.loads(ctx.job.metadata)
    # overwrite the url and token
    ctx._info.url = metadata.get("url")
    ctx._info.token = metadata.get("token")
    if not ctx._info.url or not ctx._info.token:
        raise ValueError("url or token is not set in metadata")
    await ctx.connect()

    ctx.room.register_text_stream_handler("lk.transcription", text_stream_handler)

    # Initialize and start worker
    avatar_options = AvatarOptions(
        video_width=512,
        video_height=512,
        video_fps=25,
        audio_sample_rate=16000,
        audio_channels=1,
    )
    video_gen: FloatVideoGen = ctx.proc.userdata["video_gen"]
    video_gen.start(ref_image=REF_IMAGE_PATH)  # TODO: upload image
    runner = AvatarRunner(
        ctx.room,
        audio_recv=DataStreamAudioReceiver(ctx.room, frame_size_ms=2000),
        video_gen=video_gen,
        options=avatar_options,
    )
    await runner.start()

    async def on_shutdown():
        logging.info("Shutting down avatar worker...")
        await runner.aclose()
        video_gen.close()
        logging.info("Avatar worker shut down")

    ctx.add_shutdown_callback(on_shutdown)


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser: argparse.ArgumentParser):
        super().initialize(parser)
        parser.add_argument(
            "--ref_path",
            default="assets/Snipaste_2025-03-24_17-16-02.png",
            type=str,
            help="ref",
        )
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="not using crop")
        parser.add_argument(
            "--ckpt_path",
            default="checkpoints/float.pth",
            type=str,
            help="checkpoint path",
        )
        return parser


def prewarm(process: JobProcess):
    logger.info("loading video gen model")
    opt = InferenceOptions().parse([])
    opt.rank, opt.ngpus = 0, 1
    opt.seed = 15
    opt.a_cfg_scale = 2.0
    opt.e_cfg_scale = 1.0

    video_gen = FloatVideoGen(opt)
    process.userdata["video_gen"] = video_gen
    logger.info("video gen model loaded")


# TODO: load based on GPU usage

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="avatar_worker",
            num_idle_processes=1,
            job_memory_warn_mb=4000,
            initialize_process_timeout=60,
        )
    )

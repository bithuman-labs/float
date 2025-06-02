"""
Inference Stage 2
"""

import os
import torch
import cv2
import torchvision
import subprocess
import librosa
import datetime
import tempfile
import face_alignment
import time
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import deque
from tqdm import tqdm
from typing import Literal, AsyncIterator
from queue import Queue, Empty
from dataclasses import dataclass, field
from transformers import Wav2Vec2FeatureExtractor
from livekit.agents.voice.avatar import VideoGenerator, AudioSegmentEnd
from collections.abc import AsyncGenerator

from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions
from moviepy import ImageSequenceClip, AudioArrayClip
from livekit import rtc


# class ClearBufferSentinel:
#     pass


# class AudioFlushSentinel:
#     pass


Emotion = Literal["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


@dataclass
class AudioAndControl:
    audio: np.ndarray | None = field(default=None, repr=False)  # 16kHz audio fp32
    flush: bool = False
    interrupt: bool = False

    emotion: Emotion | None = None


@dataclass
class GeneratedFrame:
    index: int  # frame index
    img: np.ndarray  # uint8 [H, W, 3]
    audio: np.ndarray  # 16kHz audio float32
    idle: bool  # whether the frame is idle
    audio_segment_ended: bool  # whether the audio segment is ended

    _sample: torch.Tensor
    _wa: torch.Tensor


class ListBuffer:
    def __init__(self, num_prev_frames: int = 10):
        self.buffer: list[GeneratedFrame] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()

        self._last_fetched = deque(maxlen=num_prev_frames)
        self._last_pushed = deque(maxlen=num_prev_frames)

    def push(self, item: GeneratedFrame) -> None:
        self.buffer.append(item)
        self._last_pushed.append(item)
        self._ready.set()

    def get(self, *, timeout: float | None = None) -> GeneratedFrame | None:
        start = time.time()
        while len(self.buffer) == 0:
            if timeout is not None:
                timeout -= time.time() - start
                if timeout <= 0:
                    return None
            self._ready.wait(timeout)
            self._ready.clear()

        with self._lock:
            item = self.buffer.pop(0)
            self._last_fetched.append(item)
            return item

    def truncate(self, only_idle_frames: bool = True, keep_left: int = 10) -> None:
        with self._lock:
            if only_idle_frames:
                last_non_idle: int | None = None
                for i in reversed(range(len(self.buffer))):
                    if not self.buffer[i].idle:
                        last_non_idle = i
                        break
                if last_non_idle is not None:
                    keep_left = max(keep_left, last_non_idle + 1)

            if len(self.buffer) > keep_left:
                self.buffer = self.buffer[:keep_left]
                self._last_pushed = self._last_fetched.copy()
                self._last_pushed.extend(self.buffer)

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def last_n_data(self) -> list[GeneratedFrame]:
        return list(self._last_pushed)


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False
        )

        # wav2vec2 audio preprocessor
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, local_files_only=True
        )

        # image transform
        self.transform = A.Compose(
            [
                A.Resize(
                    height=opt.input_size,
                    width=opt.input_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A_pytorch.ToTensorV2(),
            ]
        )

    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        mult = 360.0 / img.shape[0]

        resized_img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=mult,
            fy=mult,
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        bboxes = bboxes[0]  # Just use first bbox

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * 1.6)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs  # BBox center y, bbox center x

        crop_img = img[my - bs : my + bs, mx - bs : mx + bs]
        crop_img = cv2.resize(
            crop_img,
            dsize=(self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        return crop_img

    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.process_audio(speech_array, sampling_rate)

    def process_audio(
        self, speech_array: np.ndarray, sampling_rate: int
    ) -> torch.Tensor:
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values[0]

    def preprocess(self, ref_path: str, audio_path: str | None, no_crop: bool) -> dict:
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(image=s)["image"].unsqueeze(0)

        a = None
        if audio_path is not None:
            a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {"s": s, "a": a, "p": None, "e": None}


class InferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name:
                    pass
                else:
                    print(f"! Warning; {model_name} not found in state_dict.")

        del state_dict

    def save_video(
        self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str
    ) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_filename = temp_video.name
            vid = vid_target_recon.permute(0, 2, 3, 1)
            vid = vid.detach().clamp(-1, 1).cpu()
            vid = ((vid + 1) / 2 * 255).type("torch.ByteTensor")
            torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)
            if audio_path is not None:
                with open(os.devnull, "wb") as f:
                    command = "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(
                        temp_filename, audio_path, video_path
                    )
                    subprocess.call(command, shell=True, stdout=f, stderr=f)
                if os.path.exists(video_path):
                    os.remove(temp_filename)
            else:
                os.rename(temp_filename, video_path)
            return video_path

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = "S2E",
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = False,
    ) -> str:
        tic = time.time()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        toc = time.time()
        if verbose:
            print(f"> [Done] Preprocess. {toc - tic:.2f}s")

        # inference
        tic = time.time()
        images = []
        for img, sample, wa in self.G.inference(
            data=data,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed,
        ):
            images.append(img)
        images = torch.cat(images, dim=0)
        toc = time.time()

        fps = len(images) / (toc - tic)
        if verbose:
            print(
                f"> [Done] Inference. {toc - tic:.2f}s, output shape: {images.shape}, fps: {fps:.2f}"
            )

        res_video_path = self.save_video(images, res_video_path, audio_path)
        if verbose:
            print(f"> [Done] result saved at {res_video_path}")
        return res_video_path

    @torch.no_grad()
    def run_inference_stream(
        self,
        ref_image_path: str,
        audio_queue: Queue[AudioAndControl],
        output_buffer: ListBuffer,
        stop_event: threading.Event,
        *,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        talking_emotion: Emotion | None = None,
        idle_emotion: Emotion | None = None,
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
    ) -> None:
        # prepare image features
        data = self.data_processor.preprocess(ref_image_path, None, no_crop=no_crop)
        s = data["s"]
        s_r, r_s_lambda, s_r_feats = self.G.encode_image_into_latent(
            s.to(self.opt.rank)
        )
        if "s_r" in data:
            r_s = self.G.encode_identity_into_motion(s_r)
        else:
            r_s = self.G.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        # audio buffer
        audio_buffer = np.zeros(0, dtype=np.float32)
        sample_per_clip = int(np.ceil(self.opt.sampling_rate * self.opt.wav2vec_sec))
        sample_per_frame = int(np.ceil(self.opt.sampling_rate / self.opt.fps))

        def _get_prev_data(
            frames: list[GeneratedFrame],
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if len(frames) == 0:
                return None, None

            prev_sample = torch.stack([frame._sample for frame in frames], dim=1)
            prev_wa = torch.stack([frame._wa for frame in frames], dim=1)
            return prev_sample, prev_wa

        segment_ended = True
        global_index = 0
        first_frame_delay = 0.2  # 200ms
        delay_momentum = 0.5
        while not stop_event.is_set():
            is_last_segment = False
            try:
                chunk = audio_queue.get(timeout=0.01)

                keep_left = max(
                    2, np.ceil(first_frame_delay * self.opt.fps).astype(int)
                )
                if chunk.interrupt:
                    # TODO: dynamic truncate keep_left
                    output_buffer.truncate(only_idle_frames=False, keep_left=keep_left)
                    print(f"truncate and keep {keep_left} frames")

                if chunk.flush:
                    is_last_segment = True

                if chunk.audio is not None:
                    if segment_ended:
                        # clear idle frames and generate a new segment
                        output_buffer.truncate(
                            only_idle_frames=True, keep_left=keep_left
                        )
                        print(f"truncate and keep {keep_left} frames")

                    audio_buffer = np.concatenate([audio_buffer, chunk.audio])
                    segment_ended = False
                    is_idle = False

            except Empty:
                if output_buffer.size > 20:
                    continue
                if not segment_ended or len(audio_buffer) > 0:
                    chunk = AudioAndControl(flush=True)  # raise a warning
                else:
                    chunk = AudioAndControl(
                        audio=np.zeros(sample_per_clip, dtype=np.float32),
                        emotion=idle_emotion,
                    )
                    audio_buffer = chunk.audio  # fill the buffer with zeros
                    is_idle = True

            # get the audio chunk for inference
            audio_inference: np.ndarray | None = None
            if chunk.flush and len(audio_buffer) > 0:
                segment_ended = True
                audio_inference = audio_buffer
                audio_buffer = np.zeros(0, dtype=np.float32)
            elif len(audio_buffer) >= sample_per_clip:
                audio_inference = audio_buffer[:sample_per_clip]
                audio_buffer = audio_buffer[sample_per_clip:]

            if audio_inference is None:
                continue

            # previous frames
            prev_sample, prev_wa = _get_prev_data(output_buffer.last_n_data)

            # inference
            data_inference = data.copy()
            data_inference["a"] = self.data_processor.process_audio(
                audio_inference, self.opt.sampling_rate
            ).unsqueeze(0)

            num_frames = int(len(audio_inference) / sample_per_frame)
            if num_frames <= 0:
                continue

            tic = time.perf_counter()
            for i, (sample, wa) in enumerate(
                self.G.sample(
                    data=data_inference,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=chunk.emotion or talking_emotion,
                    nfe=nfe,
                    seed=seed,
                    prev_x=prev_sample,
                    prev_wa=prev_wa,
                )
            ):
                img = self.G.decode_latent_single_image(
                    s_r=s_r, s_r_feats=s_r_feats, r_d_t=sample
                )[0]
                img = img.permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                img = ((img + 1) / 2 * 255).astype(np.uint8)

                output_buffer.push(
                    GeneratedFrame(
                        index=global_index,
                        img=img,
                        audio=audio_inference[
                            i * sample_per_frame : (i + 1) * sample_per_frame
                        ],
                        idle=is_idle,
                        audio_segment_ended=is_last_segment
                        if i == num_frames - 1
                        else False,
                        _sample=sample,
                        _wa=wa,
                    )
                )
                if i == 0:
                    delay = time.perf_counter() - tic
                    first_frame_delay = (
                        first_frame_delay * (1 - delay_momentum)
                        + delay * delay_momentum
                    )
                global_index += 1
                if i >= num_frames - 1:
                    break


class FloatVideoGen(VideoGenerator):
    def __init__(self, opt, *, loop: asyncio.AbstractEventLoop | None = None):
        super().__init__()
        self.agent = InferenceAgent(opt)
        self.opt = opt

        self._loop = loop
        self.input_queue = Queue[AudioAndControl]()
        self.output_buffer = ListBuffer(num_prev_frames=opt.num_prev_frames)
        self.output_queue = asyncio.Queue[GeneratedFrame](maxsize=2)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.inference_thread: threading.Thread | None = None
        self.reader_thread: threading.Thread | None = None
        self.resampler: rtc.AudioResampler | None = None

    def start(self, *, ref_image: str):
        # Start inference thread
        self._loop = self._loop or asyncio.get_event_loop()

        self.inference_thread = threading.Thread(
            target=self.agent.run_inference_stream,
            args=(
                ref_image,
                self.input_queue,
                self.output_buffer,
                self.stop_event,
            ),
            kwargs=dict(
                a_cfg_scale=self.opt.a_cfg_scale,
                r_cfg_scale=self.opt.r_cfg_scale,
                e_cfg_scale=self.opt.e_cfg_scale,
                talking_emotion=None,
                idle_emotion=self.opt.emo,
                nfe=self.opt.nfe,
                no_crop=self.opt.no_crop,
                seed=self.opt.seed,
            ),
        )
        self.inference_thread.start()

        # Start reader thread
        self.reader_thread = threading.Thread(target=self._reader_loop)
        self.reader_thread.start()

    def _reader_loop(self):
        """Thread that reads from output_buffer and puts into async_queue"""
        while not self.stop_event.is_set():
            frame = self.output_buffer.get(timeout=0.1)
            if frame is None:
                continue
            fut = asyncio.run_coroutine_threadsafe(
                self.output_queue.put(frame), self._loop
            )
            while not self.stop_event.is_set():
                try:
                    fut.result(timeout=0.5)
                    break
                except TimeoutError:
                    continue

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        async def _push_impl(frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
            if isinstance(frame, AudioSegmentEnd):
                control = AudioAndControl(flush=True)
            else:
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                audio_data = np.clip(
                    audio_data.astype(np.float32) / np.iinfo(np.int16).max, -1, 1
                )
                control = AudioAndControl(audio=audio_data)
            await self._loop.run_in_executor(
                self.executor, self.input_queue.put, control
            )

        async for f in self._resample_audio(frame):
            await _push_impl(f)

    async def clear_buffer(self) -> None:
        def _clear_buffer():
            while not self.input_queue.empty():
                self.input_queue.get()
            self.input_queue.put(AudioAndControl(interrupt=True))

        await self._loop.run_in_executor(self.executor, _clear_buffer)

    async def _resample_audio(
        self, frame: rtc.AudioFrame | AudioSegmentEnd
    ) -> AsyncGenerator[rtc.AudioFrame | AudioSegmentEnd, None]:
        if isinstance(frame, AudioSegmentEnd):
            if self.resampler:
                for f in self.resampler.flush():
                    yield f
            yield frame
            self.resampler = None
            return

        if self.resampler is None and frame.sample_rate != self.opt.sampling_rate:
            self.resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate, output_rate=self.opt.sampling_rate
            )
        if self.resampler:
            for f in self.resampler.push(frame):
                yield f
        else:
            yield frame

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._iter_impl()

    async def _iter_impl(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        while not self.stop_event.is_set():
            try:
                frame: GeneratedFrame = await self.output_queue.get()
                # Convert frame to video frame
                h, w = frame.img.shape[:2]
                rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_img[:, :, :3] = frame.img
                rgba_img[:, :, 3] = 255
                video_frame = rtc.VideoFrame(
                    data=rgba_img.tobytes(),
                    type=rtc.VideoBufferType.RGBA,
                    width=w,
                    height=h,
                )
                yield video_frame

                # Convert audio to audio frame
                if frame.audio is not None and not frame.idle:
                    audio_data = (
                        np.clip(frame.audio, -1, 1) * np.iinfo(np.int16).max
                    ).astype(np.int16)
                    audio_frame = rtc.AudioFrame(
                        data=audio_data.tobytes(),
                        sample_rate=self.opt.sampling_rate,
                        num_channels=1,
                        samples_per_channel=len(audio_data),
                    )
                    yield audio_frame

                if frame.audio_segment_ended:
                    yield AudioSegmentEnd()

            except asyncio.QueueEmpty:
                continue

    def close(self):
        print("closing video_gen")
        self.stop_event.set()
        if self.inference_thread:
            self.inference_thread.join()
        if self.reader_thread:
            self.reader_thread.join()
        self.executor.shutdown(wait=True)
        print("video_gen closed")


def main(agent: InferenceAgent, ref_image_path: str, audio_path: str, opt):
    audio_queue = Queue()
    output_buffer = ListBuffer(num_prev_frames=opt.num_prev_frames)
    stop_event = threading.Event()
    frames: list[GeneratedFrame] = []

    def _read_frames(output_buffer: ListBuffer):
        pbar = tqdm(total=len(frames), desc="Reading frames", unit="frame")
        while not stop_event.is_set():
            frame = output_buffer.get(timeout=0.2)
            if frame is None:
                continue
            frames.append(frame)
            pbar.update(1)
        pbar.close()

    def _send_audio(
        audio: np.ndarray | str,
        chunk_size: int = 160,
        interrupt: bool = False,
        emotion: Emotion | None = None,
    ):
        if isinstance(audio, str):
            audio_np, _ = librosa.load(audio, sr=agent.opt.sampling_rate)
        else:
            audio_np = audio

        if interrupt:
            while not audio_queue.empty():
                audio_queue.get()
            audio_queue.put(AudioAndControl(interrupt=True))

        for i in range(0, len(audio_np), chunk_size):
            audio_queue.put(
                AudioAndControl(audio=audio_np[i : i + chunk_size], emotion=emotion)
            )
        audio_queue.put(AudioAndControl(flush=True))

    inference_thread = threading.Thread(
        target=agent.run_inference_stream,
        args=(ref_image_path, audio_queue, output_buffer, stop_event),
        kwargs=dict(
            a_cfg_scale=opt.a_cfg_scale,
            r_cfg_scale=opt.r_cfg_scale,
            e_cfg_scale=opt.e_cfg_scale,
            talking_emotion=None,
            idle_emotion=opt.emo,
            nfe=opt.nfe,
            no_crop=opt.no_crop,
            seed=opt.seed,
        ),
    )
    inference_thread.start()

    read_thread = threading.Thread(target=_read_frames, args=(output_buffer,))
    read_thread.start()

    while len(frames) < 25 * 1:
        time.sleep(0.2)

    _send_audio("assets/sample.wav", emotion="surprise")
    time.sleep(1)
    _send_audio(audio_path, interrupt=True, emotion="angry")

    while len(frames) < 25 * 40:
        time.sleep(1)

    stop_event.set()

    read_thread.join()
    inference_thread.join()

    # add reset frames
    frames += output_buffer.buffer

    images = [f.img for f in frames]
    audios = [f.audio for f in frames]
    audios = np.concatenate(audios)
    print("audios", audios.shape, "images", len(images))
    print("image_duration", len(images) / agent.opt.fps)
    print("audio_duration", audios.shape[0] / agent.opt.sampling_rate)

    # write mp4 using moviepy with audio
    write_video(images, audios, agent.opt.fps, agent.opt.sampling_rate, "test.mp4")


def write_video(
    images: list[np.ndarray],
    audios: np.ndarray,
    fps: int,
    sampling_rate: int,
    video_path: str,
):
    """
    Write video with audio using moviepy

    Args:
        images: List of image arrays (H, W, 3) in uint8 format
        audios: Audio array (samples,) in float32 format
        fps: Video frame rate
        sampling_rate: Audio sampling rate
        video_path: Output video file path
    """
    # Create video clip from images
    video_clip = ImageSequenceClip(images, fps=fps)

    # Create audio clip from audio array
    if audios.ndim == 1:
        audios = audios[:, None]

    audio_clip = AudioArrayClip(audios, fps=sampling_rate * 2)

    # Set audio to video
    final_clip = video_clip.with_audio(audio_clip)

    # Write the final video
    final_clip.write_videofile(video_path, codec="libx264")

    # Clean up
    final_clip.close()
    audio_clip.close()
    video_clip.close()


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path", default=None, type=str, help="ref")
        parser.add_argument("--aud_path", default=None, type=str, help="audio")
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="not using crop")
        parser.add_argument(
            "--res_video_path", default=None, type=str, help="res video path"
        )
        parser.add_argument(
            "--ckpt_path",
            default="/home/nvadmin/workspace/taek/float-pytorch/checkpoints/float.pth",
            type=str,
            help="checkpoint path",
        )
        parser.add_argument(
            "--res_dir", default="./results", type=str, help="result dir"
        )
        return parser


if __name__ == "__main__":
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    agent = InferenceAgent(opt)
    os.makedirs(opt.res_dir, exist_ok=True)

    # -------------- input -------------
    ref_path = opt.ref_path
    aud_path = opt.aud_path
    # ----------------------------------

    if opt.res_video_path is None:
        video_name = os.path.splitext(os.path.basename(ref_path))[0]
        audio_name = os.path.splitext(os.path.basename(aud_path))[0]
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        res_video_path = os.path.join(
            opt.res_dir,
            "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s.mp4"
            % (
                call_time,
                video_name,
                audio_name,
                opt.nfe,
                opt.seed,
                opt.a_cfg_scale,
                opt.e_cfg_scale,
                opt.emo,
            ),
        )
    else:
        res_video_path = opt.res_video_path

    # agent.run_inference(
    #     res_video_path,
    #     ref_path,
    #     aud_path,
    #     a_cfg_scale=opt.a_cfg_scale,
    #     r_cfg_scale=opt.r_cfg_scale,
    #     e_cfg_scale=opt.e_cfg_scale,
    #     emo=opt.emo,
    #     nfe=opt.nfe,
    #     no_crop=opt.no_crop,
    #     seed=opt.seed,
    #     verbose=True,
    # )

    main(agent, ref_path, aud_path, opt)

#!/usr/bin/env python3
"""Drag-and-drop interface for the transcriber.

Same pipeline as `transcribe.py`, driven from a browser instead of a shell: drop
an audio or video file, get the transcript and the subtitle files back. Serves
both deployments -- `docker compose up ui` on a GPU box reaches it at
localhost:7860, and the Colab notebook launches the identical app on a borrowed
T4.

The model is loaded once and kept, so the first transcription pays the load cost
and later ones do not.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

import transcribe as T

# Video containers are listed too: PyAV pulls the audio stream out of an mp4 or
# a mov, so a screen recording of a call works without converting it first.
ACCEPTED = [".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac",
            ".mp4", ".mov", ".mkv", ".webm", ".avi"]

# The offline image bakes in exactly one model and sets HF_HUB_OFFLINE=1, so
# offering the others there would hand the user a dropdown whose other entries
# fail at load. Compose passes the baked model in; unset (Colab, a plain venv)
# means downloads work and the full list is honest.
MODEL_CHOICES = [
    name.strip()
    for name in os.environ.get(
        "STT_MODELS", "large-v3,large-v3-turbo,medium,small"
    ).split(",")
    if name.strip()
]

_models: dict[tuple[str, str, str], WhisperModel] = {}


def get_model(name: str, device: str, compute_type: str) -> WhisperModel:
    key = (name, device, compute_type)
    if key not in _models:
        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]


def run(
    audio_file,
    languages: list[str],
    primary: str,
    model_name: str,
    threshold: float,
    window: float,
    split_silence: float,
    beam_size: int,
    progress=gr.Progress(),
):
    if audio_file is None:
        raise gr.Error("Drop a file first.")
    if not languages:
        raise gr.Error("Pick at least one language.")
    if primary not in languages:
        raise gr.Error(f"'{primary}' is the fallback language, so it has to be among the selected ones.")

    source = Path(audio_file)
    started = time.time()

    progress(0.02, desc="Reading the file")
    audio = decode_audio(str(source), sampling_rate=T.SAMPLE_RATE)
    minutes = len(audio) / T.SAMPLE_RATE / 60

    progress(0.06, desc="Finding speech")
    windows = T.group_speech_windows(audio, window, 500, 200, split_silence)
    if not windows:
        raise gr.Error("No speech found. Does the file actually have an audio track?")

    # Chosen after the file is read so the wait is attributable: a slow first
    # run is the download, not the audio.
    progress(0.10, desc=f"Loading {model_name}")
    device = "cuda" if _cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = get_model(model_name, device, compute_type)

    lines: list[T.Line] = []
    unsure = 0
    for index, (start, end) in enumerate(windows):
        progress(
            0.10 + 0.88 * index / len(windows),
            desc=f"Transcribing {index + 1}/{len(windows)}",
        )
        produced = T.transcribe(
            model, audio, [(start, end)], languages, primary, threshold, beam_size
        )
        # Counted per window, not per line: one window emits several lines that
        # all carry its single classification, so counting lines would report
        # more uncertain windows than there are windows.
        if produced and produced[0].language_probability < 0.75:
            unsure += 1
        lines.extend(produced)

    if not lines:
        raise gr.Error("Speech was found but nothing was transcribed. Try a different model size.")

    progress(0.99, desc="Writing files")
    out_dir = Path(tempfile.mkdtemp(prefix="stt_"))
    written = T.write_outputs(lines, out_dir / source.stem)

    spread: dict[str, int] = {}
    for line in lines:
        spread[line.language] = spread.get(line.language, 0) + 1

    elapsed = (time.time() - started) / 60
    summary = "\n".join([
        f"{minutes:.1f} min of audio, {len(windows)} windows, {len(lines)} lines",
        f"languages: " + ", ".join(f"{k} {v}" for k, v in sorted(spread.items())),
        f"low-confidence windows: {unsure} of {len(windows)}"
        + (" — check those lines in the JSON" if unsure else ""),
        f"took {elapsed:.1f} min on {device}",
    ])

    transcript = "\n".join(
        f"[{line.language}] {line.text}" if len(spread) > 1 else line.text
        for line in lines
    )
    return transcript, [str(p) for p in written], summary


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def build() -> gr.Blocks:
    with gr.Blocks(title="STT_MultiLingva") as ui:
        gr.Markdown(
            "# STT_MultiLingva\n"
            "Drop a recording, get the transcript. Audio is processed by this "
            "process and is not sent anywhere.\n\n"
            "Whisper otherwise decides the language once and applies it to the "
            "whole file; this classifies every speech window separately, so a "
            "meeting that switches between languages does not come out in one."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_file = gr.File(
                    label="Recording", file_types=ACCEPTED, type="filepath"
                )
                languages = gr.CheckboxGroup(
                    ["ru", "en", "hy"],
                    value=["ru", "en"],
                    label="Languages to expect",
                    info="Narrow this to what is actually spoken — a wider set means more chances to guess wrong.",
                )
                primary = gr.Radio(
                    ["ru", "en", "hy"],
                    value="ru",
                    label="Fallback language",
                    info="Used for windows too short or unclear to classify.",
                )
                go = gr.Button("Transcribe", variant="primary")

                with gr.Accordion("Tuning", open=False):
                    model_name = gr.Dropdown(
                        MODEL_CHOICES,
                        value=MODEL_CHOICES[0],
                        label="Model",
                        info="large-v3 for quality. turbo is much faster but weaker on Armenian.",
                    )
                    threshold = gr.Slider(
                        0.3, 0.95, value=0.6, step=0.05,
                        label="Confidence threshold",
                        info="Below this a window falls back. Raise if wrong languages appear.",
                    )
                    split_silence = gr.Slider(
                        0.2, 2.0, value=0.7, step=0.1,
                        label="Pause that ends a window (s)",
                        info="Lower it if two speakers in different languages end up merged.",
                    )
                    window = gr.Slider(
                        5.0, 30.0, value=30.0, step=1.0,
                        label="Max window (s)",
                    )
                    beam_size = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Beam size",
                        info="1 is a fast rough pass over a long file.",
                    )

            with gr.Column(scale=2):
                summary = gr.Textbox(label="Run", lines=4, interactive=False)
                transcript = gr.Textbox(label="Transcript", lines=26)
                downloads = gr.File(label="SRT / TXT / JSON")

        go.click(
            run,
            inputs=[audio_file, languages, primary, model_name, threshold,
                    window, split_silence, beam_size],
            outputs=[transcript, downloads, summary],
        )

    return ui


if __name__ == "__main__":
    share = "--share" in sys.argv
    build().queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        # Colab needs a tunnel to be reachable; a local box does not, and
        # opening one there would publish the app to the internet.
        share=share,
        max_file_size="2gb",
    )

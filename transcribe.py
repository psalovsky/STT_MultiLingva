#!/usr/bin/env python3
"""Transcribe a multilingual meeting recording entirely on local hardware.

Whisper detects the language once, from the first window, and then treats the
whole recording as monolingual. On a meeting where people switch between
Russian and English that locks onto whichever language opened the call and the
other one comes out transliterated, translated, or dropped.

This runs language detection per speech window instead: voice activity
detection splits the audio, each window is classified against an allowed set of
languages, and only then transcribed with that language pinned. Windows the
classifier is unsure about fall back to the primary language rather than
guessing.

Nothing leaves the machine. Model weights are read from a local directory or
the HuggingFace cache; set HF_HUB_OFFLINE=1 once they are in place to make that
guarantee enforceable rather than a promise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

SAMPLE_RATE = 16000


@dataclass
class Line:
    start: float
    end: float
    language: str
    language_probability: float
    text: str
    speaker: str | None = None


def group_speech_windows(
    audio: np.ndarray,
    max_window: float,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[tuple[int, int]]:
    """Merge VAD speech regions into windows of at most `max_window` seconds.

    Language detection needs a few seconds of speech to be reliable, and Whisper
    decodes 30s at a time, so isolated VAD regions are merged up to that bound.
    A window is cut at a silence boundary, never mid-speech, so a window never
    straddles the point where a speaker switches language.
    """
    regions = get_speech_timestamps(
        audio,
        VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms),
        sampling_rate=SAMPLE_RATE,
    )
    if not regions:
        return []

    limit = int(max_window * SAMPLE_RATE)
    windows: list[tuple[int, int]] = []
    start, end = regions[0]["start"], regions[0]["end"]

    for region in regions[1:]:
        if region["end"] - start <= limit:
            end = region["end"]
        else:
            windows.append((start, end))
            start, end = region["start"], region["end"]
    windows.append((start, end))
    return windows


def pick_language(
    model: WhisperModel,
    chunk: np.ndarray,
    allowed: list[str],
    primary: str,
    threshold: float,
) -> tuple[str, float]:
    """Classify one window, restricted to the languages we expect to hear.

    Whisper ranks all 99 languages it knows. On a Russian meeting that regularly
    surfaces Ukrainian or Bulgarian as a near-miss, so the ranking is filtered to
    the allowed set before picking a winner. Below `threshold` the window is not
    distinctive enough to trust -- short interjections score badly no matter the
    model -- and the primary language is used instead.
    """
    _, _, all_probs = model.detect_language(audio=chunk)
    scores = {lang: prob for lang, prob in all_probs if lang in allowed}
    if not scores:
        return primary, 0.0

    best = max(scores, key=scores.get)
    if scores[best] < threshold:
        return primary, scores[best]
    return best, scores[best]


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    windows: list[tuple[int, int]],
    allowed: list[str],
    primary: str,
    threshold: float,
    beam_size: int,
) -> list[Line]:
    lines: list[Line] = []

    for index, (start, end) in enumerate(windows, start=1):
        chunk = audio[start:end]
        offset = start / SAMPLE_RATE

        language, probability = pick_language(model, chunk, allowed, primary, threshold)
        segments, _ = model.transcribe(
            chunk,
            language=language,
            beam_size=beam_size,
            # The window is already known to be speech, and each is decoded in
            # isolation, so carrying text across windows would let one bad
            # decode seed the next.
            condition_on_previous_text=False,
        )

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            lines.append(
                Line(
                    start=offset + segment.start,
                    end=offset + segment.end,
                    language=language,
                    language_probability=round(probability, 3),
                    text=text,
                )
            )

        print(
            f"  [{index}/{len(windows)}] {offset:7.1f}s  {language}  p={probability:.2f}",
            file=sys.stderr,
        )

    return lines


def attach_speakers(lines: list[Line], audio_path: Path, hf_token: str | None) -> bool:
    """Label each line with a speaker, if pyannote is installed and available.

    Kept optional on purpose: pyannote's weights are gated on HuggingFace and
    need a one-time authenticated download, which some environments will not
    permit. Transcription proceeds without it.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return False

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in pipeline(str(audio_path)).itertracks(yield_label=True)
    ]

    for line in lines:
        midpoint = (line.start + line.end) / 2
        # A line can straddle a speaker change; the midpoint attributes it to
        # whoever holds the floor for most of it.
        match = next((s for start, end, s in turns if start <= midpoint <= end), None)
        line.speaker = match

    return True


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_outputs(lines: list[Line], stem: Path) -> list[Path]:
    srt = stem.with_suffix(".srt")
    with srt.open("w", encoding="utf-8") as handle:
        for index, line in enumerate(lines, start=1):
            speaker = f"[{line.speaker}] " if line.speaker else ""
            handle.write(
                f"{index}\n"
                f"{format_timestamp(line.start)} --> {format_timestamp(line.end)}\n"
                f"{speaker}{line.text}\n\n"
            )

    txt = stem.with_suffix(".txt")
    with txt.open("w", encoding="utf-8") as handle:
        for line in lines:
            speaker = f"[{line.speaker}] " if line.speaker else ""
            handle.write(f"{speaker}{line.text}\n")

    js = stem.with_suffix(".json")
    js.write_text(
        json.dumps([asdict(line) for line in lines], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return [srt, txt, js]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("audio", type=Path, help="input recording, any format PyAV reads")
    parser.add_argument("--model", default="large-v3", help="size or path to a local model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="default",
        help="float16 on a GPU, int8 on CPU; 'default' lets CTranslate2 choose",
    )
    parser.add_argument(
        "--languages",
        default="ru,en",
        help="languages to consider, comma separated; add hy for Armenian",
    )
    parser.add_argument(
        "--primary",
        default="ru",
        help="fallback for windows the classifier is unsure about",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="minimum confidence")
    parser.add_argument("--window", type=float, default=30.0, help="max window, seconds")
    parser.add_argument("--min-silence-ms", type=int, default=500)
    parser.add_argument("--speech-pad-ms", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--diarize", action="store_true", help="label speakers via pyannote")
    parser.add_argument("--hf-token", default=None, help="only used by --diarize")
    parser.add_argument("--out", type=Path, default=None, help="output stem")
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"no such file: {args.audio}")

    allowed = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if args.primary not in allowed:
        parser.error(f"--primary {args.primary} is not in --languages {allowed}")

    # Make the output directory now, not after transcribing. A missing one is
    # otherwise discovered only at the write, which on a long recording is
    # hours of GPU time thrown away.
    stem = args.out or args.audio.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Decoding {args.audio}", file=sys.stderr)
    audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
    print(f"  {len(audio) / SAMPLE_RATE / 60:.1f} minutes", file=sys.stderr)

    windows = group_speech_windows(
        audio, args.window, args.min_silence_ms, args.speech_pad_ms
    )
    if not windows:
        print("No speech found. Check that the file has an audio track.", file=sys.stderr)
        return 1
    speech = sum(end - start for start, end in windows) / SAMPLE_RATE
    print(f"  {len(windows)} windows, {speech / 60:.1f} minutes of speech", file=sys.stderr)

    print(f"Loading {args.model} on {args.device}", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    lines = transcribe(
        model, audio, windows, allowed, args.primary, args.threshold, args.beam_size
    )

    if args.diarize and not attach_speakers(lines, args.audio, args.hf_token):
        print("pyannote.audio is not installed; skipping speaker labels", file=sys.stderr)

    written = write_outputs(lines, stem)

    counts: dict[str, int] = {}
    for line in lines:
        counts[line.language] = counts.get(line.language, 0) + 1
    spread = ", ".join(f"{lang} {n}" for lang, n in sorted(counts.items()))
    print(f"\n{len(lines)} lines ({spread})", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

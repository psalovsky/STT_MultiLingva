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
import math
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
    split_silence: float,
) -> list[tuple[int, int]]:
    """Merge VAD speech regions into windows of at most `max_window` seconds.

    Language detection needs a few seconds of speech to be reliable, so isolated
    VAD regions are merged -- but only across pauses short enough to be breath
    within one person's turn. A gap of `split_silence` or more ends the window
    regardless of how much room is left, because that is where a speaker change
    happens, and a window spanning one is decoded entirely in whichever language
    its opening utterance was classified as. Merging on window room alone would
    swallow exactly the Russian-pause-English sequence this tool exists for.
    """
    # Padding is deliberately switched off here and applied at the end. The VAD
    # grows every region by speech_pad_ms on each side, and where the silence
    # between two regions is shorter than twice that, it splits the difference
    # instead -- so the gap this function can see is the real pause minus 0.4s,
    # or zero. Measuring --split-silence against that would silently require a
    # 1.1s pause to mean 0.7, and 0.7-1.1s is ordinary turn-taking: exactly the
    # switch this is supposed to catch.
    regions = get_speech_timestamps(
        audio,
        VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=0),
        sampling_rate=SAMPLE_RATE,
    )
    if not regions:
        return []

    limit = int(max_window * SAMPLE_RATE)
    split_gap = int(split_silence * SAMPLE_RATE)
    pad = int(speech_pad_ms / 1000 * SAMPLE_RATE)
    windows: list[tuple[int, int]] = []
    start, end = regions[0]["start"], regions[0]["end"]

    for region in regions[1:]:
        gap = region["start"] - end
        if gap < split_gap and region["end"] - start <= limit:
            end = region["end"]
        else:
            windows.extend(split_oversized(start, end, limit))
            start, end = region["start"], region["end"]
    windows.extend(split_oversized(start, end, limit))

    # Now restore the padding the VAD would have added, so a decoded window
    # still carries the leading and trailing moment that keeps Whisper from
    # clipping the first and last word.
    ceiling = len(audio)
    return [(max(0, s - pad), min(ceiling, e + pad)) for s, e in windows]


def split_oversized(start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Cut a stretch of uninterrupted speech down to the window limit.

    A speaker can hold the floor for minutes without a pause the VAD will call a
    boundary. Such a region has to be cut somewhere arbitrary, because the
    alternative is one enormous window whose language is decided by its opening
    seconds. Pieces are evenly sized rather than limit-then-remainder, which
    would leave a final sliver too short to classify.
    """
    span = end - start
    if span <= limit:
        return [(start, end)]

    pieces = math.ceil(span / limit)
    step = math.ceil(span / pieces)
    return [(cut, min(cut + step, end)) for cut in range(start, end, step)]


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
        # The primary's own score, not the rejected candidate's. Reporting
        # en=0.55 as "ru, p=0.55" would make the JSON say the classifier was
        # fairly confident about Russian when it was not confident about
        # anything -- and that field is what you tune --threshold against.
        return primary, scores.get(primary, 0.0)
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


def attach_speakers(
    lines: list[Line], audio_path: Path, hf_token: str | None
) -> str | None:
    """Label each line with a speaker. Returns why it could not, or None.

    Kept optional on purpose: pyannote's weights are gated on HuggingFace and
    need a one-time authenticated download, which some environments will not
    permit. Missing weights, a missing token, ungranted access and an empty
    offline cache all surface here, and by this point the transcription is
    already done -- so every one of them is reported and swallowed rather than
    allowed to discard the expensive part of the run.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return "pyannote.audio is not installed"

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        # pyannote answers an auth failure with None instead of raising.
        if pipeline is None:
            return "pyannote returned no pipeline (check the token and model access)"
        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in pipeline(str(audio_path)).itertracks(yield_label=True)
        ]
    except Exception as error:
        return f"diarization failed ({type(error).__name__}: {error})"

    for line in lines:
        # Attribute the line to whoever actually holds most of it. Sampling a
        # single instant instead would hand the whole line to a two-word
        # interjection that happens to land there, and pyannote turns can
        # overlap, so the first match is whichever came out of the iterator.
        held: dict[str, float] = {}
        for start, end, speaker in turns:
            shared = min(line.end, end) - max(line.start, start)
            if shared > 0:
                held[speaker] = held.get(speaker, 0.0) + shared
        line.speaker = max(held, key=held.get) if held else None

    return None


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def beside(stem: Path, extension: str) -> Path:
    """Append an extension to the stem instead of replacing what looks like one.

    Path.with_suffix() would replace the last dotted part, so meeting.en.wav and
    meeting.ru.wav both reduce to meeting.srt and the second run silently
    overwrites the first -- and notes.v2.final.m4a loses ".final" outright.
    """
    return stem.parent / (stem.name + extension)


def write_outputs(lines: list[Line], stem: Path) -> list[Path]:
    srt = beside(stem, ".srt")
    with srt.open("w", encoding="utf-8") as handle:
        for index, line in enumerate(lines, start=1):
            speaker = f"[{line.speaker}] " if line.speaker else ""
            handle.write(
                f"{index}\n"
                f"{format_timestamp(line.start)} --> {format_timestamp(line.end)}\n"
                f"{speaker}{line.text}\n\n"
            )

    txt = beside(stem, ".txt")
    with txt.open("w", encoding="utf-8") as handle:
        for line in lines:
            speaker = f"[{line.speaker}] " if line.speaker else ""
            handle.write(f"{speaker}{line.text}\n")

    js = beside(stem, ".json")
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
    parser.add_argument(
        "--split-silence",
        type=float,
        default=0.7,
        help="a pause this long ends the window; a speaker change lives here",
    )
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
        audio, args.window, args.min_silence_ms, args.speech_pad_ms, args.split_silence
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

    if args.diarize:
        problem = attach_speakers(lines, args.audio, args.hf_token)
        if problem:
            print(f"No speaker labels: {problem}", file=sys.stderr)

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

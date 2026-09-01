"""Exercise the pipeline without model weights.

Everything except the transcription itself is checkable on CPU with no
downloads: the windowing decisions, the language fallbacks, the offset
arithmetic, and the guarantee that a finished transcription survives whatever
diarization does. Run with `python stub_test.py`.
"""

import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

import transcribe as T

SR = T.SAMPLE_RATE
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def windows_from(regions, max_window=30.0, split_silence=0.7, pad_ms=200):
    """Drive the windowing with a fixed VAD result instead of real audio.

    The fake stands in for an unpadded VAD, which is what the code now asks for:
    regions are the true speech boundaries and the gaps between them are real
    pause lengths.
    """
    original = T.get_speech_timestamps
    seen = {}

    def fake(audio, vad_options=None, **k):
        seen["pad"] = vad_options.speech_pad_ms
        return [{"start": int(s * SR), "end": int(e * SR)} for s, e in regions]

    T.get_speech_timestamps = fake
    try:
        raw = T.group_speech_windows(
            np.zeros(SR * 600, dtype=np.float32), max_window, 500, pad_ms, split_silence
        )
    finally:
        T.get_speech_timestamps = original
    return [(round(s / SR, 2), round(e / SR, 2)) for s, e in raw], seen.get("pad")


print("\nwindowing")

# The case the tool exists for: Russian, a turn-taking pause, then English.
# Both fit inside one 30s window, so merging on window room alone would put
# them in one chunk and decode the English in Russian.
w, pad_asked = windows_from([(0, 5), (7, 12)])
check("a turn-taking pause splits the window", len(w) == 2, w)
check("the VAD is asked for unpadded regions", pad_asked == 0, pad_asked)

# The gap the code compares against --split-silence has to be the real pause.
# With padding left on, the VAD reports this 0.8s gap as 0.4s and it merges --
# and anything under 0.4s as zero.
w, _ = windows_from([(0, 5), (5.8, 12)], split_silence=0.7)
check("a 0.8s pause splits, not only a 1.1s one", len(w) == 2, w)

# Padding is restored afterwards, so a decoded window still carries the moment
# either side that keeps Whisper from clipping the first and last word.
w, _ = windows_from([(2, 5)], pad_ms=200)
check("padding is applied to the finished window", w == [(1.8, 5.2)], w)
check("padding cannot run past zero", windows_from([(0.1, 5)], pad_ms=200)[0][0][0] == 0.0)

# A breath inside one person's turn should not fragment the window: short
# pieces classify badly, which is the reason merging exists at all.
w, _ = windows_from([(0, 5), (5.3, 12)], pad_ms=0)
check("a breath does not split the window", w == [(0.0, 12.0)], w)

# One speaker holding the floor past the limit still has to be cut, or the
# window's language is decided by its opening seconds.
w, _ = windows_from([(0, 95)], max_window=30.0, pad_ms=0)
check("an oversized region is split", len(w) == 4, w)
check("every piece is within the limit", all(e - s <= 30.0 for s, e in w), w)
check("pieces are contiguous", all(round(w[i][1], 2) == round(w[i + 1][0], 2) for i in range(len(w) - 1)), w)
check("pieces cover the region", w[0][0] == 0.0 and w[-1][1] == 95.0, w)
check("no sliver at the end", w[-1][1] - w[-1][0] > 5.0, w)

check("silence only yields no windows", windows_from([])[0] == [])
check("a single short region is one window", windows_from([(0, 4)], pad_ms=0)[0] == [(0.0, 4.0)])


print("\nlanguage selection")


@dataclass
class Seg:
    start: float
    end: float
    text: str


class FakeModel:
    def __init__(self, *a, **k):
        self.calls = []

    def detect_language(self, audio=None, **k):
        table = [("en", 0.94), ("ru", 0.88), ("bg", 0.31)]
        lang, p = table[len(self.calls) % len(table)]
        return lang, p, [(lang, p), ("uk", 0.05)]

    def transcribe(self, audio, language=None, **k):
        self.calls.append(language)
        dur = len(audio) / SR
        return [Seg(0.0, dur / 2, f"first in {language}"),
                Seg(dur / 2, dur, f"second in {language}")], None


model = FakeModel()
lang, prob = T.pick_language(model, np.zeros(SR), ["ru", "en"], "ru", 0.6)
check("a confident allowed language wins", (lang, prob) == ("en", 0.94), (lang, prob))

model.calls = [None, None]  # advance the fake to the "bg" row
lang, prob = T.pick_language(model, np.zeros(SR), ["ru", "en"], "ru", 0.6)
check("a language outside the allowed set falls back", lang == "ru", lang)


class Unsure:
    """Best allowed candidate is en=0.55, under the threshold; ru sits at 0.10."""

    def detect_language(self, audio=None, **k):
        return "en", 0.55, [("en", 0.55), ("ru", 0.10)]


lang, prob = T.pick_language(Unsure(), np.zeros(SR), ["ru", "en"], "ru", 0.6)
check("a rejected candidate's score is not reported as the primary's",
      (lang, prob) == ("ru", 0.10), (lang, prob))


print("\ndiarization must never cost the transcription")

lines = [T.Line(0.0, 1.0, "ru", 0.9, "text")]


class Boom:
    @staticmethod
    def from_pretrained(*a, **k):
        raise RuntimeError("gated repo: access not granted")


sys.modules["pyannote"] = type(sys)("pyannote")
sys.modules["pyannote.audio"] = type(sys)("pyannote.audio")
sys.modules["pyannote.audio"].Pipeline = Boom

problem = T.attach_speakers(lines, "x.wav", None)
check("a gated-weights failure is reported, not raised", problem is not None, problem)
check("the failure names the cause", "access not granted" in (problem or ""), problem)


class ReturnsNone:
    @staticmethod
    def from_pretrained(*a, **k):
        return None


sys.modules["pyannote.audio"].Pipeline = ReturnsNone
problem = T.attach_speakers(lines, "x.wav", None)
check("a None pipeline is reported, not dereferenced", problem is not None, problem)


class Turn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class Diarization:
    """A holds 9 of the line's 10 seconds; B's one second covers the midpoint."""

    def itertracks(self, yield_label=True):
        return [(Turn(0.0, 4.5), None, "A"),
                (Turn(4.5, 5.5), None, "B"),
                (Turn(5.5, 10.0), None, "A")]


class Working:
    @staticmethod
    def from_pretrained(*a, **k):
        return lambda path: Diarization()


sys.modules["pyannote.audio"].Pipeline = Working
spoken = [T.Line(0.0, 10.0, "ru", 0.9, "one long line")]
problem = T.attach_speakers(spoken, "x.wav", None)
check("diarization succeeds", problem is None, problem)
check("the speaker holding most of the line wins, not the one at its midpoint",
      spoken[0].speaker == "A", spoken[0].speaker)


print("\noutput naming")

# with_suffix() would replace what looks like an extension, so these two collide
# on meeting.srt and the second run overwrites the first.
en = T.beside(pathlib.Path("/tmp/meeting.en"), ".srt")
ru = T.beside(pathlib.Path("/tmp/meeting.ru"), ".srt")
check("language-suffixed names do not collide", en != ru, (en, ru))
check("the whole stem is kept",
      T.beside(pathlib.Path("/tmp/notes.v2.final"), ".json").name == "notes.v2.final.json",
      T.beside(pathlib.Path("/tmp/notes.v2.final"), ".json").name)
check("a plain name is unchanged",
      T.beside(pathlib.Path("/tmp/call"), ".txt").name == "call.txt")


print("\nend to end with a stubbed model")

import wave

speech = np.concatenate([
    np.random.randn(int(SR * 5)) * 0.3,
    np.zeros(int(SR * 1.5)),
    np.random.randn(int(SR * 5)) * 0.3,
])
with wave.open("/tmp/stub.wav", "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((np.clip(speech, -1, 1) * 32767).astype(np.int16).tobytes())

T.WhisperModel = FakeModel
T.get_speech_timestamps = lambda *a, **k: [
    {"start": 0, "end": int(SR * 5)},
    {"start": int(SR * 6.5), "end": int(SR * 11.5)},
]
sys.argv = ["transcribe.py", "/tmp/stub.wav", "--languages", "ru,en", "--primary", "ru",
            "--window", "30", "--out", "/tmp/stubout/run"]
rc = T.main()
check("the run succeeds", rc == 0, rc)

out = json.load(open("/tmp/stubout/run.json"))
starts = [l["start"] for l in out]
check("offsets are monotonic", starts == sorted(starts), starts)
# The second region begins at 6.5s and the finished window is padded 200ms
# ahead of it, so its first line starts at 6.3 -- not back at zero, which is
# what a missing offset would look like.
check("the second window is offset, not restarted",
      6.2 <= starts[2] <= 6.6, starts)
check("output files are written", all(
    __import__("os").path.exists(f"/tmp/stubout/run{ext}") for ext in (".srt", ".txt", ".json")))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("all checks passed")

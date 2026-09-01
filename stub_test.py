"""Exercise the pipeline without model weights.

Everything except the transcription itself is checkable on CPU with no
downloads: the windowing decisions, the language fallbacks, the offset
arithmetic, and the guarantee that a finished transcription survives whatever
diarization does. Run with `python stub_test.py`.
"""

import json
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


def windows_from(regions, max_window=30.0, split_silence=0.7):
    """Drive the windowing with a fixed VAD result instead of real audio."""
    original = T.get_speech_timestamps
    T.get_speech_timestamps = lambda *a, **k: [
        {"start": int(s * SR), "end": int(e * SR)} for s, e in regions
    ]
    try:
        raw = T.group_speech_windows(
            np.zeros(SR * 600, dtype=np.float32), max_window, 500, 200, split_silence
        )
    finally:
        T.get_speech_timestamps = original
    return [(round(s / SR, 2), round(e / SR, 2)) for s, e in raw]


print("\nwindowing")

# The case the tool exists for: Russian, a turn-taking pause, then English.
# Both fit inside one 30s window, so merging on window room alone would put
# them in one chunk and decode the English in Russian.
w = windows_from([(0, 5), (7, 12)])
check("a turn-taking pause splits the window", len(w) == 2, w)

# A breath inside one person's turn should not fragment the window: short
# pieces classify badly, which is the reason merging exists at all.
w = windows_from([(0, 5), (5.3, 12)])
check("a breath does not split the window", w == [(0.0, 12.0)], w)

# One speaker holding the floor past the limit still has to be cut, or the
# window's language is decided by its opening seconds.
w = windows_from([(0, 95)], max_window=30.0)
check("an oversized region is split", len(w) == 4, w)
check("every piece is within the limit", all(e - s <= 30.0 for s, e in w), w)
check("pieces are contiguous", all(round(w[i][1], 2) == round(w[i + 1][0], 2) for i in range(len(w) - 1)), w)
check("pieces cover the region", w[0][0] == 0.0 and w[-1][1] == 95.0, w)
check("no sliver at the end", w[-1][1] - w[-1][0] > 5.0, w)

check("silence only yields no windows", windows_from([]) == [])
check("a single short region is one window", windows_from([(0, 4)]) == [(0.0, 4.0)])


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
check("the second window is offset, not restarted", starts[2] >= 6.5, starts)
check("output files are written", all(
    __import__("os").path.exists(f"/tmp/stubout/run{ext}") for ext in (".srt", ".txt", ".json")))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("all checks passed")

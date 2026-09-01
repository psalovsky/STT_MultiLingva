"""Прогон всего пайплайна с подменённой моделью: проверяем окна, оффсеты, формат."""
import sys, types, json
from dataclasses import dataclass
import transcribe as T

@dataclass
class Seg:
    start: float; end: float; text: str

class FakeModel:
    """Чередует языки по окнам, чтобы проверить и ветку порога, и подсчёт."""
    def __init__(self,*a,**k): self.calls=[]
    def detect_language(self, audio=None, **k):
        n=len(self.calls)
        table=[("en",0.94),("ru",0.88),("bg",0.31)]   # третье — ниже порога → fallback
        lang,p = table[n % len(table)]
        return lang, p, [(lang,p),("uk",0.05)]
    def transcribe(self, audio, language=None, **k):
        self.calls.append(language)
        dur=len(audio)/T.SAMPLE_RATE
        return [Seg(0.0, dur/2, f"first half in {language}"),
                Seg(dur/2, dur, f"second half in {language}")], None

T.WhisperModel = FakeModel
sys.argv = ["transcribe.py","meeting.wav","--languages","ru,en","--primary","ru",
            "--threshold","0.6","--window","15","--out","out/stub"]
rc = T.main()
print(f"\nexit={rc}")

lines=json.load(open("out/stub.json"))
print(f"строк: {len(lines)}")
langs=[l["language"] for l in lines]
print("языки по строкам:", langs)
starts=[round(l["start"],2) for l in lines]
print("старты:", starts)
assert rc==0
assert starts==sorted(starts), "оффсеты не монотонны!"
assert starts[0]>=0
assert lines[-1]["end"] > lines[0]["end"], "конец последней строки не больше первой"
# окно 3 детектится как bg (0.31 < 0.6) → должен сработать fallback на ru
assert "ru" in langs, "fallback на primary не сработал"
assert "bg" not in langs, "язык вне allowed просочился в вывод"
print("\nSRT (первые 8 строк):")
print("\n".join(open("out/stub.srt").read().splitlines()[:8]))
print("\nВСЕ ПРОВЕРКИ ПРОШЛИ")

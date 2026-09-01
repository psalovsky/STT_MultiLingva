# STT_MultiLingva — offline meeting transcription, RU / EN / HY

Transcribes a multilingual meeting on your own hardware. No audio leaves the
machine, and no external API is involved.

## Why not just run Whisper

Whisper detects the language once, from the opening window, and then treats the
whole recording as monolingual. On a call that switches between Russian and
English that locks onto whichever language opened it, and the other one comes
out transliterated, machine-translated into the first, or silently dropped.

`transcribe.py` classifies each speech window separately. Voice activity
detection finds the speech, and windows end at any pause of `--split-silence` or
more — that is where turn-taking happens, and a window spanning a speaker change
gets decoded entirely in whichever language its opening utterance was. Shorter
pauses are merged over, because a two-second fragment classifies badly. Each
window is then classified against the languages you say to expect and
transcribed with that one pinned.

Two details that matter in practice:

- The classifier's ranking covers all 99 languages Whisper knows. On a Russian
  meeting it regularly surfaces Ukrainian or Bulgarian as a near-miss, so the
  ranking is filtered to your allowed set before a winner is picked.
- Short interjections score badly no matter the model. Below `--threshold` the
  window falls back to `--primary` rather than acting on a coin flip.

## Deploy on a GPU box

```bash
mkdir -p audio out && cp /path/to/meeting.m4a audio/
docker compose build      # downloads large-v3 once, bakes it into the image
docker compose run --rm transcribe
```

Outputs land in `out/` as `.srt` (timestamps), `.txt` (plain), and `.json`
(per-line language and confidence, useful for spotting where detection was
shaky).

`HF_HUB_OFFLINE=1` is set in compose. Weights are already in the image, so the
container refuses any network call for them — the offline guarantee is enforced
rather than assumed. If your runtime perimeter has no egress at all, drop the
download line from the Dockerfile and mount a pre-populated `/models` volume.

## Run on Colab

`colab_stt_multilingva.ipynb` runs the same script on a free T4 — useful when
there is no GPU to hand and you want to see the tool work.

It is not a substitute for the Docker path on a confidential recording. Colab is
Google infrastructure: the model runs locally to the runtime, but the audio you
upload has left your machine and is on Google's servers. That is the disclosure
avoiding a transcription API was meant to prevent. Use Colab for audio you are
free to share, and your own GPU for the rest.

## Run without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python transcribe.py meeting.m4a --languages ru,en --primary ru
```

Input goes through PyAV, so any format works and the `ffmpeg` CLI is not needed.

## Sizing

One mid-range GPU is plenty. On an L4 or A10 with `large-v3`, expect roughly
40–70× real time — a four-hour meeting in well under ten minutes. A T4 works
too, slower. CPU-only runs at roughly real time: possible, not pleasant.

`--compute-type float16` on a GPU, `int8` on CPU. `default` lets CTranslate2
choose for the device.

## Armenian

`large-v3` handles Armenian out of the box at about 15.75% WER across dialects
— better than commercial APIs, but clearly worse than its Russian and English.
Add it with `--languages ru,en,hy`.

If Armenian is a large share of the recording, transcribe twice: once as above,
then re-run the windows the JSON marks `hy` through a fine-tuned model
(`Chillarmo/whisper-large-v3-turbo-armenian`, 15.31% WER / 2.86% CER) by
passing its local path to `--model`.

## Speaker labels

`--diarize` attaches speaker labels via pyannote. It is commented out of
`requirements.txt` on purpose: the weights are gated on HuggingFace and need a
one-time authenticated download. That download moves weights to you, not audio
away from you — but if even that is disallowed, fetch them on a connected
machine and copy them across. Transcription works without it; you just get no
"who said what".

## Tuning

| Flag | Default | Reach for it when |
| --- | --- | --- |
| `--threshold` | `0.6` | Too much lands on the primary language → lower it. Wrong languages appear → raise it. |
| `--split-silence` | `0.7` | Two speakers in different languages land in one window → lower it. Single sentences fragment into unclassifiable pieces → raise it. |
| `--window` | `30.0` | Caps a window even with no qualifying pause; a monologue longer than this is cut into even pieces. |
| `--min-silence-ms` | `500` | Windows cut mid-sentence → raise it. |
| `--beam-size` | `5` | Lower to 1 for a fast first pass over a long file. |

Run a five-minute excerpt first (`ffmpeg -i meeting.m4a -t 300 excerpt.wav`),
check the language spread the run prints at the end, then commit to the whole
file.

## What has and has not been tested

Verified on CPU with a stubbed model (`stub_test.py`, 21 checks): windowing
across turn-taking pauses and breaths, splitting an oversized monologue, the
allowed-set filter, the fallback path, diarization failures leaving the
transcription intact, timestamp offset arithmetic across windows, and
SRT/TXT/JSON output.

Not verified here: transcription quality itself. The environment this was built
in blocks huggingface.co, so real weights were never loaded. Run the excerpt
first.

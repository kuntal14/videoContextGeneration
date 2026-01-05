from pathlib import Path
import subprocess
import shutil
import os
import torch, csv
from faster_whisper import WhisperModel
from typing import List



# function to demux audio
def demux_audio(video_path, AUDIO_FOLDER, AUDIO_PATH):
    # check if the audio folder exists, if so delete it
    if os.path.exists(AUDIO_FOLDER):
        shutil.rmtree(AUDIO_FOLDER)
    
    # add the audio folder
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    # demux using subprocess
    try:
        """Demux the audio from the video file."""
        subprocess.run(["ffmpeg", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(AUDIO_PATH)])
        print(f"Audio demuxed and saved to {AUDIO_PATH}")
    except Exception as e:
        print(f"Error demuxing audio: {e}")

# function to transcribe audio
def transcribe_audio(AUDIO_PATH, CSV_PATH):
    # set the variables
    MODEL_SIZE = "large-v2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    print(f"🚀 Device: {DEVICE} | Model: {MODEL_SIZE}\n\n LOADING MODEL ------------")

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("\n\n TRANSCRIBING AUDIO ------------")

    import time
    start_time = time.time()
    segments, info = model.transcribe(
        str(AUDIO_PATH),
        beam_size=5,
        best_of=5,
        word_timestamps=True,
        language="en"  # or omit for auto-detect
    )
    elapsed = time.time() - start_time
    print(f"\n\n⚡ Time: {elapsed:.1f}s | Audio len: {info.duration:.1f}s → {(info.duration/elapsed):.2f}× realtime")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_idx", "word_idx", "word", "start_sec", "end_sec"])
        for seg_idx, seg in enumerate(segments):
            for w_idx, w in enumerate(seg.words):
                writer.writerow([seg_idx, w_idx, w.word,
                                 f"{w.start:.3f}", f"{w.end:.3f}"])
    print(f"✅ CSV written to {CSV_PATH}")

# convert csv to srt
def csv_to_srt(csv_path: Path, srt_path: Path,
               max_words: int = 7,
               max_seconds: float = 2.5):
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    def fmt(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    cur = {"start": None, "end": None, "words": []}
    prev_end = 0.0
    for r in rows:
        w_start = float(r["start_sec"])
        w_end   = float(r["end_sec"])
        word    = r["word"]

        if cur["start"] is None:
            cur["start"] = w_start

        # Break when we exceed limits
        if (len(cur["words"]) >= max_words) or (w_end - cur["start"] > max_seconds):
            cur["end"] = prev_end
            blocks.append(cur)
            cur = {"start": w_start, "end": None, "words": []}
        cur["words"].append(word)
        prev_end = w_end

    if cur["words"]:
        cur["end"] = prev_end
        blocks.append(cur)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, b in enumerate(blocks, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt(b['start'])} --> {fmt(b['end'])}\n")
            f.write(" ".join(b["words"]) + "\n\n")
    print(f"✅ SRT saved to {srt_path}")
    


def csv_to_transcript(csv_path: str,
                      word_col: str = "word",
                      seg_col: str = "segment_idx",
                      idx_col: str = "word_idx") -> str:
    """
    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the Whisper CSV that contains at least the columns:
        segment_idx, word_idx, <word_col>.
    word_col : str, optional
        Column name that holds the token (default: "word").
    seg_col  : str, optional
        Column name for the segment index (default: "segment_idx").
    idx_col  : str, optional
        Column name for the word index inside a segment (default: "word_idx").

    Returns
    -------
    str
        The full transcript, with proper spacing around punctuation.
    """
    csv_path = Path(csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ------------------------------------------------------------------
    # 2.1  Read the CSV into a list of [(seg, idx, word), ...]
    # ------------------------------------------------------------------
    rows: List[tuple[int, int, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Ensure the required columns exist
        for col in (seg_col, idx_col, word_col):
            if col not in reader.fieldnames:
                raise KeyError(f"Column '{col}' missing in CSV header {reader.fieldnames}")

        for line in reader:
            seg = int(line[seg_col])
            idx = int(line[idx_col])
            word = line[word_col]
            rows.append((seg, idx, word))

    # ------------------------------------------------------------------
    # 2.2  Sort just in case the CSV got shuffled (keeps logical order)
    # ------------------------------------------------------------------
    rows.sort(key=lambda x: (x[0], x[1]))   # (segment, word_index)

    # ------------------------------------------------------------------
    # 2.3  Build the raw word list
    # ------------------------------------------------------------------
    raw_words = [w for _, _, w in rows]

    # ------------------------------------------------------------------
    # 2.4  Join with a space, then fix spacing around punctuation
    # ------------------------------------------------------------------
    #  a) simple space‑joined string
    txt = "".join(raw_words)

    #  b) remove the space that appears *before* punctuation marks
    # #     (covers , . ! ? ; : )
    # txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)

    # #  c) collapse any accidental double‑spaces that may have been created
    # txt = re.sub(r"\s{2,}", " ", txt).strip()

    # #  d) (optional) capitalise the first letter of each sentence
    # #     If you don’t want this, comment the two lines below.
    # txt = re.sub(r"(^|[.!?]\s+)([a-z])",
    #              lambda m: m.group(1) + m.group(2).upper(),
    #              txt)

    return txt

# execute the functions
def execute(video_path):
    BASE_PATH = Path.home() / "context" / Path(video_path.name)
    AUDIO_FOLDER = f"{BASE_PATH}/audio"
    AUDIO_PATH = f"{AUDIO_FOLDER}/transcript_16k.wav"
    CSV_PATH = f"{AUDIO_FOLDER}/transcript_16k_word_ts.csv"
    SRT_PATH = f"{AUDIO_FOLDER}/transcript_16k_word_ts.srt"
    demux_audio(video_path, AUDIO_FOLDER, AUDIO_PATH)
    transcribe_audio(AUDIO_PATH, CSV_PATH)
    csv_to_srt(CSV_PATH, SRT_PATH)
    print("✅ Transcription complete ------ \n" + csv_to_transcript(CSV_PATH))

if __name__ == "__main__":
    video_path = Path.home()/ "Downloads" / "demo.mp4"
    print(video_path)
    execute(video_path)
import sounddevice as sd
import numpy as np
import queue
import sherpa_onnx
from huggingface_hub import hf_hub_download
import re
from difflib import SequenceMatcher

SAMPLE_RATE = 16000
REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"

# ===== DOWNLOAD MODEL =====
def download_model():
    return {
        "encoder": hf_hub_download(REPO_ID, "encoder-epoch-35-avg-6.onnx"),
        "decoder": hf_hub_download(REPO_ID, "decoder-epoch-35-avg-6.onnx"),
        "joiner": hf_hub_download(REPO_ID, "joiner-epoch-35-avg-6.onnx"),
        "tokens": hf_hub_download(REPO_ID, "tokens.txt"),
    }

def create_recognizer(model_paths):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths["encoder"],
        decoder=model_paths["decoder"],
        joiner=model_paths["joiner"],
        tokens=model_paths["tokens"],
        sample_rate=16000,
        feature_dim=80,
    )

# ===== COMMAND LIST =====
COMMANDS = {
    "bat_den": ["bật đèn", "mở đèn", "bật đèn lên"],
    "tat_den": ["tắt đèn", "tắt đèn đi"],
    "bat_quat": ["bật quạt", "mở quạt"],
    "tat_quat": ["tắt quạt"]
}

# ===== NORMALIZE =====
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũ"
                  r"ưừứựửữỳýỵỷỹđ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===== FUZZY MATCH =====
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def match_command(text):
    best_cmd = None
    best_score = 0

    for cmd, phrases in COMMANDS.items():
        for p in phrases:
            score = similarity(text, p)
            if score > best_score:
                best_score = score
                best_cmd = cmd

    if best_score > 0.6:
        return best_cmd
    return None

# ===== AUDIO QUEUE =====
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata.copy())

# ===== MAIN =====
def main():
    print("📥 Loading model...")
    model_paths = download_model()
    recognizer = create_recognizer(model_paths)

    sd.default.device = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=8000,
    )

    print("🎤 Bắt đầu nói (Ctrl+C để dừng)")
    stream.start()

    buffer = []
    last_cmd = None

    while True:
        data = q.get()
        buffer.extend(data.flatten())

        if len(buffer) > SAMPLE_RATE * 3:
            samples = np.array(buffer[:SAMPLE_RATE*3], dtype=np.float32)
            buffer = buffer[SAMPLE_RATE*3:]

            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, samples)
            recognizer.decode_streams([s])

            text = s.result.text.strip()

            if text:
                text_norm = normalize_text(text)
                cmd = match_command(text_norm)

                if cmd and cmd != last_cmd:
                    print("🎯 COMMAND:", cmd)
                    last_cmd = cmd
                else:
                    print("📝", text)

if __name__ == "__main__":
    main()
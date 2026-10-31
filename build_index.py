"""Φτιάχνει το ευρετήριο για τη σημασιολογική αναζήτηση στις συζητήσεις.

1. Παίρνει τα θέματα με απομαγνητοφώνηση (μη κενό full_transcript_text).
2. Τα χωρίζει σε κομμάτια (chunks) ~200 λέξεων με επικάλυψη 40 λέξεων.
3. Υπολογίζει embeddings με το μοντέλο intfloat/multilingual-e5-small (CPU).
4. Αποθηκεύει στο data/:
   - search_chunks.parquet    : τα κομμάτια και τα στοιχεία τους
   - search_embeddings.npy    : ένας κανονικοποιημένος πίνακας embeddings ανά κομμάτι

Εκκίνηση: python build_index.py   (την πρώτη φορά κατεβαίνει και το μοντέλο, ~470 MB)
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
PARQUET_FILE = DATA_DIR / "open_council_20260603.parquet"
CHUNKS_FILE = DATA_DIR / "search_chunks.parquet"
EMBEDDINGS_FILE = DATA_DIR / "search_embeddings.npy"

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_WORDS = 200    # μέγεθος κομματιού σε λέξεις
OVERLAP_WORDS = 40   # πόσες λέξεις μοιράζονται δύο διαδοχικά κομμάτια
MIN_WORDS = 15       # κομμάτια με λιγότερες λέξεις (π.χ. σκέτο «Ναι.») δεν έχουν νόημα

COLUMNS = [
    "city_name", "meeting_id", "meeting_date", "meeting_name",
    "subject_name", "meeting_url", "full_transcript_text",
]

# Ετικέτα ομιλητή στην αρχή κάθε γραμμής, π.χ. «SPEAKER_3: », «New speaker segment: »
SPEAKER_LABEL = re.compile(r"(?m)^[^:\n]{1,40}:\s*")


def split_into_chunks(text):
    """Χωρίζει ένα κείμενο σε κομμάτια CHUNK_WORDS λέξεων με επικάλυψη OVERLAP_WORDS."""
    words = SPEAKER_LABEL.sub("", text).split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + CHUNK_WORDS]
        if len(chunk) >= MIN_WORDS:
            chunks.append(" ".join(chunk))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def main():
    df = pd.read_parquet(PARQUET_FILE, columns=COLUMNS)
    df = df[df["full_transcript_text"].fillna("").str.strip() != ""]
    print(f"Θέματα με απομαγνητοφώνηση: {len(df)}")

    # Ημερομηνία σε ώρα Ελλάδας (όπως και στο app.py)
    df["meeting_date"] = (
        pd.to_datetime(df["meeting_date"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Athens")
        .dt.tz_localize(None)
    )

    rows = []
    for subject in df.itertuples(index=False):
        for chunk in split_into_chunks(subject.full_transcript_text):
            rows.append({
                "city_name": subject.city_name,
                "meeting_id": subject.meeting_id,
                "meeting_date": subject.meeting_date,
                "meeting_name": subject.meeting_name,
                "subject_name": subject.subject_name,
                "meeting_url": subject.meeting_url,
                "text": chunk,
            })
    chunks = pd.DataFrame(rows)
    print(f"Κομμάτια: {len(chunks)}")

    # Το e5 θέλει το πρόθεμα «passage: » στα κείμενα και «query: » στις ερωτήσεις.
    # Βάζουμε μπροστά και τον τίτλο του θέματος, ώστε να βοηθά στην αναζήτηση.
    passages = ("passage: " + chunks["subject_name"].fillna("") + ". " + chunks["text"]).tolist()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    start = time.time()
    embeddings = model.encode(
        passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # ώστε το εσωτερικό γινόμενο να είναι η ομοιότητα συνημιτόνου
        convert_to_numpy=True,
    )
    print(f"Embeddings: {embeddings.shape} σε {(time.time() - start) / 60:.1f} λεπτά")

    chunks.to_parquet(CHUNKS_FILE, index=False)
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))
    print(f"Αποθηκεύτηκαν: {CHUNKS_FILE.name}, {EMBEDDINGS_FILE.name}")


if __name__ == "__main__":
    main()

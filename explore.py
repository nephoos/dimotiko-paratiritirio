"""Πρώτη εξερεύνηση του dataset glossAPI/OpenCouncil.

Τυπώνει βασικά στατιστικά στην οθόνη και τα αποθηκεύει στο summary.txt.
Κάθε γραμμή του dataset είναι ένα θέμα (subject) μιας συνεδρίασης δημοτικού συμβουλίου.
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_FILE = Path(__file__).parent / "summary.txt"

# Βρίσκουμε το αρχείο parquet μέσα στον φάκελο data/
parquet_files = sorted(DATA_DIR.glob("*.parquet"))
if not parquet_files:
    raise SystemExit("Δεν βρέθηκε αρχείο .parquet στον φάκελο data/")

df = pd.read_parquet(parquet_files[-1])

# Μετατρέπουμε την ημερομηνία συνεδρίασης σε πραγματική ημερομηνία
df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce", utc=True)

lines = []


def out(text=""):
    """Τυπώνει μια γραμμή και την κρατάει για το summary.txt."""
    print(text)
    lines.append(str(text))


out(f"Αρχείο: {parquet_files[-1].name}")
out(f"Συνολικές γραμμές (θέματα): {len(df):,}")
out()

# Δήμοι
cities = sorted(df["city_name"].dropna().unique())
out(f"Δήμοι ({len(cities)}):")
for city in cities:
    out(f"  - {city}")
out()

# Στατιστικά ανά δήμο
by_city = df.groupby("city_name")
stats = pd.DataFrame({
    "Συνεδριάσεις": by_city["meeting_id"].nunique(),
    "Θέματα": by_city.size(),
    "Θέματα με votes_for": by_city["votes_for"].apply(lambda s: s.notna().sum()),
    "Θέματα με votes_against": by_city["votes_against"].apply(lambda s: s.notna().sum()),
    # Το 0 συχνά σημαίνει «δεν καταγράφηκε ψηφοφορία», γι' αυτό μετράμε και τις τιμές > 0
    "votes_for > 0": by_city["votes_for"].apply(lambda s: (s > 0).sum()),
    "votes_against > 0": by_city["votes_against"].apply(lambda s: (s > 0).sum()),
    "Πρώτη ημερομηνία": by_city["meeting_date"].min().dt.strftime("%Y-%m-%d"),
    "Τελευταία ημερομηνία": by_city["meeting_date"].max().dt.strftime("%Y-%m-%d"),
}).sort_values("Θέματα", ascending=False)

out("Στατιστικά ανά δήμο:")
with pd.option_context("display.width", 200, "display.max_columns", None):
    out(stats.to_string())
out()

# Μέσο μήκος κειμένου απομαγνητοφώνησης
lengths = df["full_transcript_text"].fillna("").str.len()
out(f"Μέσο μήκος full_transcript_text: {lengths.mean():,.0f} χαρακτήρες")
out(f"Θέματα χωρίς απομαγνητοφώνηση: {(lengths == 0).sum():,}")
out()

# Κάλυψη απομαγνητοφωνήσεων ανά δήμο
df["has_transcript"] = df["full_transcript_text"].fillna("").str.strip() != ""


def count_speakers(turns_json):
    """Επιστρέφει τα διαφορετικά speaker_tag_id ενός θέματος.

    Τα speaker_name/speaker_id είναι κενά στο dataset, οπότε το speaker_tag_id
    είναι το μόνο αναγνωριστικό. Είναι ξεχωριστό για κάθε συνεδρίαση, άρα μετράμε
    «φωνές ανά συνεδρίαση» και όχι μοναδικά πρόσωπα.
    """
    if not turns_json:
        return set()
    return {turn["speaker_tag_id"] for turn in json.loads(turns_json) if turn.get("speaker_tag_id")}


df["speaker_tags"] = df["speaker_turns_json"].apply(count_speakers)

by_city = df.groupby("city_name")
coverage = pd.DataFrame({
    "Θέματα": by_city.size(),
    "Με απομαγνητοφώνηση": by_city["has_transcript"].sum(),
    "Ποσοστό %": (by_city["has_transcript"].mean() * 100).round(1),
    "Συνεδριάσεις": by_city["meeting_id"].nunique(),
    "Συνεδριάσεις με απομαγν.": df[df["has_transcript"]].groupby("city_name")["meeting_id"].nunique(),
    "Διαφορετικοί ομιλητές*": by_city["speaker_tags"].apply(lambda s: len(set().union(*s))),
}).fillna(0).astype({"Συνεδριάσεις με απομαγν.": int}).sort_values("Θέματα", ascending=False)

out("Κάλυψη απομαγνητοφωνήσεων ανά δήμο:")
with pd.option_context("display.width", 200, "display.max_columns", None):
    out(coverage.to_string())
out("* Μέτρηση με βάση το speaker_tag_id (τα ονόματα ομιλητών είναι κενά στο dataset).")
out("  Το ίδιο πρόσωπο σε δύο συνεδριάσεις μετράει δύο φορές.")

SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nΤα αποτελέσματα αποθηκεύτηκαν στο {SUMMARY_FILE.name}")

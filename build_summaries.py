"""Φτιάχνει αυτόματες συνόψεις συνεδριάσεων με το Gemini API.

Για κάθε συνεδρίαση ενός δήμου:
1. Μαζεύει με τη σειρά της ημερήσιας διάταξης τον τίτλο κάθε θέματος,
   το αποτέλεσμα ψηφοφορίας (αν καταγράφηκε) και την απομαγνητοφώνηση (αν υπάρχει).
2. Ζητάει από το Gemini σύνοψη στα ελληνικά σε 4-6 κουκκίδες.
3. Αποθηκεύει τις συνόψεις στο summaries/<δήμος>.json (μετά από κάθε συνεδρίαση).

Συνεδριάσεις που έχουν ήδη σύνοψη παραλείπονται, οπότε αν διακοπεί το ξανατρέχουμε.
Το κλειδί διαβάζεται από το αρχείο .env (GEMINI_API_KEY).

Εκκίνηση:
    python build_summaries.py                    # όλες οι συνεδριάσεις της Αθήνας
    python build_summaries.py --limit 2          # μόνο οι 2 πιο πρόσφατες
    python build_summaries.py --city chania      # άλλος δήμος (city_id ή όνομα)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors

DATA_DIR = Path(__file__).parent / "data"
PARQUET_FILE = DATA_DIR / "open_council_20260603.parquet"
SUMMARIES_DIR = Path(__file__).parent / "summaries"

MODEL_NAME = "gemini-3.5-flash"  # μοντέλο «Flash», διαθέσιμο στο δωρεάν επίπεδο
WAIT_BETWEEN_REQUESTS = 10       # δευτερόλεπτα (το δωρεάν επίπεδο επιτρέπει λίγα αιτήματα/λεπτό)
MAX_RETRIES = 6                  # επαναλήψεις όταν πάρουμε 429 (όριο) ή 503 (υπερφόρτωση)
FIRST_RETRY_WAIT = 30            # δευτερόλεπτα· διπλασιάζεται σε κάθε επανάληψη

COLUMNS = [
    "city_id", "city_name", "meeting_id", "meeting_name", "meeting_date",
    "subject_name", "agenda_item_index", "votes_for", "votes_against", "votes_abstain",
    "full_transcript_text",
]

# Το ακριβές κείμενο που στέλνουμε στο Gemini (καταγράφεται και στο PROMPTS.md)
PROMPT_TEMPLATE = """Είσαι βοηθός που συνοψίζει συνεδριάσεις δημοτικών συμβουλίων για πολίτες.

Παρακάτω είναι τα θέματα μιας συνεδρίασης, με τη σειρά της ημερήσιας διάταξης. Για κάθε θέμα δίνεται ο τίτλος, το αποτέλεσμα της ψηφοφορίας (αν καταγράφηκε) και η απομαγνητοφώνηση της συζήτησης (αν υπάρχει).

Γράψε σύνοψη της συνεδρίασης στα ελληνικά σε 4 έως 6 κουκκίδες (κάθε κουκκίδα ξεκινά με «- »). Κάλυψε:
- τα κύρια θέματα που συζητήθηκαν,
- τι αποφασίστηκε,
- πού υπήρξε διαφωνία (π.χ. ψήφοι κατά, αντιρρήσεις στη συζήτηση).

Κανόνες:
- Χρησιμοποίησε ΜΟΝΟ πληροφορίες που υπάρχουν στο κείμενο παρακάτω. Μην επινοείς ονόματα, αριθμούς, αποφάσεις ή διαφωνίες.
- Αν κάτι δεν προκύπτει από το κείμενο (π.χ. δεν καταγράφηκε ψηφοφορία), μην το υποθέτεις.
- Αν δεν υπάρχει απομαγνητοφώνηση για κανένα θέμα, η σύνοψη να βασίζεται μόνο στους τίτλους και η πρώτη κουκκίδα να αναφέρει ότι δεν υπάρχουν απομαγνητοφωνήσεις και ότι η σύνοψη βασίζεται μόνο στους τίτλους των θεμάτων.
- Γράψε μόνο τις κουκκίδες, χωρίς εισαγωγή ή επίλογο.

Συνεδρίαση: {meeting_name} ({meeting_date})
Υπάρχουν απομαγνητοφωνήσεις: {transcripts_note}

{agenda}"""


def vote_text(row):
    """Κείμενο αποτελέσματος ψηφοφορίας (όπως και στο app.py)."""
    votes = [0 if pd.isna(v) else int(v)
             for v in (row.votes_for, row.votes_against, row.votes_abstain)]
    if sum(votes) == 0:
        return "Δεν καταγράφηκε ψηφοφορία"
    return f"Υπέρ {votes[0]} · Κατά {votes[1]} · Λευκά {votes[2]}"


def build_prompt(subjects):
    """Φτιάχνει το prompt για μία συνεδρίαση από τα θέματά της (ήδη ταξινομημένα)."""
    first = subjects.iloc[0]
    parts = []
    for number, row in enumerate(subjects.itertuples(index=False), start=1):
        transcript = (row.full_transcript_text or "").strip()
        parts.append(
            f"### Θέμα {number}: {row.subject_name}\n"
            f"Ψηφοφορία: {vote_text(row)}\n"
            f"Απομαγνητοφώνηση: {transcript if transcript else '(δεν υπάρχει)'}"
        )
    has_transcripts = bool((subjects["full_transcript_text"].fillna("").str.strip() != "").any())
    prompt = PROMPT_TEMPLATE.format(
        meeting_name=first.meeting_name,
        meeting_date=f"{first.meeting_date:%d/%m/%Y}",
        transcripts_note="Ναι" if has_transcripts else "Όχι",
        agenda="\n\n".join(parts),
    )
    return prompt, has_transcripts


def ask_gemini(client, prompt):
    """Στέλνει το prompt. Σε σφάλμα ορίου (429) ή υπερφόρτωσης (503) περιμένει και ξαναδοκιμάζει."""
    wait = FIRST_RETRY_WAIT
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except errors.APIError as error:
            if error.code not in (429, 503) or attempt == MAX_RETRIES:
                raise
            # Το ημερήσιο όριο δεν ξεπερνιέται με αναμονή λεπτών: σταματάμε (οι συνόψεις έχουν αποθηκευτεί)
            if error.code == 429 and "PerDay" in str(error):
                raise SystemExit("Εξαντλήθηκε το ημερήσιο όριο αιτημάτων. Ξανατρέξτε το script αύριο.")
            print(f"  Σφάλμα {error.code}, νέα προσπάθεια σε {wait} δευτ.…")
            time.sleep(wait)
            wait *= 2


def load_summaries(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_summaries(path, summaries):
    SUMMARIES_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Συνόψεις συνεδριάσεων με το Gemini.")
    parser.add_argument("--city", default="athens", help="city_id ή όνομα δήμου (προεπιλογή: athens)")
    parser.add_argument("--limit", type=int, help="μόνο οι N πιο πρόσφατες συνεδριάσεις")
    args = parser.parse_args()

    load_dotenv()  # φορτώνει το GEMINI_API_KEY από το .env
    client = genai.Client()

    df = pd.read_parquet(PARQUET_FILE, columns=COLUMNS)
    df = df[(df["city_id"] == args.city) | (df["city_name"] == args.city)]
    if df.empty:
        raise SystemExit(f"Δεν βρέθηκε δήμος «{args.city}».")
    city_id = df["city_id"].iloc[0]

    # Ημερομηνία σε ώρα Ελλάδας (όπως και στο app.py)
    df["meeting_date"] = (
        pd.to_datetime(df["meeting_date"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Athens")
        .dt.tz_localize(None)
    )
    df["agenda_order"] = pd.to_numeric(df["agenda_item_index"], errors="coerce")

    # Συνεδριάσεις από τη νεότερη στην παλαιότερη
    meeting_ids = (
        df.groupby("meeting_id")["meeting_date"].first()
        .sort_values(ascending=False).index.tolist()
    )
    if args.limit:
        meeting_ids = meeting_ids[:args.limit]

    path = SUMMARIES_DIR / f"{city_id}.json"
    summaries = load_summaries(path)
    todo = [m for m in meeting_ids if m not in summaries]
    print(f"{city_id}: {len(meeting_ids)} συνεδριάσεις, {len(todo)} χωρίς σύνοψη")

    for i, meeting_id in enumerate(todo, start=1):
        subjects = df[df["meeting_id"] == meeting_id].sort_values("agenda_order", na_position="last")
        prompt, has_transcripts = build_prompt(subjects)
        first = subjects.iloc[0]
        print(f"[{i}/{len(todo)}] {meeting_id} – {first.meeting_name} ({len(subjects)} θέματα)")

        summaries[meeting_id] = {
            "meeting_name": first.meeting_name,
            "meeting_date": f"{first.meeting_date:%Y-%m-%d %H:%M}",
            "has_transcripts": has_transcripts,
            "model": MODEL_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "summary": ask_gemini(client, prompt),
        }
        save_summaries(path, summaries)  # αποθήκευση μετά από κάθε συνεδρίαση

        if i < len(todo):
            time.sleep(WAIT_BETWEEN_REQUESTS)

    print(f"Αποθηκεύτηκαν στο {path}")


if __name__ == "__main__":
    main()

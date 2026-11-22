"""Φτιάχνει το φύλλο αξιολόγησης evaluation/private/evaluation.xlsx.

Τρέχει την αναζήτηση της εφαρμογής για 10 ερωτήσεις σε όλους τους δήμους και
μαζεύει 5 συνόψεις της Αθήνας. Το xlsx περιέχει αποσπάσματα από το gated dataset,
γι' αυτό ο φάκελος private/ δεν ανεβαίνει στο GitHub (.gitignore).

Εκκίνηση (από τη ρίζα του project): python evaluation/build_sheet.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = Path(__file__).parent / "private" / "evaluation.xlsx"

SEARCH_MODEL = "intfloat/multilingual-e5-small"  # ίδιο με το app.py
TOP_K = 5

QUERIES = [
    "παιδική φτώχεια", "καθαριότητα", "ποδηλατόδρομος", "αδέσποτα ζώα",
    "ηλεκτροφωτισμός", "στέγαση", "προσβασιμότητα ΑμεΑ", "σχολικά κτίρια",
    "κυκλοφοριακό", "πάρκα και πράσινο",
]

# Συνεδριάσεις της Αθήνας με ποικιλία: μικρές και μεγάλες, κάποιες με ψήφους κατά
MEETINGS = [
    "may29_2026",    # 2 θέματα, με ψήφους κατά
    "mar16_2026",    # 11 θέματα
    "feb16_2026",    # 25 θέματα, 6 με ψήφους κατά
    "feb26_2026",    # 51 θέματα
    "may18_3_2026",  # 126 θέματα (η μεγαλύτερη)
]

HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
SCORE_FILL = PatternFill("solid", fgColor="FFF2CC")  # οι στήλες που συμπληρώνει ο αξιολογητής


def search(chunks, embeddings, model, query, top_k=TOP_K):
    """Ίδια λογική με το search() του app.py, σε όλους τους δήμους."""
    query_emb = model.encode(f"query: {query}", normalize_embeddings=True)
    scores = embeddings @ query_emb
    best = np.argsort(-scores)[:top_k]
    return chunks.iloc[best]


def vote_text(row):
    """Ίδιο κείμενο ψηφοφορίας με το app.py."""
    votes = [row["votes_for"], row["votes_against"], row["votes_abstain"]]
    votes = [0 if pd.isna(v) else int(v) for v in votes]
    if sum(votes) == 0:
        return "Δεν καταγράφηκε ψηφοφορία"
    return f"Υπέρ {votes[0]} · Κατά {votes[1]} · Λευκά {votes[2]}"


def write_sheet(ws, headers, rows, widths, score_columns):
    """Γράφει ένα φύλλο με επικεφαλίδες, αναδίπλωση κειμένου και φαρδιές στήλες."""
    ws.append(headers)
    for row in rows:
        ws.append(row)
    for col, width in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, col).column_letter].width = width
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if cell.row == 1:
                cell.font = Font(bold=True)
                cell.fill = HEADER_FILL
            elif headers[cell.column - 1] in score_columns:
                cell.fill = SCORE_FILL
    ws.freeze_panes = "A2"


def search_rows():
    from sentence_transformers import SentenceTransformer

    chunks = pd.read_parquet(DATA_DIR / "search_chunks.parquet")
    embeddings = np.load(DATA_DIR / "search_embeddings.npy")
    model = SentenceTransformer(SEARCH_MODEL, device="cpu")

    rows = []
    for query in QUERIES:
        results = search(chunks, embeddings, model, query)
        for position, r in enumerate(results.itertuples(), start=1):
            rows.append([
                query, position, r.city_name, f"{r.meeting_date:%d/%m/%Y}",
                r.subject_name, r.text, None,
            ])
    return rows


def summary_rows():
    summaries = json.loads((ROOT / "summaries" / "athens.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(
        DATA_DIR / "open_council_20260603.parquet",
        columns=["city_id", "meeting_id", "meeting_url", "subject_name",
                 "agenda_item_index", "votes_for", "votes_against", "votes_abstain"],
    )
    df = df[df["city_id"] == "athens"]

    rows = []
    for meeting_id in MEETINGS:
        summary = summaries[meeting_id]
        subjects = df[df["meeting_id"] == meeting_id].copy()
        subjects["order"] = pd.to_numeric(subjects["agenda_item_index"], errors="coerce")
        subjects = subjects.sort_values("order", na_position="last")
        agenda = "\n".join(
            f"{i}. {s['subject_name']} — {vote_text(s)}"
            for i, (_, s) in enumerate(subjects.iterrows(), start=1)
        )
        rows.append([
            summary["meeting_date"][:10], summary["meeting_name"],
            subjects["meeting_url"].iloc[0], summary["summary"], agenda,
            None, None, None, None,
        ])
    return rows


INSTRUCTIONS = [
    ["Πώς βαθμολογώ"],
    [""],
    ["Γενικά"],
    ["• Συμπληρώνω μόνο τις κίτρινες στήλες. Δεν αλλάζω τις άλλες στήλες ούτε τη σειρά των γραμμών."],
    ["• Αφήνω κενό ό,τι δεν μπορώ να κρίνω· το κενό δεν μετράει στους υπολογισμούς."],
    [""],
    ["Φύλλο «Αναζήτηση»"],
    ["• Για κάθε ερώτηση υπάρχουν τα 5 πρώτα αποτελέσματα της αναζήτησης σε όλους τους δήμους."],
    ["• Στη στήλη «Σχετικό;» γράφω 1 αν το απόσπασμα (ή ο τίτλος του θέματος) αφορά πραγματικά το θέμα της ερώτησης, αλλιώς 0."],
    ["• Παράδειγμα: για «ποδηλατόδρομος», ένα απόσπασμα για ποδηλατόδρομο στην παραλία = 1· ένα απόσπασμα για στάθμευση αυτοκινήτων χωρίς αναφορά σε ποδήλατα = 0."],
    ["• Κρίνω το νόημα, όχι αν εμφανίζεται ακριβώς η λέξη."],
    ["• Από αυτά υπολογίζεται το Precision@5: πόσα από τα 5 αποτελέσματα ήταν σχετικά (π.χ. 3 στα 5 = 0,6)."],
    [""],
    ["Φύλλο «Συνόψεις»"],
    ["• Διαβάζω τη σύνοψη και τη συγκρίνω με τους τίτλους των θεμάτων και τις ψηφοφορίες (και, αν χρειαστεί, με τη συνεδρίαση στο OpenCouncil από το link)."],
    ["• Ακρίβεια (1-5): είναι σωστά όσα γράφει; 5 = όλα σωστά, 3 = κάποια λάθη, 1 = πολλά ή σοβαρά λάθη."],
    ["• Πληρότητα (1-5): καλύπτει τα σημαντικά θέματα και τις διαφωνίες; 5 = όλα τα σημαντικά, 3 = λείπουν αρκετά, 1 = λείπουν τα περισσότερα."],
    ["• Επινοήσεις; (ναι/όχι): «ναι» αν η σύνοψη αναφέρει κάτι που δεν υπάρχει στα δεδομένα (π.χ. όνομα, αριθμό, απόφαση ή ψηφοφορία που δεν έγινε)."],
    ["• Σχόλια: σύντομα, με δικά μου λόγια (π.χ. «λάθος αριθμός ψήφων στο θέμα 2»). Όχι αντιγραφή μεγάλων αποσπασμάτων."],
    [""],
    ["Μετά"],
    ["• Αποθηκεύω το αρχείο (ίδιο όνομα, ίδια θέση) και τρέχω: python evaluation/compute_results.py"],
    ["• Δημιουργείται το evaluation/EVALUATION.md με τα αποτελέσματα."],
]


def main():
    wb = Workbook()

    ws = wb.active
    ws.title = "Αναζήτηση"
    write_sheet(
        ws,
        ["Ερώτηση", "Θέση (1-5)", "Δήμος", "Ημερομηνία", "Τίτλος θέματος", "Απόσπασμα",
         "Σχετικό; (1=ναι, 0=όχι)"],
        search_rows(),
        widths=[20, 10, 16, 13, 45, 100, 14],
        score_columns={"Σχετικό; (1=ναι, 0=όχι)"},
    )

    ws = wb.create_sheet("Συνόψεις")
    write_sheet(
        ws,
        ["Ημερομηνία", "Όνομα συνεδρίασης", "Link στο OpenCouncil", "Σύνοψη",
         "Θέματα και ψηφοφορίες", "Ακρίβεια (1-5)", "Πληρότητα (1-5)",
         "Επινοήσεις; (ναι/όχι)", "Σχόλια"],
        summary_rows(),
        widths=[12, 22, 26, 80, 80, 11, 11, 13, 40],
        score_columns={"Ακρίβεια (1-5)", "Πληρότητα (1-5)", "Επινοήσεις; (ναι/όχι)", "Σχόλια"},
    )
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].hyperlink = row[0].value
        row[0].font = Font(color="0563C1", underline="single")

    ws = wb.create_sheet("Οδηγίες")
    for line in INSTRUCTIONS:
        ws.append(line)
    ws.column_dimensions["A"].width = 120
    for row in ws.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
        if row[0].value in {"Πώς βαθμολογώ", "Γενικά", "Φύλλο «Αναζήτηση»", "Φύλλο «Συνόψεις»", "Μετά"}:
            row[0].font = Font(bold=True, size=14 if row[0].row == 1 else 12)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(f"Αποθηκεύτηκε: {OUTPUT}")


if __name__ == "__main__":
    main()

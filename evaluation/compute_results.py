"""Υπολογίζει τα αποτελέσματα από το συμπληρωμένο evaluation/private/evaluation.xlsx.

Γράφει το evaluation/EVALUATION.md μόνο με νούμερα, τίτλους ερωτήσεων/συνεδριάσεων
και τα σχόλια του αξιολογητή. Δεν αντιγράφει αποσπάσματα ή συνόψεις (gated dataset).

Εκκίνηση (από τη ρίζα του project): python evaluation/compute_results.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
INPUT = HERE / "private" / "evaluation.xlsx"
OUTPUT = HERE / "EVALUATION.md"

RELEVANT = "Σχετικό; (1=ναι, 0=όχι)"
ACCURACY = "Ακρίβεια (1-5)"
COMPLETENESS = "Πληρότητα (1-5)"
INVENTED = "Επινοήσεις; (ναι/όχι)"
COMMENTS = "Σχόλια"
MAX_COMMENT = 300  # τα σχόλια κόβονται, για να μη μπαίνουν μεγάλα κομμάτια κειμένου


def fmt(value, digits=2):
    return "–" if pd.isna(value) else f"{value:.{digits}f}".replace(".", ",")


def search_section(search):
    search[RELEVANT] = pd.to_numeric(search[RELEVANT], errors="coerce")
    lines = [
        "## Αναζήτηση (Precision@5)",
        "",
        "Precision@5 = πόσα από τα 5 πρώτα αποτελέσματα κρίθηκαν σχετικά. "
        "Τα κενά κελιά δεν μετρούν.",
        "",
        "| Ερώτηση | Σχετικά | Βαθμολογημένα | Precision@5 |",
        "|---|---|---|---|",
    ]
    precisions = []
    for query, group in search.groupby("Ερώτηση", sort=False):
        judged = group[RELEVANT].dropna()
        precision = judged.mean() if len(judged) else float("nan")
        if len(judged):
            precisions.append(precision)
        lines.append(f"| {query} | {int(judged.sum())} | {len(judged)} | {fmt(precision)} |")
    mean = sum(precisions) / len(precisions) if precisions else float("nan")
    lines += ["", f"**Μέσο Precision@5: {fmt(mean)}** (σε {len(precisions)} ερωτήσεις)", ""]
    return lines


def summaries_section(summaries):
    for col in (ACCURACY, COMPLETENESS):
        summaries[col] = pd.to_numeric(summaries[col], errors="coerce")
    invented = summaries[INVENTED].fillna("").astype(str).str.strip().str.lower()
    n_yes = int(invented.isin({"ναι", "ναί", "yes", "1"}).sum())
    n_judged = int((invented != "").sum())

    lines = [
        "## Συνόψεις",
        "",
        "| Ημερομηνία | Συνεδρίαση | Ακρίβεια | Πληρότητα | Επινοήσεις |",
        "|---|---|---|---|---|",
    ]
    for (_, row), flag in zip(summaries.iterrows(), invented):
        lines.append(
            f"| {row['Ημερομηνία']} | {row['Όνομα συνεδρίασης']} | {fmt(row[ACCURACY], 0)} | "
            f"{fmt(row[COMPLETENESS], 0)} | {flag or '–'} |"
        )
    lines += [
        "",
        f"- **Μέση ακρίβεια:** {fmt(summaries[ACCURACY].mean())} / 5 "
        f"(σε {summaries[ACCURACY].notna().sum()} συνόψεις)",
        f"- **Μέση πληρότητα:** {fmt(summaries[COMPLETENESS].mean())} / 5 "
        f"(σε {summaries[COMPLETENESS].notna().sum()} συνόψεις)",
        f"- **Συνόψεις με επινοήσεις:** {n_yes} από {n_judged}",
        "",
        "### Σχόλια",
        "",
    ]
    comments = [
        (row["Ημερομηνία"], str(row[COMMENTS]).strip())
        for _, row in summaries.iterrows()
        if pd.notna(row[COMMENTS]) and str(row[COMMENTS]).strip()
    ]
    for date, comment in comments:
        if len(comment) > MAX_COMMENT:
            comment = comment[:MAX_COMMENT].rstrip() + "…"
        lines.append(f"- **{date}:** {comment}")
    if not comments:
        lines.append("Δεν υπάρχουν σχόλια.")
    lines.append("")
    return lines


def main():
    search = pd.read_excel(INPUT, sheet_name="Αναζήτηση")
    summaries = pd.read_excel(INPUT, sheet_name="Συνόψεις")
    # Μόνο οι στήλες που χρειάζονται, για να μη διαρρεύσουν κείμενα στο αποτέλεσμα
    summaries = summaries[["Ημερομηνία", "Όνομα συνεδρίασης",
                           ACCURACY, COMPLETENESS, INVENTED, COMMENTS]]

    lines = [
        "# Αξιολόγηση αναζήτησης και συνόψεων",
        "",
        "Χειροκίνητη αξιολόγηση σε δείγμα: 10 ερωτήσεις αναζήτησης (5 πρώτα αποτελέσματα "
        "η καθεμία, σε όλους τους δήμους) και 5 αυτόματες συνόψεις συνεδριάσεων της Αθήνας. "
        "Το φύλλο με τα αποσπάσματα δεν δημοσιεύεται, γιατί προέρχεται από gated dataset.",
        "",
    ]
    lines += search_section(search)
    lines += summaries_section(summaries)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Αποθηκεύτηκε: {OUTPUT}")


if __name__ == "__main__":
    main()

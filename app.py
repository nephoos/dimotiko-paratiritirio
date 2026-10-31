"""Δημοτικό Παρατηρητήριο – dashboard με Streamlit.

Δεδομένα: dataset glossAPI/OpenCouncil από το Hugging Face.
Κάθε γραμμή του dataset είναι ένα θέμα (subject) μιας συνεδρίασης δημοτικού συμβουλίου.

Εκκίνηση: streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
HF_REPO_ID = "glossAPI/OpenCouncil"
HF_FILENAME = "open_council_20260603.parquet"

# Οι στήλες που χρειάζεται η εφαρμογή (το αρχείο έχει πολλές περισσότερες)
COLUMNS = [
    "city_id", "city_name", "meeting_id", "meeting_name", "meeting_date",
    "subject_id", "subject_name", "agenda_item_index", "topic_name",
    "decision_pdf_url", "votes_for", "votes_against", "votes_abstain",
    "meeting_url", "subject_url", "full_transcript_text",
]

st.set_page_config(page_title="Δημοτικό Παρατηρητήριο", page_icon="🏛️", layout="wide")


def get_hf_token():
    """Βρίσκει token για το Hugging Face (το dataset είναι gated).

    Τοπικά αρκεί το `hf login`. Online το βάζουμε ως HF_TOKEN στα secrets.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None


def get_data_file():
    """Επιστρέφει το αρχείο parquet. Αν δεν υπάρχει στο data/, το κατεβάζει."""
    local_file = DATA_DIR / HF_FILENAME
    if local_file.exists():
        return local_file

    from huggingface_hub import hf_hub_download

    with st.spinner("Κατέβασμα δεδομένων από το Hugging Face…"):
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir=DATA_DIR,
            token=get_hf_token(),
        )
    return Path(path)


@st.cache_data(show_spinner="Φόρτωση δεδομένων…")
def load_data():
    """Φορτώνει και καθαρίζει τα δεδομένα. Το αποτέλεσμα κρατιέται στην cache."""
    df = pd.read_parquet(get_data_file(), columns=COLUMNS)

    # Ημερομηνία σε ώρα Ελλάδας
    df["meeting_date"] = (
        pd.to_datetime(df["meeting_date"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Athens")
        .dt.tz_localize(None)
    )

    # Το meeting_id επαναλαμβάνεται σε διαφορετικούς δήμους (π.χ. may29_2026),
    # γι' αυτό φτιάχνουμε μοναδικό κλειδί συνεδρίασης μαζί με τον δήμο.
    df["meeting_key"] = df["city_id"] + "/" + df["meeting_id"]

    # Αν το θέμα έχει απομαγνητοφώνηση (κρατάμε μόνο τη σημαία, όχι το κείμενο)
    df["has_transcript"] = df["full_transcript_text"].fillna("").str.strip() != ""
    df = df.drop(columns=["full_transcript_text"])

    # Όταν υπέρ, κατά και λευκά είναι όλα 0, δεν καταγράφηκε ψηφοφορία
    votes = df[["votes_for", "votes_against", "votes_abstain"]].fillna(0)
    df["has_vote"] = votes.sum(axis=1) > 0

    # Σειρά στην ημερήσια διάταξη (κενό = άγνωστη, πάει στο τέλος)
    df["agenda_order"] = pd.to_numeric(df["agenda_item_index"], errors="coerce")

    df["topic_name"] = df["topic_name"].fillna("").str.strip().replace("", "Χωρίς θεματική")
    return df


def vote_text(row):
    """Κείμενο αποτελέσματος ψηφοφορίας για ένα θέμα."""
    if not row["has_vote"]:
        return "Δεν καταγράφηκε ψηφοφορία"
    return (
        f"Υπέρ {int(row['votes_for'])} · Κατά {int(row['votes_against'])} "
        f"· Λευκά {int(row['votes_abstain'])}"
    )


def percent(part, total):
    return f"{part / total * 100:.1f}%" if total else "–"


# ---------------------------------------------------------------- Πλαϊνή μπάρα

df = load_data()

meetings_per_city = df.groupby("city_name")["meeting_key"].nunique().sort_values(ascending=False)

st.sidebar.title("🏛️ Δημοτικό Παρατηρητήριο")
city = st.sidebar.selectbox(
    "Δήμος",
    options=meetings_per_city.index.tolist(),
    format_func=lambda name: f"{name} ({meetings_per_city[name]} συνεδριάσεις)",
)
page = st.sidebar.radio("Σελίδα", ["Επισκόπηση δήμου", "Συνεδριάσεις"])
st.sidebar.caption("Πηγή δεδομένων: OpenCouncil (glossAPI) στο Hugging Face.")

city_df = df[df["city_name"] == city]

# ---------------------------------------------------------- Επισκόπηση δήμου

if page == "Επισκόπηση δήμου":
    st.title(f"Επισκόπηση: {city}")

    n_meetings = city_df["meeting_key"].nunique()
    n_subjects = len(city_df)
    n_transcript = int(city_df["has_transcript"].sum())
    n_vote = int(city_df["has_vote"].sum())

    first, last = city_df["meeting_date"].min(), city_df["meeting_date"].max()
    st.caption(f"Περίοδος: {first:%d/%m/%Y} – {last:%d/%m/%Y}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Συνεδριάσεις", n_meetings)
    c2.metric("Θέματα", n_subjects)
    c3.metric("Με απομαγνητοφώνηση", percent(n_transcript, n_subjects),
              help=f"{n_transcript} από {n_subjects} θέματα")
    c4.metric("Με καταγεγραμμένη ψηφοφορία", percent(n_vote, n_subjects),
              help=f"{n_vote} από {n_subjects} θέματα")

    left, right = st.columns(2)

    with left:
        st.subheader("Συνεδριάσεις ανά μήνα")
        meetings = city_df.drop_duplicates("meeting_key")
        months = meetings["meeting_date"].dt.to_period("M")
        # Δείχνουμε και τους μήνες χωρίς συνεδρίαση (με 0)
        all_months = pd.period_range(months.min(), months.max(), freq="M")
        per_month = months.value_counts().reindex(all_months, fill_value=0)
        month_df = pd.DataFrame({
            "Μήνας": all_months.to_timestamp(),
            "Συνεδριάσεις": per_month.values,
        })
        fig = px.bar(month_df, x="Μήνας", y="Συνεδριάσεις")
        fig.update_xaxes(dtick="M1", tickformat="%m/%Y", title=None)
        fig.update_yaxes(title="Συνεδριάσεις")
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Οι 10 πιο συχνές θεματικές")
        topics = (
            city_df["topic_name"].value_counts().head(10)
            .rename_axis("Θεματική").reset_index(name="Θέματα")
        )
        fig = px.bar(topics, x="Θέματα", y="Θεματική", orientation="h")
        fig.update_yaxes(categoryorder="total ascending", title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Ψηφοφορίες")
    voted = city_df[city_df["has_vote"]]
    if voted.empty:
        st.info("Δεν υπάρχουν καταγεγραμμένες ψηφοφορίες για αυτόν τον δήμο.")
    else:
        st.caption(
            f"Μετράνε μόνο τα {len(voted)} θέματα με καταγεγραμμένη ψηφοφορία. "
            "Θέματα όπου υπέρ, κατά και λευκά είναι όλα 0 δεν συμπεριλαμβάνονται."
        )
        totals = pd.DataFrame({
            "Ψήφος": ["Υπέρ", "Κατά", "Λευκά"],
            "Σύνολο ψήφων": [
                int(voted["votes_for"].sum()),
                int(voted["votes_against"].sum()),
                int(voted["votes_abstain"].sum()),
            ],
        })
        fig = px.bar(totals, x="Ψήφος", y="Σύνολο ψήφων", color="Ψήφος",
                     color_discrete_map={"Υπέρ": "#2e9e5b", "Κατά": "#d64545", "Λευκά": "#9a9a9a"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- Συνεδριάσεις

else:
    st.title(f"Συνεδριάσεις: {city}")

    meetings = (
        city_df.groupby("meeting_key")
        .agg(
            meeting_name=("meeting_name", "first"),
            meeting_date=("meeting_date", "first"),
            meeting_url=("meeting_url", "first"),
            subjects=("subject_id", "size"),
        )
        .sort_values("meeting_date", ascending=False)
    )

    meeting_key = st.selectbox(
        "Επιλέξτε συνεδρίαση (νεότερη πρώτα)",
        options=meetings.index.tolist(),
        format_func=lambda key: (
            f"{meetings.at[key, 'meeting_date']:%d/%m/%Y} – "
            f"{meetings.at[key, 'meeting_name']} ({meetings.at[key, 'subjects']} θέματα)"
        ),
    )
    meeting = meetings.loc[meeting_key]

    st.subheader(meeting["meeting_name"])
    st.caption(f"Ημερομηνία: {meeting['meeting_date']:%d/%m/%Y %H:%M}")
    if meeting["meeting_url"]:
        st.markdown(f"[Η συνεδρίαση στο OpenCouncil]({meeting['meeting_url']})")

    subjects = city_df[city_df["meeting_key"] == meeting_key].sort_values(
        "agenda_order", na_position="last"
    )

    table = pd.DataFrame({
        "Α/Α": subjects["agenda_order"].astype("Int64"),
        "Τίτλος": subjects["subject_name"],
        "Θεματική": subjects["topic_name"],
        "Ψηφοφορία": subjects.apply(vote_text, axis=1),
        "Απομαγνητοφώνηση": subjects["has_transcript"].map({True: "✅ Ναι", False: "— Όχι"}),
        "Απόφαση (PDF)": subjects["decision_pdf_url"].replace("", None),
    })

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Α/Α": st.column_config.NumberColumn(width="small"),
            "Τίτλος": st.column_config.TextColumn(width="large"),
            "Απόφαση (PDF)": st.column_config.LinkColumn(display_text="Άνοιγμα PDF"),
        },
    )
    st.caption(
        "Α/Α: σειρά στην ημερήσια διάταξη (κενό όταν δεν είναι γνωστή). "
        "«Δεν καταγράφηκε ψηφοφορία» σημαίνει ότι υπέρ, κατά και λευκά ήταν όλα 0 στα δεδομένα."
    )

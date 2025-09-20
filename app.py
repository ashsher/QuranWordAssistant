from flask import Flask, render_template, request
import pandas as pd
import re
from markupsafe import Markup
import csv

# Machine learning / embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity as torch_cosine

# Local imports
from topics import TOPICS


# === CONSTANTS ===
SURAH_NAMES = {
    1: "Al-Fatihah", 2: "Al-Baqarah", 3: "Al-Imran", 4: "An-Nisa", 5: "Al-Maidah",
    6: "Al-Anam", 7: "Al-Araf", 8: "Al-Anfal", 9: "At-Tawbah", 10: "Yunus",
    11: "Hud", 12: "Yusuf", 13: "Ar-Rad", 14: "Ibrahim", 15: "Al-Hijr",
    16: "An-Nahl", 17: "Al-Isra", 18: "Al-Kahf", 19: "Maryam", 20: "Ta-Ha",
    21: "Al-Anbiya", 22: "Al-Hajj", 23: "Al-Muminun", 24: "An-Nur", 25: "Al-Furqan",
    26: "Ash-Shuara", 27: "An-Naml", 28: "Al-Qasas", 29: "Al-Ankabut", 30: "Ar-Rum",
    31: "Luqman", 32: "As-Sajdah", 33: "Al-Ahzab", 34: "Saba", 35: "Fatir",
    36: "Ya-Sin", 37: "As-Saffat", 38: "Sad", 39: "Az-Zumar", 40: "Ghafir",
    41: "Fussilat", 42: "Ash-Shura", 43: "Az-Zukhruf", 44: "Ad-Dukhan", 45: "Al-Jathiyah",
    46: "Al-Ahqaf", 47: "Muhammad", 48: "Al-Fath", 49: "Al-Hujurat", 50: "Qaf",
    51: "Adh-Dhariyat", 52: "At-Tur", 53: "An-Najm", 54: "Al-Qamar", 55: "Ar-Rahman",
    56: "Al-Waqiah", 57: "Al-Hadid", 58: "Al-Mujadilah", 59: "Al-Hashr", 60: "Al-Mumtahanah",
    61: "As-Saff", 62: "Al-Jumuah", 63: "Al-Munafiqun", 64: "At-Taghabun", 65: "At-Talaq",
    66: "At-Tahrim", 67: "Al-Mulk", 68: "Al-Qalam", 69: "Al-Haqqah", 70: "Al-Maarij",
    71: "Nuh", 72: "Al-Jinn", 73: "Al-Muzzammil", 74: "Al-Muddaththir", 75: "Al-Qiyamah",
    76: "Al-Insan", 77: "Al-Mursalat", 78: "An-Naba", 79: "An-Naziat", 80: "Abasa",
    81: "At-Takwir", 82: "Al-Infitar", 83: "Al-Mutaffifin", 84: "Al-Inshiqaq", 85: "Al-Buruj",
    86: "At-Tariq", 87: "Al-Ala", 88: "Al-Ghashiyah", 89: "Al-Fajr", 90: "Al-Balad",
    91: "Ash-Shams", 92: "Al-Lail", 93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin",
    96: "Al-Alaq", 97: "Al-Qadr", 98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat",
    101: "Al-Qariah", 102: "At-Takathurm", 103: "Al-Asr", 104: "Al-Humazah", 105: "Al-Fil",
    106: "Quraish", 107: "Al-Maun", 108: "Al-Kawthar", 109: "Al-Kafirun", 110: "An-Nasr",
    111: "Al-Masad", 112: "Al-Ikhlas", 113: "Al-Falaq", 114: "An-Nas"
}

PROPHETS = [
    "Adam", "Elisha", "Job", "David", "Dhul-Kifl", "Aaron", "Hud", "Abraham",
    "Enoch", "Elias", "Jesus", "Isaac", "Ismael", "Lot", "Moses", "Noah",
    "Salih", "Shu'ayb", "Solomon", "Ezra", "Jacob", "John", "Johan", "Joseph",
    "Zachariya", "Muhammad"
]


# === HIGHLIGHT HELPERS ===
def highlight_text(text, query):
    """Highlight a search query in text (case-insensitive)."""
    if not query:
        return text
    regex = re.compile(re.escape(query), re.IGNORECASE)
    return Markup(regex.sub(lambda m: f"<span class='highlight'>{m.group(0)}</span>", text))


def highlight_keywords(text, keywords):
    """Highlight a list of keywords in text."""
    for kw in keywords:
        regex = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        text = regex.sub(r'<span class="highlight">\1</span>', text)
    return Markup(text)


# === APP & DATA LOADING ===
app = Flask(__name__)

# Load Quran dataset
quran = []
with open("static/quran.csv", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Text"].strip():
            quran.append({
                "Surah": int(row["Surah"]),
                "Ayah": int(row["Ayah"]),
                "Text": row["Text"].strip()
            })
quran['Surah'] = quran['Surah'].astype(int)
quran['Ayah'] = quran['Ayah'].astype(int)
quran['Text'] = quran['Text'].astype(str)

# Models
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(quran['Text'].str.lower())

model = SentenceTransformer("all-MiniLM-L6-v2")
ayah_texts = quran["Text"].tolist()
ayah_embeddings = model.encode(ayah_texts, convert_to_tensor=True)


# === SEARCH ===
def search_quran(query, top_n=5, threshold=0.2):
    """Search ayahs using TF-IDF + cosine similarity."""
    if not query:
        return []

    query_vec = vectorizer.transform([query.lower()])
    similarities = sklearn_cosine(query_vec, X).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = []

    for idx in top_indices:
        score = similarities[idx]
        if score >= threshold:
            results.append({
                "surah": int(quran.loc[idx, "Surah"]),
                "surah_name": SURAH_NAMES.get(int(quran.loc[idx, "Surah"]), ""),
                "ayah": int(quran.loc[idx, "Ayah"]),
                "text": highlight_text(quran.loc[idx, "Text"], query),
                "score": float(score)
            })

    # sort results logically
    return sorted(results, key=lambda x: (x["surah"], x["ayah"])) or [{"text": "Nothing found"}]


# === TOPICS ===
def get_topic_ayahs(topic, top_n=10):
    """Retrieve top ayahs for a given topic using embeddings."""
    topic_emb = model.encode([topic], convert_to_tensor=True)
    sims = torch_cosine(topic_emb, ayah_embeddings).squeeze()

    top_indices = torch.topk(sims, top_n).indices.tolist()

    results = []
    for idx in top_indices:
        results.append({
            "surah": int(quran.loc[idx, "Surah"]),
            "surah_name": SURAH_NAMES.get(int(quran.loc[idx, "Surah"]), ""),
            "ayah": int(quran.loc[idx, "Ayah"]),
            "text": highlight_keywords(quran.loc[idx, "Text"], [topic]),
            "score": float(sims[idx])
        })

    return results


# === ROUTES ===
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["GET", "POST"])
def search():
    results, query, top_n = [], "", 5
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        top_n = int(request.form.get("top_n", 5))
        results = search_quran(query, top_n=top_n)
    return render_template("search.html", results=results, query=query, top_n=top_n)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/read")
def read():
    """Read the Quran surah by surah."""
    sorted_quran = quran.sort_values(by=["Surah", "Ayah"])
    grouped_quran = sorted_quran.groupby("Surah")

    surahs = [
        {
            "surah_num": int(num),
            "surah_name": SURAH_NAMES.get(int(num), f"Surah {num}"),
            "verses": verses.to_dict(orient="records")
        }
        for num, verses in grouped_quran
    ]

    surahs = sorted(surahs, key=lambda s: s["surah_num"])
    return render_template("read.html", surahs=surahs)


@app.route("/topics", methods=["GET", "POST"])
def topics():
    selected_topic, results = None, None
    if request.method == "POST":
        selected_topic = request.form.get("topic")
        if selected_topic:
            results = get_topic_ayahs(selected_topic, top_n=10)

    return render_template(
        "topics.html",
        topics=TOPICS,
        selected_topic=selected_topic,
        results=results
    )


# === MAIN ===
if __name__ == "__main__":
    app.run(debug=True)

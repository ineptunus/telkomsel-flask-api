from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════

model_sentiment = load_model("model_sentiment_cnn_lstm_fasttext.keras")
model_hate      = load_model("model_hate_speech_cnn_lstm_fasttext.keras")
model_sarcasm   = load_model("model_sarcasm_cnn_lstm_fasttext.keras")

with open("model_sentiment_cnn_lstm_fasttext_tokenizer.pkl", "rb") as f:
    tokenizer_sentiment = pickle.load(f)

with open("model_hate_speech_cnn_lstm_fasttext_tokenizer.pkl", "rb") as f:
    tokenizer_hate = pickle.load(f)

with open("model_sarcasm_cnn_lstm_fasttext_tokenizer.pkl", "rb") as f:
    tokenizer_sarcasm = pickle.load(f)

print("✅ Semua model berhasil di-load")

# ══════════════════════════════════════════════════════════════
# BLACKLIST KATA KASAR
# ══════════════════════════════════════════════════════════════

KATA_KASAR = {
    "kontol", "anjing", "anjir", "anjrot", "bangsat",
    "goblok", "tolol", "bodoh", "babi", "tai", "tahi",
    "kampret", "bajingan", "brengsek", "sialan", "keparat",
    "monyet", "asu", "jancok", "dancok", "jancuk", "dancuk",
    "memek", "ngentot", "ngewe", "pecun", "pelacur", "sundal",
    "bedebah", "setan", "iblis", "kurang ajar", "tengik",
    "sampah", "kimak", "pantek", "pukimak", "celaka",
}

def cek_kata_kasar(text_cleaned):
    words = set(text_cleaned.lower().split())
    return bool(words & KATA_KASAR)

# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

MAX_LEN = 100

NORMALISASI = {
    # Telkomsel
    "tsel"        : "telkomsel",
    "tlkms"       : "telkomsel",
    # Kata alay umum
    "ilang"       : "hilang",
    "lemot"       : "lambat",
    "ga"          : "tidak",
    "gak"         : "tidak",
    "gk"          : "tidak",
    "nggak"       : "tidak",
    "ngga"        : "tidak",
    "udah"        : "sudah",
    "udh"         : "sudah",
    "bgt"         : "banget",
    "aja"         : "saja",
    "aj"          : "saja",
    "yg"          : "yang",
    "dgn"         : "dengan",
    "krn"         : "karena",
    "karna"       : "karena",
    "tp"          : "tapi",
    "tpi"         : "tapi",
    "emang"       : "memang",
    "emg"         : "memang",
    "kalo"        : "kalau",
    "klo"         : "kalau",
    "gimana"      : "bagaimana",
    "gw"          : "saya",
    "gue"         : "saya",
    "lo"          : "kamu",
    "lu"          : "kamu",
    "elu"         : "kamu",
    "msh"         : "masih",
    "lg"          : "lagi",
    "jg"          : "juga",
    "sm"          : "sama",
    "blm"         : "belum",
    "sdh"         : "sudah",
    "pake"        : "pakai",
    "bs"          : "bisa",
    "knp"         : "kenapa",
    "hrs"         : "harus",
    # Kata kasar — normalisasi ke bentuk standar
    "asu"         : "anjing",
    "anjg"        : "anjing",
    "anjir"       : "anjing",
    "anjrot"      : "anjing",
    "anying"      : "anjing",
    "kntl"        : "kontol",
    "kontoll"     : "kontol",
    "bngst"       : "bangsat",
    "gblok"       : "goblok",
    "gblk"        : "goblok",
    "goblk"       : "goblok",
    "tll"         : "tolol",
    "bodo"        : "bodoh",
    "kamprett"    : "kampret",
    "bajingann"   : "bajingan",
    # Sarcasm keywords
    "mantul"      : "mantap betul",
    "recommended" : "direkomendasikan",
    "provider"    : "penyedia",
    "delay"       : "lambat",
    "noob"        : "tidak kompeten",
}

# Pola regex kata kasar dengan variasi penulisan
POLA_KASAR = [
    (r'k[\*@o0ck]nt[o0]l+',   'kontol'),
    (r'b[4a]ngs[4a]t+',       'bangsat'),
    (r'g[o0]bl[o0]k+',        'goblok'),
    (r'[a4]nj[i1]ng+',        'anjing'),
    (r'[a4]nj[i1]r+',         'anjing'),
    (r'[a4]nj[ro][o0]t+',     'anjing'),
    (r't[o0]l[o0]l+',         'tolol'),
    (r'b[o0]d[o0]h+',         'bodoh'),
    (r's[i1][a4]l[a4]n+',     'sialan'),
    (r'b[a4]b[i1]+',          'babi'),
    (r't[a4][i1]+',           'tai'),
    (r'j[a4]nc[o0]k+',        'jancok'),
    (r'j[a4]nc[u]k+',         'jancuk'),
    (r'k[e3]p[a4]r[a4]t+',    'keparat'),
    (r'm[o0]ny[e3]t+',        'monyet'),
    (r'[a4]su+',               'asu'),
]

def clean_text(text):
    text = str(text).lower()

    # 1. Normalisasi karakter berulang (kontollll → kontoll)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 2. Normalisasi kata kasar dengan pola regex
    #    (sebelum hapus karakter spesial)
    for pola, ganti in POLA_KASAR:
        text = re.sub(pola, ganti, text)

    # 3. Hapus URL, mention, hashtag
    text = re.sub(r'http\S+|www\S+',  ' ', text)
    text = re.sub(r'@\w+',            ' ', text)
    text = re.sub(r'#(\w+)',          r'\1', text)

    # 4. Hapus placeholder scraping
    text = re.sub(r'\b(user|pengguna)\b', ' ', text)

    # 5. Hapus emoji & non-ASCII
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    text = re.sub(r'\\x[0-9a-fA-F]{2}', ' ', text)

    # 6. Hapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 7. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Normalisasi kata alay
    words = text.split()
    words = [NORMALISASI.get(w, w) for w in words]

    return " ".join(words)

# ══════════════════════════════════════════════════════════════
# FUNGSI PREDIKSI
# ══════════════════════════════════════════════════════════════

def predict_multiclass(text, model, tokenizer):
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    pad     = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred    = model.predict(pad, verbose=0)[0]
    idx     = int(np.argmax(pred))
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[idx], round(float(np.max(pred)) * 100, 2)

def predict_binary(text, model, tokenizer, threshold=0.5):
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    pad     = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    raw     = float(model.predict(pad, verbose=0)[0][0])
    label   = 1 if raw >= threshold else 0
    conf    = raw if label == 1 else 1 - raw
    return label, round(conf * 100, 2), round(raw, 4)

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "status"  : "OK",
        "message" : "Telkomsel AI API aktif"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' diperlukan"}), 400

    text    = data.get("text", "")
    cleaned = clean_text(text)
    
    

    # ── Sentiment ──
    sentiment, conf_sentiment = predict_multiclass(
        text, model_sentiment, tokenizer_sentiment)

    # ── Hate Speech ──
    hate, conf_hate, raw_hate = predict_binary(
        text, model_hate, tokenizer_hate, threshold=0.55)

    # Override model dengan blacklist kata kasar
    if cek_kata_kasar(cleaned):
        hate      = 1
        conf_hate = 99.0
        raw_hate  = 0.99

    # ── Sarcasm ──
    sarcasm, conf_sarcasm, raw_sarcasm = predict_binary(
        text, model_sarcasm, tokenizer_sarcasm, threshold=0.50)
    if is_sarcasm_rule(cleaned):
        sarcasm = 1
        conf_sarcasm = 92.0
        raw_sarcasm = 1.0

    return jsonify({
        "text"                : text,
        "cleaned_text"        : cleaned,
        "hate_speech"         : hate,
        "sarcasm"             : sarcasm,
        "sentiment"           : sentiment,
        "raw_hate"            : raw_hate,
        "raw_sarcasm"         : raw_sarcasm,
        "confidence_hate"     : conf_hate,
        "confidence_sarcasm"  : conf_sarcasm,
        "confidence_sentiment": conf_sentiment,
        "blacklist_triggered" : bool(cek_kata_kasar(cleaned)),
    })

def is_sarcasm_rule(text):
    sarcasm_patterns = [
        "saking bagus",
        "bagus banget",
        "mantap banget",
        "hebat banget",
        "keren banget",
        "terbaik",
        "recommended"
    ]

    negative_context = [
        "lemot", "gangguan", "hilang", "putus", "mahal",
        "sampah", "jelek", "parah", "error", "gabisa",
        "kaga", "tidak bisa", "gak bisa"
    ]

    return any(p in text for p in sarcasm_patterns) and any(n in text for n in negative_context)

# ── Endpoint cek kesehatan API ──
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "OK",
        "models"  : {
            "sentiment" : "loaded",
            "hate_speech": "loaded",
            "sarcasm"   : "loaded"
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
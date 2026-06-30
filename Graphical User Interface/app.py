import os
import json
import uuid
from datetime import datetime
from predict import predict_image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# inisialisasi aplikasi Flask
app = Flask(__name__)

# konfigurasi folder upload
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# file penyimpanan histori klasifikasi (JSON)
HISTORY_FILE = os.path.join("data", "history.json")
os.makedirs("data", exist_ok=True)

# ekstensi file yang diizinkan
ALLOWED = {"png", "jpg", "jpeg"}

def allowed_file(name):
    # cek apakah file punya ekstensi dan ekstensinya ada di daftar ALLOWED
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


# ── Fungsi bantu histori ──────────────────────────────────────────────────────

def load_history():
    """Muat histori dari file JSON. Kembalikan list kosong jika belum ada."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_history(records):
    """Simpan list histori ke file JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def add_history_record(label, confidence, probs, elapsed, image_file):
    """Tambahkan satu record baru ke histori."""
    records = load_history()
    record = {
        "id":         str(uuid.uuid4())[:8],          # ID unik singkat
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":       datetime.now().strftime("%d %b %Y"),
        "time":       datetime.now().strftime("%H:%M"),
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "probs":      probs,
        "elapsed":    elapsed,
        "image_file": image_file,                      # path relatif dari static/
    }
    records.insert(0, record)   # terbaru di atas
    save_history(records)
    return record


# ── Route ─────────────────────────────────────────────────────────────────────

# halaman beranda
@app.route("/")
def index():
    return render_template("index.html")


# halaman tentang aplikasi
@app.route("/about")
def about():
    return render_template("about.html")


# halaman klasifikasi
@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        # ambil file dari form upload
        f = request.files.get("file")

        # validasi: file harus ada
        if not f or f.filename == "":
            return render_template("classify.html", error="Pilih gambar dulu.")

        # validasi: ekstensi harus valid
        if not allowed_file(f.filename):
            return render_template("classify.html", error="Format harus png/jpg/jpeg.")

        # amankan nama file, tambahkan prefix unik agar tidak tumpang-tindih
        ext      = f.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex[:10]}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        f.save(save_path)

        # panggil fungsi prediksi
        label, confidence, probs, elapsed = predict_image(save_path)

        # simpan ke histori
        image_file = f"uploads/{filename}"
        add_history_record(label, confidence, probs, elapsed, image_file)

        # kirim hasil ke halaman classify
        return render_template(
            "classify.html",
            label=label,
            confidence=round(confidence * 100, 2),
            probs=probs,
            image_file=image_file,
            elapsed=elapsed
        )

    # jika GET, tampilkan halaman upload biasa
    return render_template("classify.html")


# halaman histori klasifikasi
@app.route("/history")
def history():
    records = load_history()
    return render_template("history.html", records=records)


# hapus semua histori (POST untuk keamanan)
@app.route("/history/clear", methods=["POST"])
def history_clear():
    save_history([])
    return redirect(url_for("history"))


# jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
import os
import re
import uuid
import tempfile
import warnings
import shutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import torch
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Dict, Any, List

# --- IMPORT SDK GEMINI TERBARU ---
from google import genai
from google.genai import types

# --- IMPORT FASTAPI & RAG TOOLS ---
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- IMPORT HUGGING FACE & GROQ INFERENCE CLIENT ---
from huggingface_hub import InferenceClient
from groq import Groq

warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

# Cetak setiap request HTTP beserta jam dan status kodenya (Gemini, Groq, HF, Qdrant), supaya
# percobaan ulang otomatis dari SDK ikut terlihat di log. Hapus baris setLevel kalau terlalu ramai.
logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.INFO)

app = FastAPI(title="PRANATA AI Worker (Ingest & Generate)")

# =================================================================
# 1. KONFIGURASI DAN INISIALISASI
# =================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "bps_knowledge_bge_m3_v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Chunk v4 hasil evaluasi: 2000/400, dipotong di akhir kalimat (HARUS sama dengan scripts/local_ingest_v4.py).
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
# Nilai tertinggi yang tidak membuang definisi benar saat kalibrasi; hanya batas bawah, bukan penyaring relevansi.
RAG_SCORE_THRESHOLD = 0.55
# Jatah waktu total satu permintaan narasi. HARUS lebih kecil dari timeout Laravel (300 detik di
# DashboardController::generateNarrative), supaya worker selalu menjawab sebelum Laravel menyerah.
NARRATIVE_BUDGET_SECONDS = 240
# Batas waktu SATU percobaan HTTP ke Gemini. Bawaan klien Interactions tidak punya timeout,
# sehingga request bisa menggantung lama saat server Google sedang ramai.
GEMINI_TIMEOUT_SECONDS = 150

# Init Gemini
client_gemini = None
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY belum diset!", flush=True)
else:
    client_gemini = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={
            "timeout": GEMINI_TIMEOUT_SECONDS * 1000,  # SDK memakai milidetik
            # Klien Interactions otomatis mengulang request yang gagal 408/409/429/5xx (bawaannya
            # sampai 3 kali, tanpa terlihat di log). Dibatasi 1 kali supaya kegagalan cepat terlihat.
            "retry_options": {"attempts": 1},
        },
    )
    print("✅ Terhubung ke Google Gemini API", flush=True)

# Init Hugging Face Inference
client_hf = None
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN (Hugging Face) belum diset!", flush=True)
else:
    # timeout: kalau HF tidak menjawab dalam 60 detik, anggap gagal supaya fallback ke Groq
    # masih sempat berjalan sebelum Laravel menyerah menunggu.
    client_hf = InferenceClient(api_key=HF_TOKEN, timeout=60)
    print("✅ Terhubung ke Hugging Face Inference API", flush=True)

# Init Groq Client (Untuk GPT-OSS-120B & fallback otomatis; Llama 3.3 di Groq dihentikan 16 Agustus 2026)
client_groq = None
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY belum diset! Fitur Groq nonaktif.", flush=True)
else:
    client_groq = Groq(api_key=GROQ_API_KEY)
    print("✅ Terhubung ke Groq Cloud API", flush=True)

# Init Qdrant
client_qdrant = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"✅ Terhubung ke Qdrant Cloud (Collection: {COLLECTION_NAME})", flush=True)
    except Exception as e:
        print(f"⚠️ Gagal koneksi Qdrant: {e}", flush=True)

# Init Embedding
print(f"⏳ Loading Model Embedding: {EMBEDDING_MODEL_NAME}...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Running Embedding on: {device.upper()}", flush=True)

model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

embedding_model = None  # Tetap terdefinisi walaupun load gagal, supaya bisa dicek dengan jelas
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./.cache_models"
    )
    print("✅ Model Embedding Siap.", flush=True)
except Exception as e:
    print(f"❌ Gagal load model embedding: {e}", flush=True)

# Pastikan Qdrant Collection siap
if client_qdrant:
    if not client_qdrant.collection_exists(COLLECTION_NAME):
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )
    
    try:
        client_qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("🔑 Payload Index untuk 'metadata.source' berhasil dipastikan siap.", flush=True)
    except Exception as e:
        print(f"⚠️ Gagal/Sudah ada payload index untuk metadata.source: {e}", flush=True)

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def clean_text(text):
    """Membersihkan teks dari spasi berlebih atau karakter yang tidak diperlukan."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '') 
    return text.strip()

def extract_year_from_filename(filename):
    """Mengekstrak angka (khususnya tahun) dari nama file dokumen yang diunggah."""
    match = re.search(r'\d{4}', filename)
    return int(match.group(0)) if match else 0

def is_document_exist_in_qdrant(filename):
    """Mengecek apakah dokumen dengan nama file tersebut sudah pernah diproses dan tersimpan di database vektor (Qdrant)."""
    if not client_qdrant: return False
    try:
        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=filename))]
        )
        res = client_qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=scroll_filter, limit=1)
        return len(res[0]) > 0
    except:
        return False

# =================================================================
# 3. BACKGROUND TASKS (INGEST)
# =================================================================
def task_ingest_from_files(file_records: List[Dict[str, str]]):
    """Proses latar belakang (background task) untuk mengekstrak teks dari file PDF/TXT, mengubahnya menjadi vektor (embedding), lalu menyimpannya ke Qdrant."""
    print(f"🚀 Memulai Ingestion untuk {len(file_records)} dokumen yang diunggah...")

    if embedding_model is None or client_qdrant is None:
        print("❌ Ingest dibatalkan: model embedding atau Qdrant tidak tersedia.")
        for record in file_records:
            if os.path.exists(record["tmp_path"]):
                os.remove(record["tmp_path"])
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                   separators=CHUNK_SEPARATORS, keep_separator="end")

    for record in file_records:
        tmp_path = record["tmp_path"]
        filename = record["filename"]
        
        if is_document_exist_in_qdrant(filename):
            print(f"⏩ Skip: {filename} sudah ada di Qdrant.")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue
            
        print(f"⬇️ Mengekstrak Dokumen: {filename}...")
        try:
            loader = PyMuPDFLoader(tmp_path)
            raw_docs = loader.load()
            cleaned_docs = []
            file_year = extract_year_from_filename(filename)
            
            for doc in raw_docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata.update({
                    "source": filename, 
                    "year": file_year, 
                    "page": doc.metadata.get("page", 0) + 1
                })
                cleaned_docs.append(doc)

            splits = text_splitter.split_documents(cleaned_docs)
            if not splits:
                print(f"⚠️ Tidak ada teks yang bisa diambil dari {filename} (mungkin PDF hasil scan).")
                continue

            # Embedding sekaligus dalam batch (jauh lebih cepat daripada satu per satu)
            vectors = embedding_model.embed_documents([doc.page_content for doc in splits])

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"page_content": doc.page_content, "metadata": doc.metadata},
                )
                for doc, vector in zip(splits, vectors)
            ]

            # Upsert bertahap (per 256 point) supaya request tidak melewati batas ukuran body Qdrant.
            # Kalau salah satu tahap gagal, point yang SUDAH dikirim oleh task ini dihapus lagi,
            # sehingga tidak ada PDF yang tersimpan setengah jadi. Penghapusan memakai ID point
            # milik task ini (bukan nama file), jadi aman walaupun ada task lain untuk file yang sama.
            UPSERT_BATCH = 256
            sent_ids = []
            try:
                for i in range(0, len(points), UPSERT_BATCH):
                    batch = points[i:i + UPSERT_BATCH]
                    sent_ids.extend(p.id for p in batch)  # dicatat SEBELUM dikirim, untuk jaga-jaga
                    client_qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            except Exception:
                try:
                    client_qdrant.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.PointIdsList(points=sent_ids)
                    )
                    print(f"🧹 Sisa data {filename} yang sempat tersimpan sudah dihapus. File bisa di-upload ulang.")
                except Exception as cleanup_error:
                    print(f"⚠️ Gagal membersihkan sisa data {filename}: {cleanup_error}")
                raise
            print(f"✅ Sukses Ingest ({len(splits)} chunks): {filename}")
                
        except Exception as e:
            print(f"❌ Error Ingest {filename}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
    print("🎉 Proses Ingestion dari File Upload Selesai!")

# =================================================================
# 4. API ENDPOINTS & RAG LOGIC
# =================================================================
class CheckFilesRequest(BaseModel):
    filenames: List[str]

@app.post("/check-missing-files")
def check_missing_files(req: CheckFilesRequest):
    """Endpoint untuk memeriksa apakah ada file yang seharusnya ada tetapi belum masuk ke database vektor."""
    if not client_qdrant:
        return {"missing_filenames": req.filenames}
    try:
        # Setiap putaran hanya mencari file yang BELUM ditemukan. File yang sudah ketemu
        # dikeluarkan dari filter, sehingga chunk-nya tidak terambil lagi.
        # Pengaman: hanya nama yang ada di `remaining` yang dihitung, dan loop berhenti jika
        # satu putaran tidak menemukan nama baru. Jadi `remaining` pasti menyusut minimal 1
        # tiap putaran, dan loop paling banyak berjalan sebanyak jumlah nama yang dicek.
        remaining = set(req.filenames)
        existing_files = set()
        while remaining:
            records, _ = client_qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.source", match=models.MatchAny(any=list(remaining)))]
                ),
                limit=1000,
                with_payload=["metadata.source"],
                with_vectors=False
            )
            if not records:
                break

            found_now = set()
            for record in records:
                metadata = (record.payload or {}).get("metadata")
                source = metadata.get("source") if isinstance(metadata, dict) else None
                if isinstance(source, str) and source in remaining:
                    found_now.add(source)

            if not found_now:
                # Ada record tapi tidak ada nama yang cocok: hentikan supaya tidak berputar selamanya
                print("⚠️ check_missing_files: record ditemukan tapi tidak ada nama yang cocok, loop dihentikan.", flush=True)
                break

            existing_files |= found_now
            remaining -= found_now

        missing_filenames = [filename for filename in req.filenames if filename not in existing_files]
        return {"missing_filenames": missing_filenames}
    except Exception as e:
        print(f"❌ Error saat bulk check_missing_files: {e}")
        return {"missing_filenames": req.filenames}

class DeleteRequest(BaseModel):
    filename: str

@app.delete("/delete-by-file")
def delete_by_file(req: DeleteRequest):
    """Endpoint untuk menghapus semua data vektor yang berasal dari file tertentu di Qdrant."""
    if not client_qdrant:
        return {"status": "error", "message": "Database Qdrant tidak terhubung."}
    try:
        client_qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=req.filename)
                    )
                ]
            )
        )
        print(f"✅ Berhasil menghapus vektor untuk file: {req.filename}")
        return {"status": "success", "message": f"Berhasil menghapus data untuk {req.filename}"}
    except Exception as e:
        print(f"❌ Error saat menghapus {req.filename}: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-all")
def delete_all():
    """Endpoint untuk menghapus keseluruhan koleksi data di Qdrant."""
    if not client_qdrant:
        return {"status": "error", "message": "Database Qdrant tidak terhubung."}
    try:
        # Hapus koleksi untuk mereset data
        if client_qdrant.collection_exists(COLLECTION_NAME):
            client_qdrant.delete_collection(collection_name=COLLECTION_NAME)
        
        # Buat ulang koleksi
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )
        client_qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("✅ Berhasil menghapus seluruh data vektor dan mereset koleksi.")
        return {"status": "success", "message": "Berhasil menghapus seluruh data pengetahuan"}
    except Exception as e:
        print(f"❌ Error saat menghapus seluruh data: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-ingest")
def upload_ingest(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Endpoint API untuk menerima unggahan file, lalu menjadwalkan tugas ingesting (ekstraksi) secara otomatis di latar belakang."""
    file_records = []
    for file in files:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            
        file_records.append({
            "tmp_path": tmp_path, 
            "filename": file.filename
        })

    background_tasks.add_task(task_ingest_from_files, file_records)
    return {
        "status": "started", 
        "message": f"Menerima {len(files)} file. Proses Ingest berjalan di background."
    }

@app.get("/")
def home():
    """Endpoint dasar/root untuk mengecek apakah server FastAPI ini sedang hidup (aktif)."""
    return {
        "status": "Active", 
        "mode": "Worker (Ingest & Generate)",
        "collection": COLLECTION_NAME
    }

class BPSDataInput(BaseModel):
    category: str
    subject: str
    indicator: str
    data_json: Dict[str, Any]
    # Wajib diisi: model dipilih pengguna lewat pengaturan di aplikasi web (tanpa nilai default)
    model_id: str

def parse_table(raw_data):
    """Mengubah data JSON mentah menjadi teks berbentuk tabel atau daftar agar lebih mudah dibaca oleh model AI."""
    try:
        if "headers" in raw_data and "rows" in raw_data:
            headers = [str(h.get('value', '')).strip() for h in raw_data['headers']]
            md_table = "| " + " | ".join(headers) + " |\n"
            md_table += "|" + "|".join(["---" for _ in headers]) + "|\n"
            for row in raw_data['rows']:
                row_vals = [str(c.get('value', '')).strip() for c in row]
                md_table += "| " + " | ".join(row_vals) + " |\n"
            return md_table
        elif isinstance(raw_data, dict):
            md_table = "| Kunci | Nilai |\n|---|---|\n"
            for k, v in raw_data.items():
                md_table += f"| {k} | {v} |\n"
            return md_table
        else:
            return str(raw_data)
    except Exception as e:
        return f"Format data tidak valid: {str(e)}"

_ATAS_DASAR_HARGA_RE = re.compile(r"\batas dasar harga (berlaku|konstan)\b", re.IGNORECASE)

def konsep_dari_nama_indikator(nama: str) -> str:
    """Inti konsep dari nama indikator untuk query RAG, karena nama indikator sering berupa judul tabel:
    "Jumlah Pegawai Negeri Sipil Menurut Jabatan dan Jenis Kelamin" -> "Pegawai Negeri Sipil".
    Hanya untuk pencarian; nama lengkap indikator tetap dikirim ke model."""
    teks = nama.strip()
    adh = _ATAS_DASAR_HARGA_RE.search(teks)  # "Atas Dasar Harga Berlaku/Konstan" bagian dari konsep
    teks = _ATAS_DASAR_HARGA_RE.sub(" ", teks)
    teks = re.sub(r"^\s*\[[^\]]*\]\s*", "", teks)
    teks = re.sub(r"\s*\((?![A-Z][A-Z0-9]{0,5}\))[^)]*\)", "", teks)
    teks = re.sub(r"\s+(menurut|berdasarkan)\s+.*$", "", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\s+per\s+(kecamatan|kelurahan|desa|kabupaten|kota|wilayah)\b.*$", "", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\s+hasil\s+.*$", "", teks, flags=re.IGNORECASE)
    teks = re.sub(r"^(jumlah|banyaknya)\s+", "", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\b(triwulanan|tahunan|bulanan)\b", " ", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\b(19|20)\d{2}\b", " ", teks)
    teks = re.sub(r"\s{2,}", " ", teks).strip(" ,-")
    if adh:
        teks = f"{teks} {adh.group(0)}".strip()
    return teks or nama.strip()

def get_rag_context(query, limit=3):
    """Mencari referensi dokumen terdekat dari Qdrant (RAG - Retrieval Augmented Generation) berdasarkan query untuk diberikan kepada AI sebagai contekan."""
    if not client_qdrant:
        return "Database Qdrant tidak terhubung."
    if embedding_model is None:
        return "Model embedding tidak tersedia, referensi tidak dapat diambil."
    try:
        vector = embedding_model.embed_query(query)
        # Pencarian eksak: indeks HNSW terbukti melewatkan hasil terbaik di korpus ini; waktunya praktis sama.
        pencarian_eksak = models.SearchParams(exact=True)
        try:
            res = client_qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit,
                score_threshold=RAG_SCORE_THRESHOLD,
                search_params=pencarian_eksak,
                with_payload=True,
            )
            points = res.points
        except AttributeError:
            res = client_qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=limit,
                score_threshold=RAG_SCORE_THRESHOLD,
                search_params=pencarian_eksak,
                with_payload=True,
            )
            points = res

        contexts = []
        for point in points:
            source = point.payload.get("metadata", {}).get("source", "Dokumen BPS")
            year = point.payload.get("metadata", {}).get("year", "?")
            page = point.payload.get("metadata", {}).get("page", "?")
            content = point.payload.get("page_content", "")
            contexts.append(
                f"--- [Sumber: {source} | Thn: {year} | Hal: {page}] ---\n{content}"
            )

        # === TAMBAHKAN KODE PRINT INI AGAR MUNCUL DI LOGS TERMINAL ===
        if contexts:
            print(
                f"\n🔍 [RAG RETRIEVAL SUKSES] Ditemukan {len(contexts)} dokumen referensi:",
                flush=True,
            )
            for idx, ctx in enumerate(contexts, 1):
                print(f"[{idx}] {ctx[:150]}... [lanjut ke sistem]", flush=True)
            print("=" * 60, flush=True)
        else:
            print(
                "\n⚠️ [RAG KOSONG] Tidak ada referensi PDF yang cocok dengan threshold.",
                flush=True,
            )
        # =============================================================

        return (
            "\n\n".join(contexts)
            if contexts
            else "Tidak ditemukan referensi spesifik di database."
        )
    except Exception as e:
        print(f"RAG Search Error: {e}", flush=True)
        return "Gagal mengakses Knowledge Base."

# =================================================================
# SYSTEM PROMPT RINGKAS (Llama 3.3 70B dan GPT-OSS 120B, semua penyedia)
# =================================================================
# Awalnya dibuat untuk Groq (hemat token). Kini dipakai Llama dan GPT-OSS di penyedia mana pun,
# sama dengan konfigurasi ringkas yang digunakan saat evaluasi. Gemini memakai SYSTEM_PROMPT_NARASI.
def get_groq_compact_prompt():
    return """Anda adalah analis data senior dan editor publikasi Badan Pusat Statistik (BPS) yang berpengalaman menyusun Berita Resmi Statistik. Tulis narasi analisis statistik dalam Bahasa Indonesia baku yang akurat secara matematis dan setara kualitas publikasi resmi BPS.

LANGKAH 1 — ANALISIS SINGKAT (ditulis lebih dulu, maksimal 6 baris)
<langkah_analisis>
1. Periodisasi & satuan: [tahunan/bulanan/triwulanan; satuan data]
2. Nilai tertinggi & terendah: [nilai, periode, dan wilayah/kategori; hanya satu tertinggi dan satu terendah]
3. Dua data terakhir: [nilai terbaru, nilai sebelumnya, selisih; persentase perubahan hanya jika satuan data bukan persen]
4. Pola tren: [tanda arah setiap perubahan berurutan: + naik, - turun, = tetap; nama pola; periode titik balik]
</langkah_analisis>
Pola tren (baris 4) ditentukan dari tanda arah: semua + berarti "konsisten naik"; semua - berarti "konsisten turun"; semua = berarti "stagnan"; jika tanda + minimal dua kali lebih banyak daripada tanda - dan nilai akhir lebih tinggi daripada nilai awal, berarti "cenderung naik"; kebalikannya "cenderung turun"; selain itu "fluktuatif". Titik balik adalah periode tempat tanda berubah dari + ke - atau sebaliknya. Jika tabel memuat banyak wilayah/kategori, baris 4 hanya untuk seri utama (baris Jumlah/Total atau wilayah induk).
Jangan mendaftar seluruh data. Tag </langkah_analisis> WAJIB ditutup, lalu LANGSUNG tulis narasi. Narasi adalah bagian terpenting dan tidak boleh kosong.

LANGKAH 2 — NARASI (3 paragraf, atau 4 jika ada rincian wilayah/kategori)
Paragraf 1 — Definisi: apa indikator ini, maknanya, dan cara membaca angkanya berdasarkan referensi (diparafrasekan, bukan disalin). Tutup dengan rentang periode data yang tersedia.
Paragraf 2 — Historis: pergerakan data secara kronologis sesuai pola tren dan titik balik pada baris 4, serta nilai tertinggi dan terendah beserta periode/wilayahnya. Jangan menyebut rata-rata.
Paragraf 3 — Terkini (paling penting): nilai terbaru, nilai sebelumnya, dan selisihnya; persentase perubahan hanya untuk data yang satuannya bukan persen. Untuk data bulanan/triwulanan, tambahkan perbandingan dengan periode yang sama tahun sebelumnya jika datanya tersedia; jika tidak, lewati tanpa mengarang angka.
Paragraf 4 (opsional): hanya jika data memuat rincian wilayah/kategori/komponen; sebutkan yang paling menonjol.

ATURAN
1. Desimal memakai koma (14,52), ribuan memakai titik (1.234.567). Nilai data ditulis persis seperti tabel; selisih mengikuti jumlah desimal data; persentase perubahan ditulis dua angka di belakang koma.
2. Sertakan satuan jika indikator memiliki satuan; untuk indeks, selisih disebut "poin". Untuk data bersatuan persen (tingkat, rasio persen, inflasi, pertumbuhan), selisih disebut "persen poin" dan JANGAN menulis persentase perubahannya, karena persen dari persen membingungkan pembaca. Persentase perubahan hanya untuk data berupa jumlah atau indeks (jiwa, rupiah, ton, IPM).
3. Jangan mengarang angka; setiap angka harus berasal dari data atau dihitung dari data.
4. Mulai langsung dari kalimat pertama narasi tanpa judul atau label, dan jangan membuka dengan "Mengacu pada...".
5. Tanpa Markdown (tanda bintang, pagar, tanda hubung sebagai poin, atau penomoran); tulis paragraf teks biasa.
6. Tanpa kesimpulan, opini, proyeksi, atau rekomendasi kebijakan.
7. Bahasa Indonesia baku tanpa istilah asing. Hindari istilah teknis (delta, MoM, YoY, observasi, anomali); gunakan "selisih", "perubahan", atau "dibandingkan periode yang sama tahun sebelumnya".
8. Gaya formal khas publikasi resmi, tetapi mengalir dan mudah dipahami masyarakat umum.

CONTOH KELUARAN (hanya untuk menunjukkan format dan gaya penulisan; JANGAN menyalin angka, periode, maupun nama wilayahnya)
<langkah_analisis>
1. Periodisasi & satuan: tahunan; indeks
2. Nilai tertinggi & terendah: tertinggi 70,82 (2024); terendah 68,45 (2019)
3. Dua data terakhir: 70,82 (2024) dan 70,28 (2023); selisih +0,54 poin; persentase perubahan 0,77 persen
4. Pola tren: + + + + + (konsisten naik); tidak ada titik balik
</langkah_analisis>
Indeks Pembangunan Manusia (IPM) merupakan indikator komposit yang mengukur capaian pembangunan manusia berbasis sejumlah komponen dasar kualitas hidup, meliputi dimensi umur panjang dan hidup sehat, pengetahuan, serta standar hidup layak. Semakin tinggi nilai IPM, semakin baik kualitas pembangunan manusia di suatu wilayah. Berdasarkan data yang tersedia dari tahun 2019 hingga 2024, berikut adalah perkembangan IPM di Kabupaten Contoh.

Sepanjang periode 2019 hingga 2024, IPM Kabupaten Contoh menunjukkan tren yang konsisten meningkat. Pada tahun 2019, IPM tercatat sebesar 68,45 dan terus meningkat hingga mencapai 70,82 pada tahun 2024, dengan kenaikan paling kecil terjadi pada tahun 2020 yang hanya sebesar 0,12 poin. Nilai tertinggi tercatat pada tahun 2024 sebesar 70,82, sedangkan nilai terendah terjadi pada tahun 2019 sebesar 68,45.

Pada tahun 2024, IPM Kabupaten Contoh tercatat sebesar 70,82, meningkat sebesar 0,54 poin atau 0,77 persen dibandingkan tahun 2023 yang tercatat sebesar 70,28. Kenaikan ini melanjutkan tren peningkatan IPM yang stabil dalam lima tahun terakhir. Dengan capaian 70,82, IPM Kabupaten Contoh telah melampaui angka 70 yang menandai kelompok pembangunan manusia kategori tinggi.
"""

# =================================================================
# SYSTEM PROMPT NARASI (Gemini & Hugging Face)
# Ditulis tanpa indentasi supaya spasi tidak ikut terhitung sebagai token.
# Setiap aturan hanya ditulis SATU kali, dan contoh narasi sudah disesuaikan dengan aturan.
# =================================================================
SYSTEM_PROMPT_NARASI = """Anda adalah analis data senior dan editor publikasi Badan Pusat Statistik (BPS) yang berpengalaman menyusun Berita Resmi Statistik. Tugas Anda adalah menulis narasi analisis statistik dalam Bahasa Indonesia baku yang akurat secara matematis dan setara kualitas publikasi resmi BPS.

================================================================
BAGIAN A — ANALISIS SINGKAT (DITULIS LEBIH DULU)
================================================================
Sebelum narasi, tulis analisis singkat PERSIS dengan format berikut (maksimal 6 baris):

<langkah_analisis>
1. Periodisasi & satuan: [tahunan/bulanan/triwulanan; satuan data]
2. Nilai tertinggi & terendah: [nilai, periode, dan wilayah/kategori; hanya satu tertinggi dan satu terendah]
3. Dua data terakhir: [nilai terbaru, nilai sebelumnya, selisih; persentase perubahan hanya jika satuan data bukan persen; untuk data bulanan/triwulanan tambahkan perbandingan dengan periode yang sama tahun sebelumnya jika datanya tersedia]
4. Pola tren: [tanda arah setiap perubahan berurutan: + naik, - turun, = tetap; nama pola; periode titik balik]
</langkah_analisis>

Ketentuan bagian ini:
- Pola tren (baris 4) ditentukan dari tanda arah: semua + berarti "konsisten naik"; semua - berarti "konsisten turun"; semua = berarti "stagnan"; jika tanda + minimal dua kali lebih banyak daripada tanda - dan nilai akhir lebih tinggi daripada nilai awal, berarti "cenderung naik"; kebalikannya "cenderung turun"; selain itu "fluktuatif". Titik balik adalah periode tempat tanda berubah dari + ke - atau sebaliknya. Jika tabel memuat banyak wilayah/kategori, baris 4 hanya untuk seri utama (baris Jumlah/Total atau wilayah induk).
- JANGAN mendaftar seluruh data atau menulis perhitungan untuk setiap periode; untuk tren cukup tuliskan tandanya. Perhitungan lain cukup dilakukan tanpa ditulis.
- Tag </langkah_analisis> WAJIB ditutup, lalu LANGSUNG tulis narasi. Narasi adalah bagian terpenting dan tidak boleh kosong.

================================================================
BAGIAN B — NARASI (3 PARAGRAF, ATAU 4 JIKA ADA RINCIAN)
================================================================
Paragraf 1 — Definisi (2–4 kalimat)
Jelaskan apa indikator ini, bagaimana diukur, dan cara membaca angkanya (misalnya "semakin tinggi semakin baik"), berdasarkan KONTEKS REFERENSI. Parafrasekan dengan kata-kata sendiri, terjemahkan jika referensi berbahasa Inggris, dan jelaskan maknanya, bukan rumusnya. Jika referensi tidak relevan, gunakan definisi umum resmi BPS. Tutup paragraf dengan rentang periode data, misalnya: "Berdasarkan data yang tersedia dari tahun 2019 hingga 2024, berikut adalah perkembangan [indikator] di [wilayah]."

Paragraf 2 — Perkembangan historis (3–5 kalimat)
Ceritakan pergerakan data secara kronologis sesuai pola tren dan titik balik pada baris 4 analisis: kapan naik, kapan turun, dan kapan arahnya berbalik. Sebutkan nilai tertinggi dan terendah beserta periodenya, serta wilayah/kategorinya jika ada. Jangan menyebut rata-rata.

Paragraf 3 — Perubahan terkini (3–5 kalimat, paling penting)
Bandingkan dua data terakhir dengan mencantumkan nilai terbaru, nilai sebelumnya, selisih, dan arahnya. Persentase perubahan hanya ditulis jika satuan data BUKAN persen.
- Data tahunan berupa jumlah atau indeks: "Pada tahun [terbaru], [indikator] di [wilayah] tercatat sebesar [nilai] [satuan], [meningkat/menurun] sebesar [selisih] [satuan] atau [X] persen dibandingkan tahun [sebelumnya] yang tercatat sebesar [nilai] [satuan]."
- Data tahunan bersatuan persen: "Pada tahun [terbaru], [indikator] di [wilayah] tercatat sebesar [nilai] persen, [naik/turun] [selisih] persen poin dibandingkan tahun [sebelumnya] yang tercatat sebesar [nilai] persen."
- Data bulanan/triwulanan: tulis perbandingan dengan periode sebelumnya, lalu perbandingan dengan periode yang sama tahun sebelumnya JIKA datanya tersedia. Jika tidak tersedia, lewati tanpa mengarang angka.
Paragraf ini boleh ditutup dengan satu atau dua kalimat yang mengaitkan perubahan terkini dengan tren, rekor tertinggi/terendah, atau cara membaca angka indikator.

Paragraf 4 — Rincian (opsional)
Tulis hanya jika data memuat rincian wilayah, kategori, atau komponen: sebutkan bagian yang paling menonjol atau paling berpengaruh terhadap perubahan.

================================================================
BAGIAN C — ATURAN PENULISAN
================================================================
1. Format angka: desimal memakai koma (14,52) dan ribuan memakai titik (1.234.567). Nilai data ditulis persis seperti di tabel. Selisih mengikuti jumlah desimal data; persentase perubahan ditulis dengan dua angka di belakang koma.
2. Satuan: sertakan satuan pada setiap angka jika indikator memiliki satuan (persen, jiwa, rupiah, ton, hektar, dst.). Untuk indeks tanpa satuan, selisihnya disebut "poin".
3. Data bersatuan persen (tingkat, rasio persen, inflasi, pertumbuhan): selisih disebut "persen poin", dan JANGAN menulis persentase perubahannya, karena persen dari persen membingungkan pembaca dan tidak lazim dalam publikasi BPS. Persentase perubahan hanya dipakai untuk data berupa jumlah atau indeks (jiwa, rupiah, ton, hektar, IPM, IHK).
4. Jangan mengarang angka. Setiap angka harus berasal dari data atau dihitung dari data.
5. Mulai langsung dari kalimat pertama narasi tanpa judul atau label, dan jangan membuka dengan "Mengacu pada..." atau nama publikasi.
6. Tanpa format Markdown (tanda bintang, pagar, garis bawah, tanda hubung sebagai poin, atau penomoran). Tulis paragraf teks biasa.
7. Tanpa kesimpulan, opini, proyeksi, rekomendasi kebijakan, atau kalimat normatif seperti "diperlukan upaya...".
8. Seluruh narasi dalam Bahasa Indonesia baku tanpa istilah asing.
9. Hindari istilah teknis yang sulit dipahami masyarakat umum (delta, MoM, YoY, observasi, anomali, titik data). Gunakan "selisih", "perubahan", "dibandingkan bulan sebelumnya", atau "dibandingkan periode yang sama tahun sebelumnya".
10. Gaya bahasa formal khas publikasi resmi, tetapi mengalir dan mudah dipahami. Diksi yang dianjurkan: "tercatat sebesar", "meningkat/menurun sebesar", "dibandingkan dengan", "mencapai titik tertinggi", "berada pada titik terendah", "bergerak fluktuatif".

================================================================
CONTOH NARASI
================================================================
Contoh data tahunan (Indeks Pembangunan Manusia):
"Indeks Pembangunan Manusia (IPM) merupakan indikator komposit yang mengukur capaian pembangunan manusia berbasis sejumlah komponen dasar kualitas hidup, meliputi dimensi umur panjang dan hidup sehat, pengetahuan, serta standar hidup layak. Semakin tinggi nilai IPM, semakin baik kualitas pembangunan manusia di suatu wilayah. Berdasarkan data yang tersedia dari tahun 2019 hingga 2024, berikut adalah perkembangan IPM di Kabupaten Contoh.

Sepanjang periode 2019 hingga 2024, IPM Kabupaten Contoh menunjukkan tren yang konsisten meningkat. Pada tahun 2019, IPM tercatat sebesar 68,45 dan terus meningkat hingga mencapai 70,82 pada tahun 2024, dengan kenaikan paling kecil terjadi pada tahun 2020 yang hanya sebesar 0,12 poin. Nilai tertinggi tercatat pada tahun 2024 sebesar 70,82, sedangkan nilai terendah terjadi pada tahun 2019 sebesar 68,45.

Pada tahun 2024, IPM Kabupaten Contoh tercatat sebesar 70,82, meningkat sebesar 0,54 poin atau 0,77 persen dibandingkan tahun 2023 yang tercatat sebesar 70,28. Kenaikan ini melanjutkan tren peningkatan IPM yang stabil dalam lima tahun terakhir. Dengan capaian 70,82, IPM Kabupaten Contoh telah melampaui angka 70 yang menandai kelompok pembangunan manusia kategori tinggi."

Contoh data bulanan (Inflasi):
"Inflasi merupakan kecenderungan naiknya harga barang dan jasa secara umum dan terus-menerus dalam jangka waktu tertentu. Tingkat inflasi dihitung berdasarkan perubahan Indeks Harga Konsumen (IHK) dari satu periode ke periode berikutnya. Nilai inflasi positif menunjukkan terjadinya kenaikan harga secara umum, sedangkan nilai negatif atau deflasi menunjukkan penurunan harga. Berdasarkan data yang tersedia dari Januari 2024 hingga Juni 2025, berikut adalah perkembangan inflasi bulanan di Kota Contoh.

Sepanjang periode Januari 2024 hingga Juni 2025, inflasi bulanan Kota Contoh bergerak fluktuatif dengan kisaran antara negatif 0,12 persen hingga 1,05 persen. Inflasi tertinggi tercatat pada Desember 2024 sebesar 1,05 persen, sementara deflasi terdalam terjadi pada Maret 2025 sebesar negatif 0,12 persen. Secara umum, inflasi cenderung meningkat pada akhir tahun dan melambat pada awal tahun berikutnya.

Pada Juni 2025, inflasi Kota Contoh tercatat sebesar 0,34 persen, naik 0,16 persen poin dibandingkan Mei 2025 yang tercatat sebesar 0,18 persen. Jika dibandingkan dengan Juni 2024 yang tercatat sebesar 0,27 persen, inflasi bulan ini lebih tinggi 0,07 persen poin. Tingkat inflasi bulan ini merupakan yang tertinggi dalam semester pertama tahun 2025."
"""


def build_user_prompt(category, indicator, subject, context, data_table):
    """Susun user prompt: informasi indikator, konteks RAG, dan tabel data (tanpa indentasi)."""
    return f"""Kategori : {category}
Indikator: {indicator}
Wilayah  : {subject}

KONTEKS REFERENSI (dasar definisi indikator; parafrasekan, jangan disalin mentah):
{context}

DATA STATISTIK YANG DIANALISIS:
{data_table}

Tulis <langkah_analisis> singkat sesuai format, tutup tagnya, lalu tulis narasi sesuai instruksi sistem. Pastikan setiap angka di narasi sesuai dengan data di atas."""

# =================================================================
# HELPER: PEMBERSIHAN TAG <langkah_analisis> & DETEKSI OUTPUT BOCOR
# =================================================================
# Toleran terhadap variasi spasi/kapitalisasi tag dari model, dan terhadap
# tag penutup yang hilang (biasanya karena generasi terpotong oleh max_tokens).
_ANALYSIS_TAG_RE = re.compile(r'<\s*langkah_analisis\s*>.*?<\s*/\s*langkah_analisis\s*>', re.DOTALL | re.IGNORECASE)
_ANALYSIS_OPEN_RE = re.compile(r'<\s*langkah_analisis\s*>', re.IGNORECASE)
_ANALYSIS_CLOSE_RE = re.compile(r'<\s*/\s*langkah_analisis\s*>', re.IGNORECASE)
# Judul baris yang HANYA muncul di template langkah_analisis, tidak pernah di narasi prosa.
_LEAKED_ANALYSIS_HEADINGS_RE = re.compile(
    r'^\s*\d+\.\s*(tipe periodisasi|periodisasi\s*&?\s*satuan|nilai tertinggi\s*(dan|&)\s*terendah|analisis\s*\d*\s*data terakhir|dua data terakhir|pola tren)',
    re.IGNORECASE | re.MULTILINE
)

def strip_analysis_tag(text: str) -> str:
    """Buang blok <langkah_analisis>...</langkah_analisis>. Jika tag pembuka ada
    tapi tag penutup hilang (generasi terpotong), tutup dulu di akhir teks supaya
    sisa chain-of-thought yang terpotong ikut terbuang, bukan malah lolos ke user."""
    if not text:
        return ""
    if _ANALYSIS_OPEN_RE.search(text) and not _ANALYSIS_CLOSE_RE.search(text):
        text = text + "</langkah_analisis>"
    return _ANALYSIS_TAG_RE.sub('', text).strip()

def looks_like_leaked_analysis(text: str) -> bool:
    """Heuristik: teks yang tersisa setelah stripping ternyata masih berupa
    bocoran langkah analisis (daftar bernomor/berpoin pendek), bukan narasi
    prosa 3 paragraf. Ini terjadi saat model TIDAK PERNAH memakai tag
    <langkah_analisis> sama sekali (jadi tidak ada yang bisa di-strip) atau
    berhenti generate sebelum sempat menulis narasi."""
    if not text:
        return True
    if _LEAKED_ANALYSIS_HEADINGS_RE.search(text):
        return True
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return True
    list_like = sum(1 for l in lines if re.match(r'^\s*(\d+\.|\*|-)\s', l))
    # Narasi golden umumnya >500 karakter & berupa paragraf, bukan daftar.
    if len(text) < 500 and list_like >= max(2, len(lines) // 2):
        return True
    return False

def temperature_untuk(model_id):
    """Temperature sesuai anjuran resmi pengembang: GPT-OSS 1.0 (OpenAI), Llama 3.3 0.6 (Meta)."""
    ml = model_id.lower()
    if "gpt-oss" in ml:
        return 1.0
    if "llama" in ml or "versatile" in ml:
        return 0.6
    return 0.3

def build_compact_prompt(category, subject, indicator, data_table, context, extra_instruction=""):
    """Susun system prompt dan user prompt RINGKAS untuk Llama 3.3 70B dan GPT-OSS 120B."""
    system_ringkas = get_groq_compact_prompt() + extra_instruction
    user_ringkas = f"""Kategori: {category}
Indikator: {indicator}
Wilayah: {subject}

Referensi:
{context}

Data:
{data_table}

Tulis narasi analisis sesuai instruksi sistem."""
    return system_ringkas, user_ringkas

def _ambil(obj, nama):
    """Ambil field dari objek SDK atau dict (Groq memberi objek, Hugging Face kadang memberi dict)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(nama)
    return getattr(obj, nama, None)

def log_token_openai(label, chat, durasi_api):
    """Cetak pemakaian token dari respons format OpenAI (Groq & Hugging Face), setara log 📊 Gemini.
    Di format ini completion_tokens SUDAH termasuk token berpikir, jadi output = completion - berpikir,
    supaya artinya sama dengan "output" di log Gemini (teks yang terlihat saja)."""
    try:
        usage = getattr(chat, "usage", None)
        completion = _ambil(usage, "completion_tokens")
        berpikir = _ambil(_ambil(usage, "completion_tokens_details"), "reasoning_tokens")
        if berpikir is None or completion is None:
            # Llama tidak berpikir; untuk GPT-OSS sebagian penyedia tidak memisahkan token berpikir.
            catatan = " (sudah termasuk token berpikir)" if "gpt-oss" in label.lower() else ""
            rincian = f"berpikir: -, output: {completion}{catatan}"
        else:
            rincian = f"berpikir: {berpikir}, output: {completion - berpikir}"
        # finish_reason "length" = output terpotong oleh max_tokens; "stop" = model berhenti sendiri.
        finish = getattr(chat.choices[0], "finish_reason", None) if chat.choices else None
        antre = _ambil(usage, "queue_time")  # khusus Groq: lama request menunggu antrean server
        info_antre = f", antre: {antre:.1f} detik" if isinstance(antre, (int, float)) else ""
        print(f"📊 Token {label} — {rincian}, selesai karena: {finish}{info_antre}, "
              f"durasi API: {durasi_api:.1f} detik", flush=True)
    except Exception as log_error:
        print(f"⚠️ Gagal mencatat pemakaian token {label}: {log_error}", flush=True)

def generate_via_groq_fallback(category, subject, indicator, data_table, rag_query,
                                groq_model="openai/gpt-oss-120b", context=None, extra_instruction=""):
    """Jalankan narasi via Groq Cloud dengan prompt ringkas (dipakai untuk GPT-OSS 120B)."""
    # Reuse context yang sudah diambil kalau ada (hemat 1 panggilan Qdrant); kalau tidak,
    # ambil ulang dengan limit kecil supaya muat di TPM limit Groq.
    groq_context = context if context is not None else get_rag_context(rag_query, limit=1)
    groq_system, groq_user = build_compact_prompt(
        category, subject, indicator, data_table, groq_context, extra_instruction
    )

    # Khusus gpt-oss: level penalaran medium supaya tidak kehabisan token sebelum narasi ditulis.
    # Dikirim lewat extra_body supaya tetap jalan walaupun versi SDK groq di Space belum
    # mengenal argumen reasoning_effort (argumen langsung bisa memicu TypeError di SDK lama).
    extra_params = {"extra_body": {"reasoning_effort": "medium"}} if "gpt-oss" in groq_model else {}

    t_api = time.monotonic()
    chat = client_groq.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": groq_system},
            {"role": "user", "content": groq_user}
        ],
        temperature=temperature_untuk(groq_model),
        max_tokens=4096,
        **extra_params
    )
    log_token_openai(f"Groq ({groq_model})", chat, time.monotonic() - t_api)
    return chat.choices[0].message.content

# =================================================================
# PEMETAAN MODEL KE PENYEDIA
# =================================================================
# - Gemini        : Google Gemini API, semua versi (3, 3.5, 3.6 Flash) lewat Interactions API dengan
#                   pengaturan yang sama: system instruction terpisah, thinking HIGH, output maks. 16.384 token.
# - Llama 3.3 70B : HANYA Hugging Face (Llama 3.3 70B Versatile di Groq dihentikan sejak 16 Agustus 2026).
# - GPT-OSS 120B  : Groq lebih dulu; kalau Groq gagal (misalnya kena rate limit), dialihkan ke model
#                   yang SAMA di Hugging Face. Tidak pernah dialihkan ke model lain.
HF_LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
HF_GPT_OSS_MODEL = "openai/gpt-oss-120b"
GROQ_GPT_OSS_MODEL = "openai/gpt-oss-120b"

def call_hf(model_id, system_prompt, user_prompt, extra_instruction=""):
    """Jalankan narasi via Hugging Face Inference API dengan prompt yang diberikan
    (Llama dan GPT-OSS memakai prompt ringkas dari build_compact_prompt)."""
    if not client_hf:
        raise Exception("Koneksi Hugging Face belum dikonfigurasi (HF_TOKEN kosong).")
    print(f"🚀 Mengerjakan via Hugging Face Inference API ({model_id})...", flush=True)
    hf_system = system_prompt + extra_instruction
    # gpt-oss membaca level penalaran dari baris "Reasoning: low/medium/high" di system prompt.
    # "medium" cukup untuk hitungan selisih & persentase tanpa menghabiskan jatah max_tokens.
    if "gpt-oss" in model_id.lower():
        hf_system = "Reasoning: medium\n" + hf_system
    t_api = time.monotonic()
    chat = client_hf.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": hf_system},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature_untuk(model_id),
        max_tokens=4096
    )
    log_token_openai(f"HF ({model_id})", chat, time.monotonic() - t_api)
    return chat.choices[0].message.content

def call_narrative_provider(target_model, system_prompt, user_prompt, context, rag_query,
                             category, subject, indicator, data_table, extra_instruction=""):
    """Dispatch ke penyedia sesuai model yang DIPILIH USER dan kembalikan (narrative, label_model).
    Gemini memakai prompt lengkap (system_prompt/user_prompt); Llama dan GPT-OSS memakai prompt ringkas.
    Peralihan penyedia hanya terjadi untuk model yang sama (GPT-OSS: Groq -> Hugging Face)."""
    tl = (target_model or "").lower()

    # 1. GOOGLE GEMINI (semua versi lewat Interactions API dengan pengaturan yang sama)
    if "gemini" in tl:
        if not client_gemini:
            raise Exception("Koneksi API Gemini belum dikonfigurasi (GEMINI_API_KEY kosong).")
        # Flash-Lite tetap minimal seperti sebelumnya; model Flash lainnya HIGH.
        thinking_level = "minimal" if "lite" in tl else "high"
        print(f"🚀 Mengerjakan via Google Gemini API ({target_model}, Interactions, thinking {thinking_level})...", flush=True)

        t_api = time.monotonic()
        # temperature tidak diatur (bawaan 1.0): anjuran Google untuk Gemini 3; di bawah 1.0 bisa looping.
        interaction = client_gemini.interactions.create(
            model=target_model,
            system_instruction=system_prompt + extra_instruction,
            input=user_prompt,
            generation_config={
                "thinking_level": thinking_level,
                "max_output_tokens": 16384,  # ruang untuk berpikir HIGH + narasi, supaya tidak terpotong
            },
        )
        # output_text kosong jika seluruh token habis untuk berpikir (status "incomplete")
        narrative = interaction.output_text or ""

        # Catat pemakaian token untuk memantau apakah batas masih cukup.
        # Dibungkus try/except sendiri: baris log tidak boleh menggagalkan request yang berhasil.
        try:
            usage = interaction.usage
            thoughts = getattr(usage, "total_thought_tokens", None) if usage else None
            output = getattr(usage, "total_output_tokens", None) if usage else None
            print(f"📊 Token Gemini — berpikir: {thoughts}, output: {output}, status: {interaction.status}, "
                  f"durasi API: {time.monotonic() - t_api:.1f} detik", flush=True)
        except Exception as log_error:
            print(f"⚠️ Gagal mencatat pemakaian token Gemini: {log_error}", flush=True)
        return narrative, target_model

    # 2. GPT-OSS 120B: Groq lebih dulu, lalu Hugging Face (model yang sama) sebagai cadangan
    elif "gpt-oss" in tl:
        groq_error = None
        if client_groq:
            try:
                print(f"🚀 Mengerjakan via Groq Cloud API ({GROQ_GPT_OSS_MODEL})...", flush=True)
                narrative = generate_via_groq_fallback(
                    category, subject, indicator, data_table, rag_query,
                    groq_model=GROQ_GPT_OSS_MODEL, context=context, extra_instruction=extra_instruction
                )
                return narrative, f"Groq ({GROQ_GPT_OSS_MODEL})"
            except Exception as e:
                groq_error = e
                print(f"⚠️ Groq gagal untuk GPT-OSS: {e}", flush=True)
                print(f"🔄 [Fallback] GPT-OSS dialihkan ke Hugging Face ({HF_GPT_OSS_MODEL}), model yang sama.", flush=True)
        else:
            print("⚠️ GROQ_API_KEY kosong; GPT-OSS langsung dijalankan via Hugging Face.", flush=True)
        sys_ringkas, user_ringkas = build_compact_prompt(
            category, subject, indicator, data_table, context, extra_instruction
        )
        try:
            narrative = call_hf(HF_GPT_OSS_MODEL, sys_ringkas, user_ringkas)
        except Exception as hf_error:
            if groq_error:
                raise Exception(f"Groq gagal ({groq_error}); fallback ke Hugging Face juga gagal ({hf_error})")
            raise
        if groq_error:
            return narrative, f"{HF_GPT_OSS_MODEL} [Fallback dari Groq]"
        return narrative, HF_GPT_OSS_MODEL

    # 3. LLAMA 3.3 70B: hanya Hugging Face
    elif "llama" in tl:
        sys_ringkas, user_ringkas = build_compact_prompt(
            category, subject, indicator, data_table, context, extra_instruction
        )
        narrative = call_hf(HF_LLAMA_MODEL, sys_ringkas, user_ringkas)
        return narrative, HF_LLAMA_MODEL

    else:
        raise Exception(f"Model '{target_model}' tidak dikenali. Pilih model yang tersedia di menu Model.")

# =================================================================
# PEMBATAS WAKTU PANGGILAN MODEL
# =================================================================
# Menjamin worker selalu menjawab Laravel dalam NARRATIVE_BUDGET_SECONDS, apa pun yang terjadi di
# dalam SDK (request menggantung, percobaan ulang otomatis, server penyedia sedang ramai).
# Kalau lewat batas, request dianggap gagal dengan pesan jelas; thread latar dibiarkan selesai sendiri.
_model_executor = ThreadPoolExecutor(max_workers=8)

def panggil_dengan_batas_waktu(batas_detik, fn, *args, **kwargs):
    """Jalankan fn(*args, **kwargs), tetapi menyerah setelah batas_detik."""
    future = _model_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=max(batas_detik, 1))
    except FuturesTimeout:
        if future.done():
            raise  # TimeoutError dari dalam fn sendiri, bukan karena batas waktu ini
        future.cancel()  # hanya berhasil kalau belum sempat berjalan
        raise Exception(
            f"Model tidak menjawab dalam {batas_detik:.0f} detik (server penyedia kemungkinan sedang ramai). "
            "Coba lagi beberapa menit lagi atau pilih model lain."
        )

@app.post("/generate-narrative")
def generate_narrative(input_data: BPSDataInput):
    """Endpoint API Utama! Menerima perintah dari web Laravel, menyusun konteks (RAG), memanggil AI, lalu mengembalikan teks narasinya ke Laravel."""
    mulai = time.monotonic()
    data_table = parse_table(input_data.data_json)
    # Query RAG memakai konsep indikator; nama lengkap indikator tetap dipakai di prompt ke model.
    rag_query = f"Definisi konsep {konsep_dari_nama_indikator(input_data.indicator)} dan cara interpretasinya menurut BPS."
    print(f"🔎 Query RAG: {rag_query}", flush=True)
    
    if not (input_data.model_id or "").strip():
        return {
            "status": "error",
            "error_message": "Model AI belum dipilih. Pilih model terlebih dahulu pada menu pengaturan model AI.",
            "used_model": "",
            "rag_preview": ""
        }

    # Deteksi apakah akan butuh Groq (untuk menentukan jumlah RAG context)
    target_model = input_data.model_id
    # Llama dan GPT-OSS memakai konfigurasi ringkas: prompt ringkas dan konteks RAG 1 potongan
    # (sama dengan konfigurasi saat evaluasi). Gemini memakai prompt lengkap dan 3 potongan konteks.
    pakai_prompt_ringkas = any(k in target_model.lower() for k in ("gpt-oss", "llama"))
    rag_limit = 1 if pakai_prompt_ringkas else 3
    context = get_rag_context(rag_query, limit=rag_limit)
    
    system_prompt = SYSTEM_PROMPT_NARASI
    user_prompt = build_user_prompt(
        input_data.category, input_data.indicator, input_data.subject, context, data_table
    )
    
    # Tidak ada peralihan ke MODEL LAIN. Satu-satunya fallback adalah Groq -> Hugging Face untuk model
    # yang sama (gpt-oss-120b), ditangani di call_narrative_provider dan
    # tercatat di used_model. Selain itu, error asli dikembalikan ke user apa adanya.
    try:
        sisa_awal = NARRATIVE_BUDGET_SECONDS - (time.monotonic() - mulai)
        narrative, final_model = panggil_dengan_batas_waktu(
            sisa_awal, call_narrative_provider,
            target_model, system_prompt, user_prompt, context, rag_query,
            input_data.category, input_data.subject, input_data.indicator, data_table
        )
    except Exception as e:
        error_msg = f"Model {target_model} gagal memproses data. Detail: {str(e)}"
        print(f"❌ {error_msg}", flush=True)

        return {
            "status": "error",
            "error_message": error_msg,
            "used_model": target_model,
            "rag_preview": context[:300] + "..." if context else ""
        }

    cleaned_narrative = strip_analysis_tag(narrative)
    raw_terakhir = narrative  # disimpan untuk log diagnosis kalau narasi gagal
    durasi_percobaan_1 = time.monotonic() - mulai
    print(f"⏱️ Percobaan pertama ({target_model}) selesai dalam {durasi_percobaan_1:.1f} detik.", flush=True)

    # --- SAFETY NET: kalau hasil ternyata cuma bocoran langkah_analisis (model lupa
    # pakai tag, atau generasi terpotong sebelum sempat menulis narasi), coba ULANG
    # SEKALI dengan MODEL YANG SAMA (target_model) plus pengingat lebih tegas — BUKAN
    # pindah ke provider/model lain. Ini tidak menambah latensi pada jalur normal,
    # hanya jalan kalau percobaan pertama memang menghasilkan output yang rusak.
    # Retry hanya kalau sisa jatah waktu cukup (percobaan kedua kira-kira selama percobaan
    # pertama); kalau tidak, langsung pesan gagal daripada keburu diputus Laravel.
    sisa_waktu = NARRATIVE_BUDGET_SECONDS - (time.monotonic() - mulai)
    if looks_like_leaked_analysis(cleaned_narrative) and sisa_waktu < durasi_percobaan_1 * 1.2:
        print(f"⏭️ [RETRY DILEWATI] Sisa waktu {sisa_waktu:.0f} detik tidak cukup untuk percobaan kedua "
              f"(perkiraan ~{durasi_percobaan_1:.0f} detik).", flush=True)
    elif looks_like_leaked_analysis(cleaned_narrative):
        print(f"⚠️ [RETRY] Output dari {target_model} hanya berisi langkah analisis (kemungkinan terpotong/tag hilang). Mengulang sekali dengan model yang sama...", flush=True)
        try:
            retry_narrative, _ = panggil_dengan_batas_waktu(
                sisa_waktu, call_narrative_provider,
                target_model, system_prompt, user_prompt, context, rag_query,
                input_data.category, input_data.subject, input_data.indicator, data_table,
                extra_instruction=(
                    "\n\nPERHATIAN: Percobaan sebelumnya GAGAL karena hanya menghasilkan langkah "
                    "analisis tanpa narasi akhir. WAJIB tutup tag </langkah_analisis> lebih awal dan "
                    "lebih singkat, lalu LANGSUNG tulis narasi 3 paragraf lengkap sesudahnya."
                )
            )
            raw_terakhir = retry_narrative
            retry_cleaned = strip_analysis_tag(retry_narrative)
            if retry_cleaned and not looks_like_leaked_analysis(retry_cleaned):
                cleaned_narrative = retry_cleaned
                final_model = f"{final_model} [Retry]"
                print(f"✅ [RETRY] Berhasil menghasilkan narasi lengkap pada percobaan kedua (model sama: {target_model}).", flush=True)
        except Exception as retry_error:
            print(f"⚠️ [RETRY] Percobaan ulang gagal: {retry_error}", flush=True)

    if not cleaned_narrative or looks_like_leaked_analysis(cleaned_narrative):
        # Kembalikan sebagai ERROR (bukan success) supaya teks peringatan tidak pernah
        # bisa tersimpan ke database sebagai narasi dan ikut ke dashboard/ekspor PDF-Excel.
        error_msg = (
            f"Model {target_model} tidak menghasilkan narasi lengkap (kemungkinan terpotong karena "
            "batas token atau data terlalu panjang). Coba gunakan model lain atau kurangi jumlah baris data."
        )
        print(f"❌ {error_msg}", flush=True)
        # Log diagnosis sementara: tampilkan output mentah model untuk mencari penyebab kegagalan.
        print("----- OUTPUT MENTAH MODEL (maks. 1500 karakter) -----\n" + (raw_terakhir or "")[:1500]
              + "\n----- AKHIR OUTPUT MENTAH -----", flush=True)
        return {
            "status": "error",
            "error_message": error_msg,
            "used_model": final_model,
            "rag_preview": context[:300] + "..." if context else ""
        }

    return {
        "status": "success",
        "narrative_result": cleaned_narrative,
        "used_model": final_model,
        "rag_preview": context[:300] + "..."
    }
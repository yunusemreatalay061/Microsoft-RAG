import json
import sqlite3
import math
import time
import hashlib
import io
import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None

st.set_page_config(page_title="RAG Asistanı", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    #MainMenu, footer, header { visibility: hidden; }
    section[data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    .stTextArea textarea, .stSelectbox select { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; }
    .stButton>button { background-color: #2563eb; color: white; border: none; font-weight: bold; }
    .stDownloadButton>button { background-color: #059669; color: white; }
    </style>
""", unsafe_allow_html=True)



def veritabani_baglan():
    """
    Her bağlantıda WAL modu + senkron=NORMAL ayarlanır.
    WAL modu, yazma işlemlerinin okuma işlemlerini bloklamamasını sağlar ve disk
    I/O beklemesini azaltarak toplu insert işlemlerini belirgin şekilde hızlandırır.
    """
    conn = sqlite3.connect("rag_database.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def veritabani_ilklendir():
    conn = veritabani_baglan()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(documents)")
    columns = [col[1] for col in cursor.fetchall()]
    if "source" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN source TEXT DEFAULT 'Genel Metin'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            file_hash TEXT PRIMARY KEY,
            source TEXT,
            chunk_count INTEGER,
            created_at REAL
        )
    """)
    conn.commit()
    conn.close()

veritabani_ilklendir()


@st.cache_resource
def sistem_modellerini_yukle():
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="rag-app"))

    katalog = FoundryLocalManager.instance.catalog

    def modeli_getir(anahtar_kelime):
        for m in katalog.list_models():
            model_ismi = m.id if hasattr(m, 'id') else m.name
            if anahtar_kelime in model_ismi:
                return m
        return None

    embed_model = modeli_getir("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model = modeli_getir("qwen2.5-7b-instruct")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embed_client, chat_client

with st.spinner("Modeller yükleniyor..."):
    embed_client, chat_client = sistem_modellerini_yukle()


def metni_parcalara_bol(metin, parca_boyutu=1000, ortusme=100):
    if not metin or not metin.strip():
        return []

    metin = " ".join(metin.split())
    if len(metin) <= parca_boyutu:
        return [metin.strip()]

    parcalar = []
    baslangic = 0
    metin_uzunlugu = len(metin)

    while baslangic < metin_uzunlugu:
        bitis = min(baslangic + parca_boyutu, metin_uzunlugu)
        if bitis < metin_uzunlugu:
            son_bosluk = metin.rfind(' ', baslangic, bitis)
            if son_bosluk > baslangic:
                bitis = son_bosluk

        parca = metin[baslangic:bitis].strip()
        if parca:
            parcalar.append(parca)

        baslangic = bitis - ortusme if bitis < metin_uzunlugu else metin_uzunlugu

    return parcalar


def kosinus_benzerligi(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    try:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        m1 = math.sqrt(sum(a * a for a in v1))
        m2 = math.sqrt(sum(b * b for b in v2))
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot_product / (m1 * m2)
    except Exception:
        return 0.0


def veritabanindan_bilgi_getir(soru, secili_kaynak="Tüm Kaynaklar", top_k=3):
    try:
        soru_vektoru = embed_client.generate_embeddings([soru]).data[0].embedding
    except Exception:
        return ""

    conn = veritabani_baglan()
    cursor = conn.cursor()

    if secili_kaynak == "Tüm Kaynaklar":
        cursor.execute("SELECT content, embedding, source FROM documents")
    else:
        cursor.execute("SELECT content, embedding, source FROM documents WHERE source = ?", (secili_kaynak,))

    kayitlar = cursor.fetchall()
    conn.close()

    skorlu_kayitlar = []
    for kayit in kayitlar:
        metin, db_vektor_str, kaynak = kayit[0], kayit[1], kayit[2]
        try:
            db_vektoru = json.loads(db_vektor_str)
            skor = kosinus_benzerligi(soru_vektoru, db_vektoru)
            skorlu_kayitlar.append((skor, metin, kaynak))
        except Exception:
            continue

    skorlu_kayitlar.sort(key=lambda x: x[0], reverse=True)
    en_iyi_sonuclar = [k for k in skorlu_kayitlar[:top_k] if k[0] > 0.15]

    contextual_text = ""
    for idx, (skor, metin, kaynak) in enumerate(en_iyi_sonuclar, 1):
        contextual_text += f"[Kaynak: {kaynak}]\n{metin}\n---\n"

    return contextual_text.strip()


def kaynaklari_getir():
    conn = veritabani_baglan()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source FROM documents")
    kaynaklar = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return kaynaklar


def dosya_hashini_hesapla(dosya_bytes):
    """Aynı içeriğin (dosya adı farklı olsa bile) tekrar işlenmesini engellemek için hash."""
    return hashlib.sha256(dosya_bytes).hexdigest()


def daha_once_islendi_mi(dosya_hash):
    conn = veritabani_baglan()
    cursor = conn.cursor()
    cursor.execute("SELECT source, chunk_count FROM processed_files WHERE file_hash = ?", (dosya_hash,))
    sonuc = cursor.fetchone()
    conn.close()
    return sonuc  # None ya da (source, chunk_count)


def pdfden_metin_cikar(dosya_bytes):
    """
    PyMuPDF (fitz) varsa onunla, yoksa pypdf ile metin çıkarır.
    PyMuPDF, özellikle çok sayfalı PDF'lerde belirgin şekilde daha hızlı ve daha az
    CPU yükü oluşturur; bu da hem süreyi kısaltır hem de ısınmayı azaltır.
    """
    metin_parcalari = []
    if fitz is not None:
        with fitz.open(stream=dosya_bytes, filetype="pdf") as pdf:
            for page in pdf:
                t = page.get_text()
                if t:
                    metin_parcalari.append(t)
    elif pypdf is not None:
        reader = pypdf.PdfReader(io.BytesIO(dosya_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                metin_parcalari.append(t)
    else:
        return None
    return "\n".join(metin_parcalari)


# 5. SOL PANEL - DOSYA YÜKLEME VE KAYNAK SEÇİMİ
with st.sidebar:
    st.title("📚 Kaynak Yönetimi")

    st.subheader("1. Dosya Yükle (PDF / TXT)")
    yuklenen_dosya = st.file_uploader("Dosyanızı buraya bırakın:", type=["pdf", "txt"])

    if yuklenen_dosya is not None:
        dosya_adi = yuklenen_dosya.name
        dosya_bytes = yuklenen_dosya.getvalue()
        dosya_hash = dosya_hashini_hesapla(dosya_bytes)

        onceki_kayit = daha_once_islendi_mi(dosya_hash)
        if onceki_kayit is not None:
            onceki_kaynak, onceki_parca_sayisi = onceki_kayit
            st.info(
                f"Bu dosya içeriği daha önce '{onceki_kaynak}' olarak işlenmiş "
                f"({onceki_parca_sayisi} parça). Tekrar embed edilmeyecek."
            )
            zorla_yeniden_isle = st.checkbox("Yine de yeniden işle (mevcut kaydı silip baştan işle)")
        else:
            zorla_yeniden_isle = False

        islenmeye_hazir = (onceki_kayit is None) or zorla_yeniden_isle

        if islenmeye_hazir and st.button(f"'{dosya_adi}' Dosyasını İşle ve Kaydet", use_container_width=True):
            metin = ""
            if yuklenen_dosya.type == "text/plain":
                metin = dosya_bytes.decode("utf-8")
            elif yuklenen_dosya.type == "application/pdf":
                cikan = pdfden_metin_cikar(dosya_bytes)
                if cikan is None:
                    st.error("PDF okumak için 'pip install pymupdf' (önerilen) veya 'pip install pypdf' çalıştırmalısınız.")
                else:
                    metin = cikan

            if metin.strip():
                parcalar = metni_parcalara_bol(metin)
                conn = veritabani_baglan()
                cursor = conn.cursor()

                # Yeniden işleniyorsa, önce eski kayıtları temizle
                if zorla_yeniden_isle and onceki_kayit is not None:
                    cursor.execute("DELETE FROM documents WHERE source = ?", (onceki_kayit[0],))
                    cursor.execute("DELETE FROM processed_files WHERE file_hash = ?", (dosya_hash,))
                    conn.commit()

                ilerleme = st.progress(0, text=f"0/{len(parcalar)} parça işlendi")

                # Batch boyutu 8'den 16'ya çıkarıldı: daha az sayıda embedding çağrısı,
                # daha az ek yük (overhead) demek -> daha kısa toplam süre.
                batch_size = 16
                basarili = True
                for i in range(0, len(parcalar), batch_size):
                    grup = parcalar[i:i + batch_size]
                    t0 = time.time()
                    try:
                        vektorler = embed_client.generate_embeddings(grup).data
                        cursor.executemany(
                            "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
                            [
                                (dosya_adi, parca, json.dumps(vektor_obj.embedding))
                                for parca, vektor_obj in zip(grup, vektorler)
                            ]
                        )
                        conn.commit()
                    except Exception as e:
                        st.error(f"Paket işlenirken hata: {e}")
                        basarili = False
                        break

                    islenen = min(i + batch_size, len(parcalar))
                    ilerleme.progress(islenen / len(parcalar), text=f"{islenen}/{len(parcalar)} parça işlendi")

                    # Akıllı/adaptif soğuma: batch zaten "yavaş" işlendiyse (donanım meşgul
                    # ya da model işlemi doğal olarak uzun sürdüyse) ekstra bekleme EKLENMEZ.
                    # Sadece batch beklenenden hızlı bittiyse donanıma kısa bir nefes payı
                    # verilir. Bu, sabit sleep(0.05) yerine hem toplam süreyi kısaltır hem
                    # de gerçekten gerektiğinde soğuma sağlar.
                    islem_suresi = time.time() - t0
                    if islem_suresi < 0.3:
                        time.sleep(0.03)

                if basarili:
                    cursor.execute(
                        "INSERT OR REPLACE INTO processed_files (file_hash, source, chunk_count, created_at) VALUES (?, ?, ?, ?)",
                        (dosya_hash, dosya_adi, len(parcalar), time.time())
                    )
                    conn.commit()

                conn.close()
                if basarili:
                    st.success(f"'{dosya_adi}' başarıyla veritabanına eklendi! ({len(parcalar)} parça)")
                    st.rerun()

    st.divider()

    st.subheader("2. Soru Sorulacak Kaynak")
    mevcut_kaynaklar = kaynaklari_getir()
    secili_kaynak = st.selectbox(
        "Aramanın yapılacağı dokümanı seçin:",
        options=["Tüm Kaynaklar"] + mevcut_kaynaklar
    )

    if mevcut_kaynaklar:
        with st.expander("🗑️ Kaynak Sil / Temizle"):
            silinecek_kaynak = st.selectbox("Silinecek Kaynak:", options=mevcut_kaynaklar, key="sil_box")
            if st.button("Kaynağı Veritabanından Sil"):
                conn = veritabani_baglan()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE source = ?", (silinecek_kaynak,))
                cursor.execute("DELETE FROM processed_files WHERE source = ?", (silinecek_kaynak,))
                conn.commit()
                conn.close()
                st.success(f"'{silinecek_kaynak}' silindi.")
                st.rerun()

    st.divider()
    if st.button("Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 6. ANA EKRAN - SOHBET
st.title("🤖 Yerel RAG Asistanı")
st.caption(f"Aktif Odak: **{secili_kaynak}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if kullanici_sorusu := st.chat_input("Seçili kaynağa göre soru sorun..."):
    st.chat_message("user").markdown(kullanici_sorusu)
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})

    with st.chat_message("assistant"):
        bulunan_bilgi = veritabanindan_bilgi_getir(kullanici_sorusu, secili_kaynak=secili_kaynak)

        sistem_talimati = f"""Sen verilen 'Veritabanı Bilgisi' metinlerini temel alarak soruları yanıtlayan analitik bir asistansın.

GÖREVİN VE KURALLARIN:
1. Kullanıcının sorusunu, SADECE verilen 'Veritabanı Bilgisi' içindeki gerçekler, kurallar ve bağlam doğrultusunda yanıtla.
2. Metindeki bilgileri birleştirerek MANTIKSAL ÇIKARIMLAR ve YORUMLAR yapabilirsin; ancak bu çıkarımlar MUTLAKA verilen metindeki bilgilere dayanmalıdır.
3. Verilen metinle doğrudan veya dolaylı olarak bağlantısı olmayan, dokümanda hiç geçmeyen dış dünyadan veya genel kültürden BİLGİ UYDURMA.
4. Eğer soru verilen metindeki hiçbir bilgi veya mantıksal bağlamla yanıtlanamıyorsa, dürüstçe "Aradığınız bilgi veya çıkarım yüklü dokümanlarda bulunmamaktadır." de.

Veritabanı Bilgisi:
{bulunan_bilgi}"""

        prompt_mesajlari = [
            {"role": "system", "content": sistem_talimati},
            {"role": "user", "content": kullanici_sorusu}
        ]

        yanit = chat_client.complete_chat(messages=prompt_mesajlari)
        cevap_metni = yanit.choices[0].message.content

        def kelime_kelime_akit(metin):
            for kelime in metin.split(" "):
                yield kelime + " "
                time.sleep(0.02)

        akici_cevap = st.write_stream(kelime_kelime_akit(cevap_metni))

    st.session_state.messages.append({"role": "assistant", "content": akici_cevap})
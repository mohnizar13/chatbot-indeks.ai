import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

# ========== KONFIGURASI ==========
st.set_page_config(
    page_title="Indeks AI - Asisten Pasar Modal",
    page_icon="📈",
    layout="wide"
)

# Load API Key dari secrets
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error("⚠️ API Key Gemini tidak ditemukan. Pastikan GEMINI_API_KEY sudah diset di st.secrets.")
    st.stop()

# Model LLM dengan LangChain
MODEL_NAME = "gemini-2.5-flash"

# System Instruction untuk Persona
SYSTEM_INSTRUCTION = """Anda adalah Indeks AI, seorang analis Pasar Modal Indonesia yang profesional, objektif, dan edukatif. 

Tugas Anda:
- Berikan jawaban yang ringkas, berdasarkan fakta, dan fokus pada IHSG, emiten Indonesia, dan ekonomi makro Indonesia
- Jelaskan istilah teknis dengan bahasa yang mudah dipahami untuk investor pemula hingga menengah
- Selalu objektif dan tidak memberikan rekomendasi beli/jual saham secara spesifik
- Gunakan Bahasa Indonesia yang profesional namun ramah
- Jika tidak yakin, akui keterbatasan dan sarankan konsultasi dengan profesional berlisensi

Konteks: Anda membantu investor Indonesia memahami pasar modal dalam negeri dengan lebih baik."""

# ========== INISIALISASI LLM ==========
@st.cache_resource
def init_llm():
    """Inisialisasi LangChain LLM dengan caching"""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_output_tokens=2048,
        convert_system_message_to_human=True
    )

llm = init_llm()

# ========== FUNGSI CACHING DATA ==========
@st.cache_data(ttl=300)  # Cache 5 menit
def fetch_ihsg_data():
    """Mengambil data IHSG terkini dari Yahoo Finance"""
    try:
        ticker = yf.Ticker("^JKSE")
        hist = ticker.history(period="5d")
        
        if hist.empty or len(hist) < 2:
            return None
        
        # Data hari terakhir dan sebelumnya
        latest_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        
        # Perhitungan perubahan
        change_points = latest_close - prev_close
        change_percent = (change_points / prev_close) * 100
        
        # Timestamp data
        last_date = hist.index[-1]
        last_date_str = last_date.strftime('%d %B %Y')
        day_name = last_date.strftime('%A')
        
        # Cek apakah data hari ini atau kemarin
        now = datetime.now()
        is_today = last_date.date() == now.date()
        days_ago = (now.date() - last_date.date()).days
        
        # Informasi tambahan tentang status pasar
        market_status = "hari ini" if is_today else f"{days_ago} hari lalu"
        
        return {
            "close": latest_close,
            "prev_close": prev_close,
            "change_points": change_points,
            "change_percent": change_percent,
            "date": last_date_str,
            "day_name": day_name,
            "status": "naik" if change_points > 0 else "turun" if change_points < 0 else "stagnan",
            "is_today": is_today,
            "days_ago": days_ago,
            "market_status": market_status,
            "timestamp": last_date
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)  # Cache 5 menit
def fetch_ihsg_weekly_data():
    """Mengambil data IHSG seminggu terakhir dari Yahoo Finance"""
    try:
        ticker = yf.Ticker("^JKSE")
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            return None
        
        # Ambil 7 hari trading terakhir
        hist = hist.tail(7)
        
        # Format data untuk tampilan
        weekly_data = []
        for idx, row in hist.iterrows():
            date_str = idx.strftime('%d %b %Y')
            day_name = idx.strftime('%A')
            
            # Hitung perubahan dari hari sebelumnya
            if len(weekly_data) > 0:
                prev_close = weekly_data[-1]['close']
                change = row['Close'] - prev_close
                change_pct = (change / prev_close) * 100
            else:
                change = 0
                change_pct = 0
            
            weekly_data.append({
                'date': date_str,
                'day': day_name,
                'close': row['Close'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row['Volume'],
                'change': change,
                'change_pct': change_pct
            })
        
        return weekly_data
    except Exception as e:
        return {"error": str(e)}

# ========== FUNGSI DETEKSI KATA KUNCI ==========
def is_ihsg_data_request(prompt):
    """Deteksi cerdas apakah prompt meminta data IHSG"""
    prompt_lower = prompt.lower()
    
    # Kata-kata yang menandakan pertanyaan edukasi
    exclusion_words = [
        'apa itu', 'jelaskan', 'definisi', 'maksudnya', 'pengertian',
        'bagaimana cara', 'mengapa', 'kenapa', 'perbedaan'
    ]
    if any(word in prompt_lower for word in exclusion_words):
        return False

    # Kata kunci permintaan data
    data_keywords = [
        "ihsg", "indeks harga saham gabungan", "jkse", 
        "bursa hari ini", "posisi ihsg", "berapa ihsg",
        "harga ihsg", "idx", "indeks komposit", "bursa saham indonesia",
        "saham hari ini", "pasar modal hari ini", "update ihsg",
        "pergerakan ihsg", "closing ihsg", "pembukaan ihsg", "berita ihsg"
    ]
    return any(keyword in prompt_lower for keyword in data_keywords)

def is_ihsg_weekly_request(prompt):
    """Deteksi apakah prompt meminta data mingguan"""
    prompt_lower = prompt.lower()
    weekly_keywords = [
        "seminggu", "minggu ini", "7 hari", "mingguan", "weekly",
        "seminggu terakhir", "data seminggu", "tren minggu",
        "pergerakan seminggu", "grafik minggu", "chart minggu"
    ]
    return any(keyword in prompt_lower for keyword in weekly_keywords)

# ========== FUNGSI MODE HIBRIDA WEEKLY ==========
def get_ihsg_weekly_data_and_format():
    """Mode Hibrida Weekly: Ambil data IHSG seminggu terakhir"""
    weekly_data = fetch_ihsg_weekly_data()
    
    # Jika error
    if weekly_data is None or (isinstance(weekly_data, dict) and "error" in weekly_data):
        error_msg = weekly_data.get("error", "Tidak dapat mengambil data") if isinstance(weekly_data, dict) else "Data tidak tersedia"
        return f"⚠️ Maaf, tidak dapat mengambil data mingguan IHSG dari Yahoo Finance. Error: {error_msg}"
    
    # Buat DataFrame
    df = pd.DataFrame(weekly_data)
    df['close_fmt'] = df['close'].apply(lambda x: f"{x:,.2f}")
    df['change_fmt'] = df.apply(lambda row: f"{row['change']:+,.2f} ({row['change_pct']:+.2f}%)" if row['change'] != 0 else "-", axis=1)
    df['volume_fmt'] = df['volume'].apply(lambda x: f"{x:,.0f}")
    
    display_df = df[['date', 'day', 'close_fmt', 'change_fmt', 'volume_fmt']].copy()
    display_df.columns = ['Tanggal', 'Hari', 'Penutupan', 'Perubahan', 'Volume']
    
    # Statistik
    latest = weekly_data[-1]
    oldest = weekly_data[0]
    week_change = latest['close'] - oldest['close']
    week_change_pct = (week_change / oldest['close']) * 100
    highest = max([d['high'] for d in weekly_data])
    lowest = min([d['low'] for d in weekly_data])
    avg_close = sum([d['close'] for d in weekly_data]) / len(weekly_data)
    
    stats_text = f"""
📊 **Ringkasan Data IHSG 7 Hari Trading Terakhir**

**Performa Mingguan:**
- Penutupan Awal: {oldest['close']:,.2f} ({oldest['date']})
- Penutupan Akhir: {latest['close']:,.2f} ({latest['date']})
- Perubahan Minggu: {week_change:+,.2f} poin ({week_change_pct:+.2f}%)
- Status: {"📈 MENGUAT" if week_change > 0 else "📉 MELEMAH" if week_change < 0 else "➡️ STAGNAN"}

**Statistik:**
- Tertinggi: {highest:,.2f}
- Terendah: {lowest:,.2f}
- Rata-rata: {avg_close:,.2f}
- Range: {highest - lowest:,.2f} poin

---
"""
    
    # AI Analysis
    summary_for_ai = f"""Data IHSG 7 hari trading terakhir:
- Periode: {oldest['date']} sampai {latest['date']}
- Penutupan awal: {oldest['close']:.2f}
- Penutupan akhir: {latest['close']:.2f}
- Perubahan: {week_change:+.2f} poin ({week_change_pct:+.2f}%)
- Tertinggi: {highest:.2f}, Terendah: {lowest:.2f}

Detail harian:
{chr(10).join([f"- {d['date']} ({d['day']}): {d['close']:.2f} ({d['change']:+.2f} / {d['change_pct']:+.2f}%)" for d in weekly_data])}"""
    
    weekly_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"""Sebagai analis pasar, berikan analisis singkat (2-3 paragraf) tentang pergerakan IHSG seminggu terakhir:

{summary_for_ai}

Format:
1. Gambaran umum performa mingguan (naik/turun berapa persen)
2. Highlight hari dengan pergerakan signifikan
3. Observasi pola atau tren (jika ada)

PENTING: Tetap objektif, jangan buat prediksi atau rekomendasi. Fokus pada fakta data.""")
    ])
    
    try:
        ai_analysis = llm.invoke(weekly_prompt.format_messages()).content
    except:
        ai_analysis = "Analisis AI tidak tersedia saat ini."
    
    result = f"""{ai_analysis}

{stats_text}

**📋 Tabel Data Lengkap:**
"""
    
    return result, display_df

# ========== FUNGSI MODE HIBRIDA ==========
def get_ihsg_data_and_format():
    """Mode Hibrida: Ambil data IHSG dan format dengan LLM"""
    data = fetch_ihsg_data()
    
    if data is None or "error" in data:
        error_msg = data.get("error", "Tidak dapat mengambil data") if data else "Data tidak tersedia"
        
        error_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=f"""Data IHSG dari sumber eksternal gagal diambil dengan error: {error_msg}. 
        
Berikan respons yang sopan kepada pengguna bahwa data pasar real-time sedang tidak tersedia dan sarankan untuk:
1. Coba beberapa saat lagi
2. Cek langsung di website BEI atau aplikasi trading
3. Tetap bisa bertanya hal lain terkait edukasi pasar modal

PENTING: Berikan respons LENGKAP dalam satu output, jangan terpotong.""")
        ])
        
        try:
            response = llm.invoke(error_prompt.format_messages())
            return response.content
        except Exception as e:
            return "⚠️ Maaf, sistem sedang mengalami kendala dalam mengakses data pasar. Silakan coba beberapa saat lagi atau tanyakan hal lain terkait edukasi investasi."
    
    data_text = f"""Data IHSG dari Yahoo Finance:
- Tanggal Data: {data['day_name']}, {data['date']}
- Status Pasar: {"Data hari ini (real-time)" if data['is_today'] else f"Data {data['days_ago']} hari lalu (bursa tutup hari ini)"}
- Penutupan: {data['close']:.2f} poin
- Perubahan: {data['change_points']:+.2f} poin ({data['change_percent']:+.2f}%)
- Status: {data['status'].upper()}
- Penutupan sebelumnya: {data['prev_close']:.2f} poin

Catatan: Data ini REAL dari Yahoo Finance ticker ^JKSE"""
    
    hybrid_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"""Bertindak sebagai penyiar berita pasar profesional, sajikan data IHSG berikut dalam format yang jelas:

{data_text}

Format respons:
1. Mulai dengan menyebutkan SUMBER DATA (Yahoo Finance) dan KAPAN data tersebut
2. Jika data bukan hari ini, jelaskan MENGAPA (bursa tutup di weekend/libur)
3. Sebutkan posisi penutupan IHSG dengan jelas
4. Jelaskan pergerakan (naik/turun) dengan angka pasti
5. Gunakan bahasa yang natural dan informatif

CONTOH FORMAT YANG BAIK:
"Berdasarkan data dari Yahoo Finance, IHSG pada hari [HARI], [TANGGAL] ditutup di level [ANGKA] poin. [Jika bukan hari ini: "Data ini merupakan penutupan terakhir karena bursa saham tidak beroperasi pada hari Sabtu/Minggu/libur"]. Indeks mengalami [NAIK/TURUN] sebesar [ANGKA] poin atau [PERSEN]% dari penutupan sebelumnya di [ANGKA] poin."

PENTING: 
- Sebutkan dengan jelas bahwa data dari Yahoo Finance
- Jika data bukan hari ini, WAJIB menjelaskan alasannya
- Berikan respons LENGKAP dalam satu output""")
    ])
    
    try:
        response = llm.invoke(hybrid_prompt.format_messages())
        
        data_detail = f"""

---
**📊 Data Mentah dari Yahoo Finance (^JKSE):**
- 📅 Tanggal: {data['day_name']}, {data['date']}
- 💰 Penutupan: **{data['close']:.2f}** poin
- 📈 Perubahan: **{data['change_points']:+.2f}** poin (**{data['change_percent']:+.2f}%**)
- 🔄 Status: **{data['status'].upper()}**
- ⏰ Data diambil: {data['market_status']}
"""
        
        return response.content + data_detail
    except Exception as e:
        return f"⚠️ Data IHSG: {data['close']:.2f} ({data['change_percent']:+.2f}%). Sistem AI sedang sibuk, data mentah ditampilkan."

# ========== FUNGSI MODE LLM MURNI ==========
def get_llm_response(prompt, chat_history):
    """Mode LLM Murni: Respons edukasi dan umum"""
    try:
        messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
        
        recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
        
        for msg in recent_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                content = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
                messages.append(AIMessage(content=content))
        
        full_prompt = f"""{prompt}

PENTING: Berikan jawaban yang LENGKAP dan TUNTAS dalam satu respons. Jangan berhenti di tengah kalimat."""
        
        messages.append(HumanMessage(content=full_prompt))
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"⚠️ Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba dengan pertanyaan yang lebih ringkas atau coba lagi."

# ========== INISIALISASI SESSION STATE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
    greeting = """👋 Selamat datang di **Indeks AI**!

Saya adalah asisten virtual Anda untuk memahami Pasar Modal Indonesia. Saya dapat membantu Anda dengan:

📊 **Informasi IHSG**: Posisi IHSG terkini, pergerakan harian  
📚 **Edukasi Investasi**: Penjelasan istilah, konsep, dan strategi pasar modal  
🎯 **Analisis Objektif**: Informasi berbasis data untuk keputusan investasi yang lebih baik

**Contoh pertanyaan:**
- "Apa itu IHSG?"
- "Berita IHSG hari ini?"
- "Data IHSG seminggu terakhir?"
- "Jelaskan perbedaan saham dan obligasi"
- "Bagaimana cara membaca laporan keuangan?"

Silakan ajukan pertanyaan Anda! 🚀

---
*Powered by LangChain + Google Gemini 2.5 Flash*"""
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": greeting
    })

# ========== UI STREAMLIT ==========

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #eee;
    }
    .tech-badge {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.8rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Indeks AI</h1>
    <p>Asisten Analisis Pasar Modal Indonesia</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Tentang Indeks AI")
    st.markdown("""
    **Indeks AI** adalah chatbot AI yang dirancang untuk membantu investor pemula memahami pasar modal Indonesia dengan lebih baik.
    
    **Fitur:**
    - 🔴 Informasi Data IHSG 
    - 📖 Edukasi Investasi
    - 🤖 Powered by Gemini 2.5 Flash
    - 🔗 Arsitektur LangChain
    - 📊 Data dari Yahoo Finance
    
    **Tech Stack:**
    """)
    
    st.markdown("""
    <div style="margin-top: 0.5rem;">
        <span class="tech-badge">🦜 LangChain</span>
        <span class="tech-badge">⚡ Gemini 2.5 Flash</span>
        <span class="tech-badge">📊 yfinance</span>
        <span class="tech-badge">🎨 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    **Disclaimer:**  
    Informasi yang diberikan bersifat edukatif dan bukan rekomendasi investasi. Selalu lakukan riset mandiri dan konsultasi dengan profesional berlisensi.
    """)
    
    st.divider()
    
    # Statistik chat
    if len(st.session_state.messages) > 1:
        user_msgs = len([m for m in st.session_state.messages if m['role'] == 'user'])
        st.metric("💬 Total Pertanyaan", user_msgs)
    
    st.divider()
    
    # Tombol clear chat
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("v2.2 - Clean UI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Tanyakan tentang IHSG, investasi, atau pasar modal Indonesia..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Menganalisis dengan Gemini 2.5 Flash..."):
            if is_ihsg_weekly_request(prompt):
                result = get_ihsg_weekly_data_and_format()
                if isinstance(result, tuple):
                    response, df = result
                    message_placeholder.markdown(response)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    weekly_data = fetch_ihsg_weekly_data()
                    if weekly_data:
                        st.line_chart(
                            data={d['date']: d['close'] for d in weekly_data},
                            use_container_width=True
                        )
                else:
                    response = result
                    message_placeholder.markdown(response)
                    
            elif is_ihsg_data_request(prompt):
                response = get_ihsg_data_and_format()
                message_placeholder.markdown(response)
            else:
                response = get_llm_response(prompt, st.session_state.messages)
                message_placeholder.markdown(response)
    
    if isinstance(result if 'result' in locals() else response, tuple):
        st.session_state.messages.append({"role": "assistant", "content": result[0]})
    else:
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
<div class="footer">
    <p>🦜 LangChain • ⚡ Google Gemini 2.5 Flash • 📊 Yahoo Finance • 🎨 Streamlit</p>
</div>
""", unsafe_allow_html=True)
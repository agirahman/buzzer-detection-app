import streamlit as st
import json
import pandas as pd
from buzzer_detector import BuzzerDetector

st.title("🕵️‍♂️ Deteksi Akun Buzzer Twitter")
st.write("Upload dataset JSON atau CSV menyah, lalu dapatkan daftar akun yang terindikasi sebagai buzzer!")

# Instructions
st.subheader("📋 Instruksi")
st.markdown("""
**Format Data yang Diharapkan:**
- **JSON:** Array of tweet objects dengan fields seperti `tweetId`, `userName`, `content`, `likeCount`, dll.
- **CSV:** Kolom dengan nama serupa.

**Contoh struktur JSON:**
```json
[
  {
    "tweetId": "123456",
    "userName": "john_doe",
    "content": "@jane_doe This is a reply tweet!",
    "likeCount": 10,
    "createdAt": "2023-01-01T00:00:00Z"
  }
]
```

**Pipeline Otomatis:**
1. **Preprocessing:** Bersihkan teks, ekstrak reply.
2. **Graph Analysis:** Bangun jaringan interaksi, hitung metrik SNA.
3. **Feature Engineering:** Hitung kesamaan narasi, frekuensi tweet.
4. **Labeling:** Terapkan aturan heuristik untuk label buzzer.
5. **Training:** Latih model XGBoost.
6. **Inference:** Prediksi probabilitas buzzer untuk semua akun.
""")

# Inisialisasi detector
detector = BuzzerDetector()

# Upload section
st.subheader("📤 Upload Dataset Mentah")
st.write("Upload file JSON atau CSV dengan kolom yang sesuai.")

uploaded_file = st.file_uploader("Pilih file JSON atau CSV", type=["json", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.json'):
            # Load JSON
            json_data = json.load(uploaded_file)
            st.success("✅ File JSON berhasil diupload!")
        elif uploaded_file.name.endswith('.csv'):
            # Load CSV and convert to expected format
            df_csv = pd.read_csv(uploaded_file)
            # Assume CSV has columns: tweetId, userName, content, etc.
            json_data = df_csv.to_dict('records')
            st.success("✅ File CSV berhasil diupload dan dikonversi!")

        # Process button
        if st.button("🔍 Proses Dataset & Deteksi Buzzer"):
            with st.spinner("Memproses dataset... Ini mungkin memakan waktu beberapa menit."):
                results = detector.process_dataset(json_data)

            if results.empty:
                st.error("❌ Dataset kosong atau tidak valid. Pastikan format data sesuai.")
            else:
                st.success("✅ Proses selesai! Berikut hasil deteksi buzzer:")

                # Display top results
                st.subheader("🎯 Top Akun Terindikasi Buzzer")
                top_results = results.head(20)
                st.dataframe(top_results[['username', 'buzzer_probability', 'is_buzzer', 'out_degree', 'narrative_similarity']].style.format({
                    'buzzer_probability': '{:.3f}',
                    'narrative_similarity': '{:.3f}'
                }))

                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil Lengkap (CSV)",
                    data=csv,
                    file_name="buzzer_detection_results.csv",
                    mime="text/csv"
                )

                # Summary stats
                st.subheader("📊 Ringkasan")
                total_accounts = len(results)
                buzzer_accounts = len(results[results['buzzer_probability'] > 0.5])
                st.metric("Total Akun Diproses", total_accounts)
                st.metric("Akun Terindikasi Buzzer (prob > 0.5)", buzzer_accounts)
                st.metric("Persentase Buzzer", f"{(buzzer_accounts/total_accounts*100):.1f}%")

    except Exception as e:
        st.error(f"❌ Error memproses file: {str(e)}")
        st.info("Pastikan file JSON memiliki struktur array tweet dengan fields: tweetId, userName, content, dll.")



st.markdown("---")
st.write("Dibuat dengan ❤️ untuk mendukung integritas ruang publik digital.")

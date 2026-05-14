# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi halaman Streamlit (judul, icon, layout)
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Judul utama aplikasi
st.title("🚢 Titanic Survival Predictor")
st.markdown("Masukkan data penumpang di bawah ini untuk melihat prediksi keselamatan.")

# --- Load Model yang Sudah Dilatih ---
@st.cache_resource
def load_model():
    # Pastikan file model.joblib ada di folder yang sama
    return joblib.load('model.joblib')

model = load_model()

# --- Membuat Form Input Data ---
with st.form("prediction_form"):
    st.subheader("Informasi Penumpang")
    
    # Membuat 2 kolom untuk tata letak yang lebih rapi
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Kelas Tiket (Pclass)
        pclass = st.selectbox(
            "Kelas Tiket (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1st (Upper)", 2: "2nd (Middle)", 3: "3rd (Lower)"}[x]
        )
        
        # 2. Jenis Kelamin (Sex)
        sex = st.selectbox(
            "Jenis Kelamin",
            options=["male", "female"]
        )
        
        # 3. Jumlah Saudara/Pasangan (SibSp)
        sibsp = st.number_input(
            "Jumlah Saudara/Pasangan (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1
        )
    
    with col2:
        # 4. Jumlah Orang Tua/Anak (Parch)
        parch = st.number_input(
            "Jumlah Orang Tua/Anak (Parch)",
            min_value=0,
            max_value=6,
            value=0,
            step=1
        )
        
        # 5. Tarif Tiket (Fare)
        fare = st.number_input(
            "Tarif Tiket (Fare)",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0,
            format="%.2f"
        )
        
        # 6. Pelabuhan Keberangkatan (Embarked)
        embarked = st.selectbox(
            "Pelabuhan Keberangkatan",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x]
        )
    
    # Tombol submit untuk melakukan prediksi
    submitted = st.form_submit_button("🔮 Prediksi Keselamatan")

# --- Logika Prediksi saat Tombol Ditekan ---
if submitted:
    # Membuat DataFrame dari input pengguna
    # Urutan fitur harus SAMA PERSIS dengan saat training di notebook
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])
    
    # Lakukan prediksi
    # Model yang disimpan adalah object RandomizedSearchCV, kita panggil .predict() dan .predict_proba()
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Probabilitas untuk kelas 'Survived' (1)
    survival_prob = prediction_proba[1] * 100
    
    # Tampilkan hasil prediksi
    st.divider()
    
    if prediction == 1:
        st.success(f"### Prediksi: SELAMAT! 🎉")
        st.markdown(f"#### Peluang Selamat: **{survival_prob:.2f}%**")
        st.balloons()
    else:
        st.error(f"### Prediksi: Tidak Selamat 😔")
        st.markdown(f"#### Peluang Selamat: **{survival_prob:.2f}%**")
        st.snow()
    
    # Menampilkan detail input untuk verifikasi
    with st.expander("Lihat Detail Data Penumpang"):
        st.json({
            "Pclass": pclass,
            "Sex": sex,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": f"${fare:.2f}",
            "Embarked": embarked
        })

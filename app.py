import streamlit as st
import joblib
import numpy as np

# Set custom page config
st.set_page_config(page_title="EMR Prediction", layout="centered")

# Custom CSS for background and footer
st.markdown("""
    <style>
        body {
            background-color: #f3e8ff; /* light purple */
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f3e8ff;
            color: black;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
        .footer img {
            height: 20px;
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('random_forest_model.pkl')

# Mapping dictionary
diagnosa_map = {
    28: "mata", 10: "dm", 20: "jantung", 14: "gerd", 19: "ispa", 8: "demam",
    40: "tht", 15: "gigi", 25: "kulit", 26: "lipoma", 39: "syaraf", 24: "kolesterol",
    9: "diare", 30: "pemeriksaan", 5: "batuk", 1: "arthritis", 42: "tonsilitis",
    2: "asam urat", 31: "pemeriksaan kandungan", 21: "kandungan", 6: "bedah",
    32: "pharyngitis", 11: "dyspepsia", 7: "dbd", 12: "fisioterapi", 27: "lutut",
    38: "sinusitis", 4: "batu empedu", 44: "vertigo", 13: "fungsi hati", 0: "alergi",
    16: "gondongan", 41: "tipes", 29: "paru", 33: "prostat", 43: "tulang",
    34: "reproduksi", 18: "isk", 17: "hepatitis", 37: "sendi", 23: "khitan",
    3: "bacterial inspection", 22: "keputihan", 36: "sariawan", 35: "sakit perut"
}

institusi_map = {
    3: "rumah sakit", 1: "laboratorium", 2: "poliklinik", 0: "dokter langganan"
}

area_map = {
    15: "sukabumi", 4: "cimahi", 6: "garut", 20: "yogyakarta", 1: "bandung",
    11: "palembang", 17: "surabaya", 19: "tasikmalaya", 13: "semarang", 16: "sumedang",
    18: "surakarta", 14: "serang", 2: "bekasi", 3: "bogor", 7: "jakarta", 0: "bandar lampung",
    5: "cirebon", 10: "malang", 8: "karawang", 9: "kediri", 12: "pekalongan"
}

tipe_map = {
    13: "spesialis mata", 0: "dokter umum", 12: "spesialis lain-lain", 20: "spesialis tht",
    10: "spesialis gigi dan mulut", 11: "spesialis kulit & kelamin", 5: "spesialis bedah ortopedi",
    17: "spesialis penyakit dalam", 9: "spesialis cardiologi", 14: "spesialis obsgyn",
    2: "spesialis anak", 3: "spesialis bedah", 4: "spesialis bedah digestive", 15: "spesialis paru",
    6: "spesialis bedah saraf", 19: "spesialis saraf (neurologi)", 8: "spesialis bedah urologi",
    16: "spesialis patologi", 7: "spesialis bedah umum", 18: "spesialis rhematologi",
    1: "spesialis ahli jiwa"
}

biaya_map = {
    1: "200001 s.d 500000", 0: "1000001 s.d 3000000", 3: "500001 s.d 1000000",
    4: "50001 s.d 200000", 2: "3000001 s.d 5000000", 6: "> 5000000", 5: "< 50000"
}

# Inverse maps
diagnosa_map_inv = {v: k for k, v in diagnosa_map.items()}
institusi_map_inv = {v: k for k, v in institusi_map.items()}
area_map_inv = {v: k for k, v in area_map.items()}
tipe_map_inv = {v: k for k, v in tipe_map.items()}
biaya_map_inv = {v: k for k, v in biaya_map.items()}

# Judul
st.title("Employee Segmentation Based on EMR Dataset using GridSearchCV-Random Forest Classifier")

# Input dropdown
diagnosa_input = st.selectbox("Choose Diagnose", list(diagnosa_map.values()))
institusi_input = st.selectbox("Choose Institution", list(institusi_map.values()))
area_input = st.selectbox("Choose Area", list(area_map.values()))
tipe_input = st.selectbox("Choose Type", list(tipe_map.values()))
biaya_input = st.selectbox("Choose Cost", list(biaya_map.values()))

# Encode input
data = np.array([[
    diagnosa_map_inv[diagnosa_input],
    institusi_map_inv[institusi_input],
    area_map_inv[area_input],
    tipe_map_inv[tipe_input],
    biaya_map_inv[biaya_input]
]])

# Prediksi
if st.button("Prediksi"):
    hasil = model.predict(data)[0]

    if hasil == 0:
        predict = "Economy-class healthcare services"
    elif hasil == 1:
        predict = "Complexs-class healthcare services"
    elif hasil == 2:
        predict = "Middle-class healthcare services"
    else:
        predict = "Unknown"

    st.success(f"Predict Result: {predict}")

# Footer dengan GitHub icon dan teks
st.markdown("""
    <div class="footer">
        <a href="https://github.com/ryanputranda/Prediksi-Layanan-Data-EMR-dengan-Random-Forest-dan-GridSearchCV" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
        </a>
        Ryan Portfolio
    </div>
""", unsafe_allow_html=True)

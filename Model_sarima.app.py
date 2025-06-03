import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Judul aplikasi
st.title("🌦️ Prediksi Cuaca Kota Bogor (SARIMA)")

# Navigasi
menu = st.sidebar.radio("📂 Menu", [
    "📄 Tampilkan Data",
    "📈 Grafik Historis",
    "🔮 Prediksi SARIMA"
])

# Load data
file_path = 'cuacabersih.xlsx'  # Pastikan file ini tersedia di direktori yang sama
df = pd.read_excel(file_path)

# Bersihkan nama kolom
df.columns = df.columns.str.strip()

# Ubah TANGGAL ke datetime dan set sebagai index
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
df.set_index('TANGGAL', inplace=True)

# Ubah nilai dengan koma ke titik dan konversi ke float
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Dropdown untuk memilih variabel yang ingin diprediksi
st.sidebar.markdown("### Pilih Variabel Cuaca:")
variable_options = {
    "Curah Hujan (mm)": "Curah hujan(mm)",
    "Temperatur Maksimum (°C)": "Temperatur maksimum(°C)",
    "Temperatur Rata-rata (°C)": "Temperatur rata-rata(°C)",
    "Kelembapan Rata-rata (%)": "Kelembapan rata-rata(%)"
}
selected_label = st.sidebar.selectbox("🧪 Variabel", list(variable_options.keys()))
selected_column = variable_options[selected_label]

# Series untuk prediksi
series = df[selected_column].dropna()

# Tampilkan Data
if menu == "📄 Tampilkan Data":
    st.subheader(f"📋 Data Historis: {selected_label}")
    st.dataframe(df[[selected_column]])

# Grafik Historis
elif menu == "📈 Grafik Historis":
    st.subheader(f"📊 Grafik Historis: {selected_label}")
    st.line_chart(series)

# Prediksi SARIMA
elif menu == "🔮 Prediksi SARIMA":
    st.subheader(f"📉 Prediksi {selected_label} dengan SARIMA")

    # Parameter ARIMA
    p = st.slider("p (AutoRegressive)", 0, 5, 1)
    d = st.slider("d (Differencing)", 0, 2, 1)
    q = st.slider("q (Moving Average)", 0, 5, 1)

    # Parameter Musiman
    P = st.slider("P Seasonal", 0, 3, 1)
    D = st.slider("D Seasonal", 0, 2, 1)
    Q = st.slider("Q Seasonal", 0, 3, 1)
    m = st.selectbox("Periode Musiman (m)", [7, 12, 30], index=1)

    # Jumlah prediksi
    n_days = st.slider("Jumlah hari ke depan", 1, 731, 30)

    # Model SARIMA
    with st.spinner("🔄 Melatih model SARIMA..."):
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_fit = model.fit(disp=False)

    # Prediksi
    forecast = model_fit.forecast(steps=n_days)
    forecast = forecast.clip(lower=0).round(2)
    forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=n_days)

    # DataFrame hasil
    forecast_df = pd.DataFrame({
        "Tanggal": forecast_dates,
        f"Prediksi {selected_label}": forecast.values
    }).set_index("Tanggal")

    # Tampilkan tabel
    st.subheader("📆 Hasil Prediksi")
    st.dataframe(forecast_df)

    # Grafik hasil
    st.subheader("📈 Grafik Prediksi")
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, label='Data Historis')
    forecast_df.plot(ax=ax, label='Prediksi', color='red')
    plt.title(f"Prediksi {selected_label}")
    plt.xlabel("Tanggal")
    plt.ylabel(selected_label)
    plt.legend()
    st.pyplot(fig)

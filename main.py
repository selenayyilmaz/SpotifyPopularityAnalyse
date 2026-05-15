import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import kagglehub
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

@st.cache_resource
def load_all_data():
    path = kagglehub.dataset_download("yamaerenay/spotify-dataset-19212020-600k-tracks")
    df = pd.read_csv(f"{path}/tracks.csv")

    features = ['danceability', 'energy', 'tempo', 'acousticness', 'loudness', 'valence']
    df_clean = df[features + ['popularity', 'name', 'artists', 'id']].dropna()

    X = df_clean[features]
    y = df_clean['popularity']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, df_clean


with st.spinner("Devasa veri seti analiz ediliyor..."):
    model, scaler, df_all = load_all_data()

st.title("🎵 Gelişmiş Spotify Popülerlik Analiz Paneli")

st.sidebar.header("🔎 Şarkı Seçim Yöntemi")
choice = st.sidebar.radio("Bir yöntem seçin:",
                          ["Şarkı Listesinden Seç", "Rastgele Şarkı Getir", "Kendi Değerlerimi Gireceğim"])

selected_song = None

if choice == "Şarkı Listesinden Seç":
    st.sidebar.subheader("Veri Setinden Ara")
    search_term = st.sidebar.text_input("Şarkı Adı Yazın:", "Bohemian Rhapsody")
    filtered_df = df_all[df_all['name'].str.contains(search_term, case=False, na=False)].head(10)

    if not filtered_df.empty:
        song_titles = filtered_df.apply(lambda x: f"{x['name']} - {x['artists']}", axis=1).tolist()
        song_select = st.sidebar.selectbox("Eşleşen Şarkılar:", song_titles)
        selected_song = filtered_df.iloc[song_titles.index(song_select)]
    else:
        st.sidebar.warning("Şarkı bulunamadı.")

elif choice == "Rastgele Şarkı Getir":
    if st.sidebar.button("Zarı At! (Yeni Şarkı Getir)"):
        selected_song = df_all.sample(1).iloc[0]
    else:
        st.sidebar.info("Butona basarak rastgele bir şarkı çekebilirsiniz.")

features_list = ['danceability', 'energy', 'tempo', 'acousticness', 'loudness', 'valence']

if selected_song is not None or choice == "Kendi Değerlerimi Gireceğim":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Şarkı Özellikleri")
        if choice != "Kendi Değerlerimi Gireceğim":
            st.markdown(f"**Seçilen Şarkı:** {selected_song['name']}")
            st.markdown(f"**Sanatçı:** {selected_song['artists']}")

            display_data = selected_song[features_list].to_dict()
            st.json(display_data)
        else:
            st.info("Yandaki panelden veya aşağıdan değerleri değiştirebilirsiniz.")
            d = st.slider("Danceability", 0.0, 1.0, 0.5)
            e = st.slider("Energy", 0.0, 1.0, 0.5)
            t = st.slider("Tempo", 50, 220, 120)
            a = st.slider("Acousticness", 0.0, 1.0, 0.2)
            l = st.slider("Loudness", -60, 0, -10)
            v = st.slider("Valence", 0.0, 1.0, 0.5)
            manual_vals = [d, e, t, a, l, v]

    with col2:
        st.subheader("🔮 Yapay Zeka Tahmini")

        if choice != "Kendi Değerlerimi Gireceğim":
            input_data = selected_song[features_list].values.reshape(1, -1)
            actual_pop = selected_song['popularity']
        else:
            input_data = np.array(manual_vals).reshape(1, -1)
            actual_pop = None

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.metric("Tahmin Edilen Popülerlik", f"{prediction:.2f} / 100")

        if actual_pop is not None:
            st.metric("Gerçek Popülerlik", f"{actual_pop}")
            hata = prediction - actual_pop
            st.write(f"**Model Yanılması:** {hata:.2f}")

            st.write("Tahmin Doğruluğu Görseli:")
            st.progress(min(int(prediction), 100))
        else:
            st.write("Kendi oluşturduğunuz şarkı için tahmin yapıldı.")
            st.progress(min(int(prediction), 100))

else:
    st.write("### 👈 Başlamak için soldaki panelden bir şarkı seçin veya aratın!")
    st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?auto=format&fit=crop&w=800&q=80")

#######################

st.divider()
st.header("📈 Veri Seti Analizi ve Model Performansı")

X_tum_veri = df_all[['danceability', 'energy', 'tempo', 'acousticness', 'loudness', 'valence']]
y_tum_veri = df_all['popularity']
X_scaled_tum = scaler.transform(X_tum_veri)

y_pred_rf = model.predict(X_scaled_tum)
rf_r2 = r2_score(y_tum_veri, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_tum_veri, y_pred_rf))

st.subheader("Model Başarı Metrikleri (Random Forest)")
col_m1, col_m2 = st.columns(2)
col_m1.metric("R² Skoru (Belirlenim Katsayısı)", f"{rf_r2:.4f}")
col_m2.metric("RMSE (Hata Payı)", f"{rf_rmse:.2f}")

st.write("---")

col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Korelasyon Isı Haritası")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    corr_matrix = df_all[['danceability', 'energy', 'tempo', 'acousticness', 'loudness', 'valence', 'popularity']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)

with col_graph2:
    st.subheader("Popülerlik Skoru Dağılımı")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    sns.histplot(df_all['popularity'], bins=50, color='skyblue', edgecolor='black', ax=ax_hist)
    ax_hist.set_xlabel("Popularity (0-100)")
    ax_hist.set_ylabel("Şarkı Sayısı")
    st.pyplot(fig_hist)
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Download latest version
path = kagglehub.dataset_download("yamaerenay/spotify-dataset-19212020-600k-tracks")
print("Path to dataset files:", path)

csv_path = f"{path}/tracks.csv"
df = pd.read_csv(csv_path)

features = ['danceability','energy','tempo',
            'acousticness', 'loudness','valence']

purpose = 'popularity'

df_model = df[features + [purpose]].copy()

df_model.dropna(inplace=True)
df_model = df_model[df_model[purpose] > 0]

X = df_model[features]
y = df_model[purpose]

# Özellik ölçeklendirme (Feature Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(10, 8))
correlation_matrix = df_model.corr()
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm", fmt=".2f",
            linewidths=0.5)
plt.title("Correlation HeatMap")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df_model[purpose], bins=50, color='skyblue', edgecolor='black')
plt.title("Popularity Score(0-100)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred = mlr_model.predict(X_test)


r2 = r2_score(y_test, y_pred)
n = len(y_test)
k = X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

print(f"R² (Determinasyon Katsayısı): {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"MAE (Ortalama Mutlak Hata): {mae:.4f}")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.4f}")

# Katsayıları (Beta) inceleme
coefficients = pd.DataFrame({'Özellik': features, 'Katsayı': mlr_model.coef_}) # [cite: 169, 170, 227]
print("\nModel Katsayıları:\n", coefficients.sort_values(by='Katsayı', ascending=False))



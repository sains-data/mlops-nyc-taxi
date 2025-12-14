# 🚀 Panduan Deploy Heroku (NYC Taxi API)

Panduan ini berisi langkah-langkah untuk mendeploy API prediksi tarif taksi ke Heroku.

## 📋 Prasyarat

1. **Heroku CLI**: [Instal di sini](https://devcenter.heroku.com/articles/heroku-cli)
2. **Akun Heroku**: Daftar di [heroku.com](https://heroku.com)
3. **Git**: Terinstal di terminal.

---

## 🛠️ Pilihan Metode Deploy

Saya menyarankan **METODE 1 (Docker)** karena lebih stabil dan sama persis dengan yang kita test di local.

### 🔵 METODE 1: Docker (Recommended)

Metode ini membungkus aplikasi + model ke dalam container.

1. **Login ke Heroku Container Registry**
   ```bash
   heroku login
   heroku container:login
   ```

2. **Buat Aplikasi**
   ```bash
   heroku create nyc-taxi-api-leccaz  # Ganti nama sesuka Anda
   ```

3. **Push Docker Image**
   Perintah ini akan build image dan upload ke Heroku (agak lama tergantung koneksi, ~400MB).
   ```bash
   heroku container:push web --app nyc-taxi-api-leccaz
   ```

4. **Release Deployment**
   Aktifkan container yang baru di-push.
   ```bash
   heroku container:release web --app nyc-taxi-api-leccaz
   ```

✅ **Selesai!** Cek endpoint health:
`https://nyc-taxi-api-leccaz.herokuapp.com/health`

---

### 🟠 METODE 2: Git Standard

Jika Anda tidak ingin menggunakan Docker dan lebih suka `git push heroku main`.

1. **Persiapan Model**
   Karena `models/` di-ignore oleh git, kita harus memaksanya masuk agar bisa dibaca di Heroku.
   ```bash
   git add -f models/production_model.joblib
   ```
   *Note: Pastikan ukuran < 100MB (Git limit). Model kita ~600KB jadi aman.*

2. **Set Config Var**
   Beri tahu aplikasi di mana lokasi modelnya.
   ```bash
   heroku config:set MODEL_PATH=models/production_model.joblib --app nyc-taxi-api-leccaz
   ```

3. **Push to Deploy**
   ```bash
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

---

## 🔍 Troubleshooting

**Error: "Model not found"**
- Pastikan `MODEL_PATH` environment variable sudah di-set.
- Pastikan file model ikut ter-upload (gunakan `git add -f`).

**Error: Slug size too large (>500MB)**
- ML libraries seperti pandas/scikit-learn/mlflow sangat besar.
- Solusi: Gunakan **Metode 1 (Docker)** karena limitnya jauh lebih besar (ribuan MB).

**MLflow Tracking**
- Di Heroku (Ephemeral filesystem), database SQLite akan reset setiap kali deploy/restart.
- Untuk production tracking, sambungkan `MLFLOW_TRACKING_URI` ke server eksternal (PostgreSQL / DagsHub).

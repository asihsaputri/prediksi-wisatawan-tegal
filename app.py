import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Setup awal (sama persis dengan notebook)
warnings.filterwarnings("ignore")

SEED = 42
os.environ['PYTHONHASHSEED']       = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXT = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def clean_list(values, ndigits=2):
    """
    Konversi array/Series numerik menjadi list Python murni yang AMAN untuk JSON:
    - np.nan / None  -> None (JS: null)
    - np.inf / -inf  -> None (JS: null)
    - sisanya        -> float biasa, dibulatkan
    Ini mencegah Flask mengirim literal NaN/Infinity yang membuat
    JSON.parse() di browser gagal dengan error "Unexpected token 'N'...".
    """
    out = []
    for v in np.asarray(values, dtype=float):
        if v is None or np.isnan(v) or np.isinf(v):
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def clean_float(value, ndigits=4):
    """Versi scalar dari clean_list, dipakai untuk metrik (MAE, MSE, dll)."""
    v = float(value)
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, ndigits)


# Fungsi utama: reproduksi notebook sel per sel
def run_model(df_raw: pd.DataFrame) -> dict:

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # SEL 1: Baca & siapkan data
    data = df_raw.copy()

    # Hapus kolom Unnamed
    unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
    data = data.drop(columns=unnamed_cols)

    # Deteksi kolom bulan & kunjungan
    date_col = val_col = None
    for c in data.columns:
        cl = c.lower()
        if any(k in cl for k in ["bulan", "tanggal", "date", "waktu", "periode"]):
            date_col = c
        elif any(k in cl for k in ["kunjungan", "wisatawan", "jumlah", "nilai", "tegal"]):
            val_col = c
    if date_col is None: date_col = data.columns[0]
    if val_col  is None: val_col  = data.columns[1] if len(data.columns) > 1 else data.columns[0]

    data = data[[date_col, val_col]].copy()
    data = data.rename(columns={date_col: 'bulan', val_col: 'kunjungan_wisatawan'})
    data['kunjungan_wisatawan'] = pd.to_numeric(data['kunjungan_wisatawan'], errors='coerce')
    data = data.reset_index(drop=True)

    # Set DatetimeIndex bulanan (MS) — seperti notebook: date_range dari 2015-01-01
    data.index = pd.date_range(start='2015-01-01', periods=len(data), freq='MS')
    data.index.name = 'bulan'
    data = data.drop(columns=['bulan'])

    # SEL 2: Deteksi missing value & Z-Score
    data['z_score'] = np.abs(
        (data['kunjungan_wisatawan'] - data['kunjungan_wisatawan'].mean())
        / data['kunjungan_wisatawan'].std()
    )

    data = data.reset_index()   # bulan jadi kolom biasa

    THRESHOLD = 3
    outliers = data[data['z_score'] > THRESHOLD]

    # SEL 3: Interpolasi & Seasonal Decompose untuk missing
    data = data.set_index('bulan')

    series        = data['kunjungan_wisatawan']
    series_interp = series.interpolate(method='time')

    dec = seasonal_decompose(series_interp, model='additive', period=12)

    filled_values = (dec.trend + dec.seasonal).bfill().ffill()

    series_filled = series.copy()
    series_filled[series.isna()] = filled_values[series.isna()]

    data['kunjungan_wisatawan_filled'] = series_filled

    # SEL 4: Ganti Outlier dengan Trend + Seasonal
    data.index = pd.to_datetime(data.index)

    series      = data['kunjungan_wisatawan_filled'].copy()
    series_asli = series.copy()  # noqa (dipertahankan sesuai notebook)

    outlier_index = pd.to_datetime(outliers['bulan'])
    mask_awal     = series.index.isin(outlier_index)

    dec      = seasonal_decompose(series, model='additive', period=12)
    trend    = dec.trend
    seasonal = dec.seasonal

    reconstructed     = (trend + seasonal).interpolate(method='time').bfill().ffill()
    series[mask_awal] = reconstructed[mask_awal]

    data['kunjungan_wisatawan_filled'] = series

    # SEL 5: ADF Test (dijalankan, hasilnya tidak dikirim ke frontend)
    series = data['kunjungan_wisatawan_filled']
    adfuller(series, regression='ct')   # sama dengan notebook, hasil diprint saja

    # SEL 6: Differencing biasa (d=1) — tidak dipakai lanjut
    series_filled = data['kunjungan_wisatawan_filled']
    diff_1 = series_filled.diff().dropna()   # noqa

    # SEL 7: Differencing musiman (D=1, s=12)
    diff_seasonal = series_filled.diff(12).dropna()

    # SEL 8: ADF pada diff musiman
    adfuller(diff_seasonal)   # dijalankan sesuai notebook

    # SEL 9: Shift + Log transform
    series_diff_seasonal = diff_seasonal.copy()

    shift_value         = abs(series_diff_seasonal.min()) + 1
    series_diff_shifted = series_diff_seasonal + shift_value

    data['diff_seasonal_log'] = np.log(series_diff_shifted)

    # SEL 10: Split train/test (90/10)
    time_series_data = data['diff_seasonal_log'].copy()

    train_size = int(len(time_series_data) * 0.9)

    train_data = time_series_data.iloc[:train_size]
    test_data  = time_series_data.iloc[train_size:]

    # SEL 11: SARIMA(1,0,2)(0,1,0,24)
    model = SARIMAX(
        train_data,
        order=(1, 0, 2),
        seasonal_order=(0, 1, 0, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_results = model.fit(disp=False)

    # SEL 12: Forecast SARIMA ke testing
    predictions       = sarimax_results.forecast(steps=len(test_data))
    predictions.index = test_data.index

    # SEL 13: Hitung residual
    predictions.index = test_data.index

    residual_test = test_data - predictions

    valid_idx     = residual_test.dropna().index
    residual_test = residual_test.loc[valid_idx]
    predictions   = predictions.loc[valid_idx]
    test_data     = test_data.loc[valid_idx]

    # SEL 14: Dataset LSTM dari residual
    def create_lstm_dataset(series, look_back):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back])
        return np.array(X), np.array(y)

    look_back = 6

    X_test_res, y_test_res = create_lstm_dataset(residual_test.values, look_back)

    # SEL 17: Train LSTM final dengan arsitektur terbaik
    K.clear_session()
    model_lstm = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(look_back, 1)),
        LSTM(16, activation='tanh'),
        Dense(1)
    ])
    model_lstm.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model_lstm.fit(X_test_res, y_test_res, epochs=100, batch_size=46, verbose=0)

    # SEL 18: Prediksi residual testing
    lstm_residual_pred = model_lstm.predict(X_test_res, verbose=0).flatten()

    residual_pred_series = pd.Series(
        lstm_residual_pred,
        index=test_data.index[look_back:],
        name='Residual_LSTM_Pred'
    )

    # SEL 19: Hybrid Prediction
    hybrid_prediction      = predictions.iloc[look_back:] + residual_pred_series
    hybrid_prediction      = hybrid_prediction.clip(lower=0)
    hybrid_prediction.name = 'Hybrid_Prediction'

    # SEL 20: Evaluasi Hybrid
    actual = test_data.loc[hybrid_prediction.index]

    mae_hybrid  = mean_absolute_error(actual, hybrid_prediction)
    mse_hybrid  = mean_squared_error(actual, hybrid_prediction)
    rmse_hybrid = np.sqrt(mse_hybrid)

    epsilon     = 1e-8
    mape_hybrid = np.mean(np.abs((actual - hybrid_prediction) / (actual + epsilon))) * 100

    # SEL 21: Forecast 12 bulan ke depan
    def normalize_index(series):
        s = series.copy()
        s.index = pd.DatetimeIndex(s.index.to_period('M').to_timestamp())
        return s

    full_series_asli    = normalize_index(np.expm1(data['diff_seasonal_log'].copy()))
    hybrid_test_asli    = normalize_index(np.expm1(hybrid_prediction))
    residual_test_clean = normalize_index(residual_test)

    model_lstm.build(input_shape=(None, look_back, 1))

    last_date    = hybrid_test_asli.index[-1]
    future_index = pd.date_range(start=last_date, periods=13, freq='MS')[1:]

    sarima_future_log       = sarimax_results.forecast(steps=12)
    sarima_future_log.index = future_index

    seed_window = [float(v) for v in residual_test_clean.values[-look_back:]]

    lstm_residual_future = []
    for _ in range(12):
        x_input  = np.array(seed_window[-look_back:], dtype=np.float32).reshape(1, look_back, 1)
        next_res = float(model_lstm(x_input, training=False).numpy().flatten()[0])
        lstm_residual_future.append(next_res)
        seed_window.append(next_res)

    lstm_residual_future = np.array(lstm_residual_future, dtype=np.float32)

    hybrid_future_log  = np.clip(sarima_future_log.values + lstm_residual_future, 0, None)
    sarima_future_asli = np.expm1(sarima_future_log.values)
    hybrid_future_asli = np.expm1(hybrid_future_log)

    df_future = pd.DataFrame({
        'Prediksi_SARIMAX': sarima_future_asli,
        'Prediksi_Hybrid':  hybrid_future_asli,
    }, index=future_index)
    df_future.index.name = 'Bulan'

    bridge = pd.DataFrame({
        'Prediksi_SARIMAX': [sarima_future_asli[0]],
        'Prediksi_Hybrid':  [hybrid_test_asli.values[-1]],
    }, index=[hybrid_test_asli.index[-1]])

    bridge_hist      = pd.Series([full_series_asli.values[-1]], index=[hybrid_test_asli.index[0]])
    hist_only        = full_series_asli[full_series_asli.index < hybrid_test_asli.index[0]]
    hist_with_bridge = pd.concat([hist_only, bridge_hist])   # noqa (sesuai notebook)

    df_future_plot = pd.concat([bridge, df_future])   # noqa (sesuai notebook)

    def fmt(idx):
        return [d.strftime('%Y-%m') for d in idx]

    hist_plot = hist_only.iloc[-12:]   # 1 tahun terakhir, sesuai zoom notebook

    return {
        "historis": {
            "dates": fmt(hist_only.index),
            "values": clean_list(hist_only.values)
        },
        "testing_actual": {
            "dates": fmt(actual.index),
            "values": clean_list(np.expm1(actual.values))
        },
        "testing_sarima": {
            "dates": fmt(actual.index),
            "values": clean_list(np.expm1(predictions.loc[actual.index].values))
        },
        "testing_hybrid": {
            "dates": fmt(actual.index),
            "values": clean_list(np.expm1(hybrid_prediction.values))
        },
        # dipakai oleh chart
        "future_hybrid": {
            "dates": fmt(df_future.index),
            "values": clean_list(df_future["Prediksi_Hybrid"].values)
        },
        "hybrid_future": {
            "dates": fmt(df_future.index),
            "values": clean_list(df_future["Prediksi_Hybrid"].values)
        },
        "sarima_future": {
            "dates": fmt(df_future.index),
            "values": clean_list(df_future["Prediksi_SARIMAX"].values)
        },
        "sarima_metrics": {
            "mae":  clean_float(mean_absolute_error(actual, predictions.loc[actual.index])),
            "mse":  clean_float(mean_squared_error(actual, predictions.loc[actual.index])),
            "rmse": clean_float(np.sqrt(mean_squared_error(actual, predictions.loc[actual.index]))),
            "mape": clean_float(np.mean(np.abs((actual - predictions.loc[actual.index]) / (actual + 1e-8))) * 100)
        },
        "hybrid_metrics": {
            "mae":  clean_float(mae_hybrid),
            "mse":  clean_float(mse_hybrid),
            "rmse": clean_float(rmse_hybrid),
            "mape": clean_float(mape_hybrid)
        },
        "future_year": int(future_index[-1].year)
    }

# Route
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Format tidak didukung. Gunakan file .csv"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        df = None
        for sep in [";", ",", "\t"]:
            try:
                tmp = pd.read_csv(filepath, sep=sep)
                if len(tmp.columns) >= 2:
                    df = tmp
                    break
            except Exception:
                continue

        if df is None or len(df.columns) < 2:
            return jsonify({"error": "File CSV harus memiliki minimal 2 kolom."}), 400

        result = run_model(df)
        # result sudah dibersihkan dari NaN/Infinity oleh clean_list/clean_float
        # di dalam run_model(), jadi aman untuk di-jsonify langsung.
        return jsonify({"success": True, "data": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.config["JSON_SORT_KEYS"] = False
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

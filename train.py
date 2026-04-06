"""
train.py — Entrenamiento del GoldTransformer sobre el dataset SMC.

Uso:
    python train.py
    python train.py --epochs 50 --batch 64 --seq 96 --lr 1e-4

Flujo:
    1. Carga el Parquet generado por GoldQuantProcessor
    2. Construye secuencias de longitud SEQ_LEN (ventana deslizante)
    3. Define labels: direction (0=Long, 1=Short, 2=NoTrade) y rr
    4. Entrena el GoldTransformer con callbacks (EarlyStopping, TensorBoard)
    5. Guarda el modelo en models/gold_transformer.keras
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from models.transformer import build_gold_transformer, SEQ_LEN, N_FEATURES

# ── Columnas de features de entrada al Transformer ────────────────────────────
# Deben coincidir con FEATURE_COLS de dataset.py, excluyendo labels
INPUT_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_lagless", "atr", "vol_zscore",
    "fvg_bull_size", "fvg_bear_size",
    "ob_freshness", "ob_type", "ob_midpoint",
    "trend", "bos_bullish", "bos_bearish",
    "choch_bullish", "choch_bearish",
    "fvg_bull", "fvg_bear",
    "bull_ob", "bear_ob",
    "equal_high_pool", "equal_low_pool",
    "in_asia_session", "judas_swing_bull", "judas_swing_bear",
    "asia_high", "asia_low",
    "m15_close", "m15_trend", "m15_ob_midpoint",
    "m15_bull_ob", "m15_bear_ob",
    "m15_bos_bullish", "m15_bos_bearish",
    "m15_fvg_bull", "m15_fvg_bear",
    "h1_close", "h1_trend", "h1_ob_midpoint",
    "h1_bull_ob", "h1_bear_ob",
]

LABEL_DIRECTION = "direction_label"
LABEL_RR        = "rr_achieved"

DATA_PATH  = Path("data/processed/gold_smc_dataset.parquet")
MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs/fit")


# ── Label engineering ─────────────────────────────────────────────────────────

def make_direction_label(df: pd.DataFrame) -> pd.Series:
    """
    Derive a 3-class direction label from OB type + DD filter:
        0 = Long  (bull_ob=True, dd_filter_ok=True)
        1 = Short (bear_ob=True, dd_filter_ok=True)
        2 = No-Trade (everything else)
    """
    label = pd.Series(2, index=df.index, dtype=np.int32)  # default: No-Trade

    if "bull_ob" in df.columns and "dd_filter_ok" in df.columns:
        long_mask  = df["bull_ob"].astype(bool) & df["dd_filter_ok"].astype(bool)
        short_mask = df["bear_ob"].astype(bool) & df["dd_filter_ok"].astype(bool)
        label[long_mask]  = 0
        label[short_mask] = 1

    return label


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    input_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from the dataset.

    Returns:
        X: (N, seq_len, n_features)  float32
        y_dir: (N,)                  int32    direction label at last bar
        y_rr:  (N, 1)               float32  RR at last bar (0 if no OB)
    """
    available = [c for c in input_cols if c in df.columns]
    n_feat    = len(available)
    print(f"  Features disponibles: {n_feat}/{len(input_cols)}")

    data      = df[available].fillna(0).astype(np.float32).to_numpy()
    direction = df[LABEL_DIRECTION].to_numpy(dtype=np.int32)
    rr        = df[LABEL_RR].fillna(0).astype(np.float32).to_numpy() \
                if LABEL_RR in df.columns else np.zeros(len(df), dtype=np.float32)

    # sliding_window_view on (N, n_feat) with window (seq_len, n_feat) →
    # shape (N - seq_len + 1, 1, seq_len, n_feat).
    # We need N - seq_len windows so that each window[i] predicts at bar i + seq_len.
    n_seq   = len(data) - seq_len
    windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, n_feat))
    X       = windows[:n_seq, 0, :, :]          # (n_seq, seq_len, n_feat)
    y_dir   = direction[seq_len:]               # label at bar i + seq_len
    y_rr    = rr[seq_len:].reshape(-1, 1)

    return X, y_dir, y_rr


# ── Entrenamiento principal ───────────────────────────────────────────────────

def train(
    epochs:    int   = 30,
    batch:     int   = 128,
    seq:       int   = SEQ_LEN,
    lr:        float = 1e-4,
    val_split: float = 0.15,
    test_split:float = 0.10,
):
    print("=" * 60)
    print("  GoldTransformer — Entrenamiento")
    print("=" * 60)

    # 1. Cargar dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATA_PATH}\n"
            "Ejecuta primero: python -m src.processor"
        )
    print(f"\n[1] Cargando dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"    Filas: {len(df):,}  |  Columnas: {df.shape[1]}")

    # 2. Generar labels
    print("\n[2] Generando labels de dirección...")
    df[LABEL_DIRECTION] = make_direction_label(df)
    dist = df[LABEL_DIRECTION].value_counts().sort_index()
    print(f"    Long: {dist.get(0,0):,}  Short: {dist.get(1,0):,}  No-Trade: {dist.get(2,0):,}")

    # Pesos de clase balanceados (sqrt para moderar la penalizacion extrema)
    # Con 17 OBs vs 370k No-Trade, sklearn daría w~10k — se aplica sqrt para
    # evitar que el modelo sobre-ajuste a los pocos ejemplos Long/Short.
    from sklearn.utils.class_weight import compute_class_weight
    import math
    classes   = np.array([0, 1, 2])
    raw_w     = compute_class_weight("balanced", classes=classes,
                                     y=df[LABEL_DIRECTION].to_numpy())
    cw        = {i: math.sqrt(w) for i, w in enumerate(raw_w)}
    print(f"    Class weights (sqrt-balanced): { {k: round(v,2) for k,v in cw.items()} }")

    # 3. Construir secuencias
    print(f"\n[3] Construyendo secuencias (ventana={seq} barras)...")
    X, y_dir, y_rr = build_sequences(df, seq, INPUT_COLS)
    print(f"    X shape: {X.shape}  dtype: {X.dtype}")
    n_feat = X.shape[2]

    # 4. Split temporal (NO shuffle — time-series)
    n         = len(X)
    n_test    = int(n * test_split)
    n_val     = int(n * val_split)
    n_train   = n - n_val - n_test

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_dir_train = y_dir[:n_train];  y_dir_val = y_dir[n_train:n_train+n_val];  y_dir_test = y_dir[n_train+n_val:]
    y_rr_train  = y_rr[:n_train];   y_rr_val  = y_rr[n_train:n_train+n_val];   y_rr_test  = y_rr[n_train+n_val:]

    print(f"    Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    # 5. Construir modelo
    print(f"\n[4] Construyendo modelo (N_FEATURES={n_feat})...")
    model = build_gold_transformer(
        seq_len    = seq,
        n_features = n_feat,
    )
    # Recompile con el learning rate del argumento CLI
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss={
            "direction": tf.keras.losses.SparseCategoricalCrossentropy(),
            "rr":        tf.keras.losses.Huber(delta=0.5),
        },
        loss_weights={"direction": 1.0, "rr": 0.3},
        metrics={"direction": ["accuracy"], "rr": ["mae"]},
    )
    model.summary(line_length=80)

    # 6. Callbacks
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_direction_accuracy",
            patience=7,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "gold_transformer_best.keras"),
            monitor="val_direction_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR),
            histogram_freq=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 7. Entrenar
    print(f"\n[5] Entrenando — {epochs} épocas máx, batch={batch}...")
    history = model.fit(
        X_train,
        {"direction": y_dir_train, "rr": y_rr_train},
        validation_data=(X_val, {"direction": y_dir_val, "rr": y_rr_val}),
        epochs=epochs,
        batch_size=batch,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Evaluar en test
    print("\n[6] Evaluando en test set...")
    test_results = model.evaluate(
        X_test,
        {"direction": y_dir_test, "rr": y_rr_test},
        verbose=0,
    )
    metrics = dict(zip(model.metrics_names, test_results))
    print(f"    Test Loss:      {metrics.get('loss', 0):.4f}")
    print(f"    Test Acc Dir:   {metrics.get('direction_accuracy', 0):.4f}  ({metrics.get('direction_accuracy', 0)*100:.1f}%)")
    print(f"    Test MAE RR:    {metrics.get('rr_mae', 0):.4f}")

    # 9. Guardar modelo final
    final_path = MODEL_DIR / "gold_transformer_final.keras"
    model.save(str(final_path))
    print(f"\n[7] Modelo guardado -> {final_path}")

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar GoldTransformer")
    parser.add_argument("--epochs", type=int,   default=30,   help="Max épocas")
    parser.add_argument("--batch",  type=int,   default=128,  help="Batch size")
    parser.add_argument("--seq",    type=int,   default=SEQ_LEN, help="Longitud secuencia")
    parser.add_argument("--lr",     type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        seq=args.seq,
        lr=args.lr,
    )

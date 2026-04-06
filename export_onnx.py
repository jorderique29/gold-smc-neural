"""
export_onnx.py — Exportar GoldTransformer a formato ONNX para MT5.

Uso:
    python export_onnx.py
    python export_onnx.py --model models/gold_transformer_final.keras
    python export_onnx.py --model models/gold_transformer_best.keras --out models/gold_transformer.onnx

Requisitos:
    pip install tf2onnx onnx onnxruntime

El archivo .onnx resultante puede cargarse en MT5 con OnnxCreate() / OnnxRun().

Notas para integración MT5 (MQL5):
    - Input:  "m5_sequence" shape [1, 96, N_FEATURES]  float32
    - Output: "direction"   shape [1, 3]               float32  (softmax probs)
              "rr"          shape [1, 1]                float32  (reward:risk)
    - La capa de atencion usa operaciones estaticas de TF → compatible con ONNX opset 13+
"""
import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np


def check_dependencies() -> bool:
    """Verificar que tf2onnx y onnxruntime esten disponibles."""
    missing = []
    try:
        import tf2onnx
    except ImportError:
        missing.append("tf2onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    if missing:
        print(f"[export_onnx] Instalando dependencias: {missing}")
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing + ["-q"]
        )
    return True


def export_to_onnx(
    model_path: str = "models/gold_transformer_final.keras",
    out_path:   str = "models/gold_transformer.onnx",
    opset:      int = 13,
) -> Path:
    """
    Exportar un modelo Keras guardado a formato ONNX.

    Args:
        model_path: ruta al archivo .keras
        out_path:   ruta destino del .onnx
        opset:      version ONNX opset (13+ recomendado para GELU y softmax)

    Returns:
        Path al archivo .onnx generado
    """
    check_dependencies()

    import tensorflow as tf
    import tf2onnx
    import onnx

    model_p = Path(model_path)
    out_p   = Path(out_path)

    if not model_p.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_p}\n"
            "Ejecuta primero: python train.py"
        )

    print(f"[export_onnx] Cargando modelo: {model_p}")
    model = tf.keras.models.load_model(str(model_p))
    model.summary(line_length=80)

    # Obtener shape real del input del modelo
    input_shape = model.input_shape  # (None, seq_len, n_features)
    seq_len  = input_shape[1]
    n_feat   = input_shape[2]
    print(f"[export_onnx] Input shape: {input_shape}")

    # Especificar la firma de entrada para tf2onnx
    input_signature = [
        tf.TensorSpec(
            shape=(None, seq_len, n_feat),
            dtype=tf.float32,
            name="m5_sequence",
        )
    ]

    print(f"[export_onnx] Convirtiendo a ONNX (opset={opset})...")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(out_p),
    )

    # Verificar el modelo ONNX
    onnx.checker.check_model(onnx_model)
    print(f"[export_onnx] Verificacion ONNX: OK")

    # Test de inferencia con onnxruntime
    print(f"[export_onnx] Test de inferencia...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_p))

    # Entrada dummy
    dummy = np.random.randn(1, seq_len, n_feat).astype(np.float32)
    outputs = sess.run(None, {"m5_sequence": dummy})

    print(f"[export_onnx] Output 'direction' shape: {outputs[0].shape}  -> {outputs[0]}")
    print(f"[export_onnx] Output 'rr'        shape: {outputs[1].shape}  -> {outputs[1]}")

    size_mb = out_p.stat().st_size / 1024 / 1024
    print(f"\n[export_onnx] Exportacion completada!")
    print(f"  Archivo:  {out_p.resolve()}")
    print(f"  Tamanio:  {size_mb:.2f} MB")
    print(f"  Opset:    {opset}")
    print(f"\n--- Integracion MT5 (MQL5) ---")
    print(f"  long model = OnnxCreate(\"{out_p.name}\", ONNX_DEFAULT);")
    print(f"  // Input:  'x' shape [1, {seq_len}, {n_feat}] float32")
    print(f"  // Output: 'direction' [1, 3]  |  'rr' [1, 1]")
    print(f"  OnnxRun(model, ONNX_DATA_TYPE_FLOAT, input, output_dir, output_rr);")

    return out_p


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar GoldTransformer a ONNX")
    parser.add_argument(
        "--model",
        default="models/gold_transformer_final.keras",
        help="Ruta al modelo .keras",
    )
    parser.add_argument(
        "--out",
        default="models/gold_transformer.onnx",
        help="Ruta de salida del .onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        out_path=args.out,
        opset=args.opset,
    )

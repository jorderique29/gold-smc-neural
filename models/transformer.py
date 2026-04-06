# models/transformer.py
"""
XAUUSD Multi-Timeframe Transformer — GPT-style Decoder-Only Architecture.

Design:
  - Decoder-only (causal self-attention) → no data leakage across time steps
  - Rotary Position Embeddings (RoPE) → better length extrapolation than learned PE
  - Pre-LayerNorm blocks → more stable training for financial time-series
  - GELU activation in FFN → smoother gradients than ReLU
  - Dual output heads: direction classification + RR regression

Input  shape: (batch, 96, 42)  →  96 M5 bars × 42 normalized features
Output:
    direction : (batch, 3)   softmax probs  [Long, Short, No-Trade]
    rr        : (batch, 1)   predicted reward:risk ratio

Reference for RoPE: Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
Reference for architecture choices: Brown et al. (2020) "Language Models are Few-Shot Learners" (GPT-3)
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Package name for @register_keras_serializable — all custom layers use this
_PKG = "GoldSMC"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN    = 96     # context window: 96 × 5min = 8 hours of M5 bars
N_FEATURES = 42     # normalized input features per bar
D_MODEL    = 128    # model embedding dimension
N_HEADS    = 4      # attention heads (D_MODEL must be divisible by N_HEADS)
D_FF       = 256    # feed-forward hidden dimension
N_LAYERS   = 3      # number of Transformer blocks
DROPOUT    = 0.1    # dropout rate
N_CLASSES  = 3      # Long / Short / No-Trade


# ── Rotary Position Embeddings (RoPE) ─────────────────────────────────────────

def _rope_frequencies(seq_len: int, d_head: int) -> tf.Tensor:
    """
    Precompute RoPE rotation frequencies.

    Returns tensor of shape (seq_len, d_head) where each position i gets
    a unique rotation angle for each dimension pair.
    """
    theta = 1.0 / (10000.0 ** (
        tf.cast(tf.range(0, d_head, 2), tf.float32) / float(d_head)
    ))  # shape: (d_head//2,)
    positions = tf.cast(tf.range(seq_len), tf.float32)  # (seq_len,)
    freqs = tf.einsum("i,j->ij", positions, theta)       # (seq_len, d_head//2)
    return tf.concat([freqs, freqs], axis=-1)             # (seq_len, d_head)


def _apply_rope(x: tf.Tensor, freqs: tf.Tensor) -> tf.Tensor:
    """
    Apply rotary embeddings to Q or K tensor.

    Args:
        x:     shape (batch, heads, seq_len, d_head)
        freqs: shape (seq_len, d_head)

    Returns:
        Rotated tensor of same shape as x.
    """
    cos = tf.cos(freqs)[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, d_head)
    sin = tf.sin(freqs)[tf.newaxis, tf.newaxis, :, :]

    # Rotate: for each dimension pair (2i, 2i+1), apply 2D rotation
    x_even = x[..., ::2]   # even indices
    x_odd  = x[..., 1::2]  # odd  indices
    x_rot  = tf.concat([-x_odd, x_even], axis=-1)  # (batch, heads, seq, d_head)

    return x * cos + x_rot * sin


# ── Causal Multi-Head Attention with RoPE ─────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package=_PKG)
class CausalMHAWithRoPE(layers.Layer):
    """
    Causal (masked) Multi-Head Attention with Rotary Position Embeddings.

    Attention is strictly causal: position i can only attend to positions <= i.
    This prevents any look-ahead bias in the sequence model.
    """

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model
        self.scale   = self.d_head ** -0.5

        self.q_proj = layers.Dense(d_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(d_model, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(d_model, use_bias=False, name="v_proj")
        self.out    = layers.Dense(d_model, use_bias=False, name="out_proj")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_len    = tf.shape(x)[1]

        def split_heads(z):
            z = tf.reshape(z, [batch_size, seq_len, self.n_heads, self.d_head])
            return tf.transpose(z, [0, 2, 1, 3])  # (B, H, T, d_head)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Apply RoPE to Q and K
        freqs = _rope_frequencies(seq_len, self.d_head)
        Q = _apply_rope(Q, freqs)
        K = _apply_rope(K, freqs)

        # Scaled dot-product attention with causal mask
        scores = tf.matmul(Q, K, transpose_b=True) * self.scale  # (B, H, T, T)

        # Causal mask: upper triangle → -inf (so softmax gives ~0 weight)
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
        )  # lower triangular (including diagonal)
        causal_bias = (1.0 - causal_mask) * -1e9
        scores = scores + causal_bias

        weights = tf.nn.softmax(scores, axis=-1)  # (B, H, T, T)
        context = tf.matmul(weights, V)           # (B, H, T, d_head)

        # Merge heads
        context = tf.transpose(context, [0, 2, 1, 3])         # (B, T, H, d_head)
        context = tf.reshape(context, [batch_size, seq_len, self.d_model])
        return self.out(context)


# ── Transformer Block (Pre-Norm) ──────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package=_PKG)
class TransformerBlock(layers.Layer):
    """
    GPT-style Transformer block with pre-LayerNorm.

    Structure (per block):
        x = x + Dropout(Attn(LayerNorm(x)))
        x = x + Dropout(FFN(LayerNorm(x)))

    Pre-norm is more stable than post-norm, especially with fewer layers
    (N_LAYERS=3) where gradient flow is critical.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn    = CausalMHAWithRoPE(d_model, n_heads)
        self.ffn     = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Attention sub-layer (pre-norm)
        x = x + self.drop1(self.attn(self.norm1(x), training=training),
                            training=training)
        # FFN sub-layer (pre-norm)
        x = x + self.drop2(self.ffn(self.norm2(x)), training=training)
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

def build_gold_transformer(
    seq_len:    int   = SEQ_LEN,
    n_features: int   = N_FEATURES,
    d_model:    int   = D_MODEL,
    n_heads:    int   = N_HEADS,
    d_ff:       int   = D_FF,
    n_layers:   int   = N_LAYERS,
    dropout:    float = DROPOUT,
    n_classes:  int   = N_CLASSES,
) -> keras.Model:
    """
    Build and compile the XAUUSD Transformer model.

    Architecture:
        Input (B, T, F)
            │
        Dense(d_model) — project features into model dimension
            │
        LayerNorm — normalize input sequence
            │
        [TransformerBlock × n_layers] — causal self-attention + FFN
            │
        x[:, -1, :] — take last token (GPT-style auto-regressive prediction)
            │
        ┌───────────────┐
        │               │
    [CLS head]     [RR head]
    Dense(d_model//2,  Dense(d_model//4,
          gelu)               gelu)
        │                       │
    Dense(n_classes)      Dense(1, relu)
        │                       │
    Softmax               rr_prediction
        │
    direction_probs

    Args:
        seq_len:    number of M5 bars in context window (default 96)
        n_features: number of input features per bar (default 42)
        d_model:    embedding dimension (default 128)
        n_heads:    attention heads (default 4)
        d_ff:       feed-forward hidden dim (default 256)
        n_layers:   number of Transformer blocks (default 3)
        dropout:    dropout rate (default 0.1)
        n_classes:  output classes (default 3: Long/Short/No-Trade)

    Returns:
        Compiled keras.Model with:
            inputs:  {"m5_sequence": (batch, seq_len, n_features)}
            outputs: {"direction": (batch, n_classes), "rr": (batch, 1)}
    """
    inp = keras.Input(shape=(seq_len, n_features), name="m5_sequence")

    # Feature projection + input normalization
    x = layers.Dense(d_model, name="feature_proj")(inp)
    x = layers.LayerNormalization(epsilon=1e-6, name="input_norm")(x)
    x = layers.Dropout(dropout, name="input_drop")(x)

    # Transformer blocks
    for i in range(n_layers):
        x = TransformerBlock(d_model, n_heads, d_ff, dropout,
                              name=f"transformer_block_{i}")(x)

    # Final normalization
    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)

    # Last-token pooling (GPT-style: predict from last position)
    last = x[:, -1, :]  # (batch, d_model)

    # Direction classification head
    dir_hidden  = layers.Dense(d_model // 2, activation="gelu",
                                name="dir_hidden")(last)
    dir_hidden  = layers.Dropout(dropout, name="dir_drop")(dir_hidden)
    dir_logits  = layers.Dense(n_classes, name="dir_logits")(dir_hidden)
    direction   = layers.Softmax(name="direction")(dir_logits)

    # RR regression head
    rr_hidden   = layers.Dense(d_model // 4, activation="gelu",
                                name="rr_hidden")(last)
    rr_pred     = layers.Dense(1, activation="relu", name="rr")(rr_hidden)

    model = keras.Model(
        inputs=inp,
        outputs={"direction": direction, "rr": rr_pred},
        name="GoldTransformer_v1",
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
        ),
        loss={
            "direction": keras.losses.SparseCategoricalCrossentropy(),
            "rr":        keras.losses.Huber(delta=0.5),
        },
        loss_weights={"direction": 1.0, "rr": 0.3},
        metrics={
            "direction": ["accuracy"],
            "rr":        ["mae"],
        },
    )

    return model


# ── Entry point for schema inspection ────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF verbose output

    model = build_gold_transformer()
    model.summary(expand_nested=True)

    total_params = model.count_params()
    print(f"\n{'='*60}")
    print(f"  GoldTransformer_v1 Architecture Summary")
    print(f"{'='*60}")
    print(f"  Input  : (batch, {SEQ_LEN}, {N_FEATURES})  "
          f"-> {SEQ_LEN} M5 bars x {N_FEATURES} features")
    print(f"  D_MODEL: {D_MODEL}  |  N_HEADS: {N_HEADS}  "
          f"|  N_LAYERS: {N_LAYERS}  |  D_FF: {D_FF}")
    print(f"  Output : direction (3-class softmax) + rr (regression)")
    print(f"  Params : {total_params:,}")
    print(f"{'='*60}")
    print(f"\n  Training targets:")
    print(f"    direction: 0=Long, 1=Short, 2=No-Trade")
    print(f"    rr       : reward:risk ratio (from compute_risk_params)")
    print(f"\n  Context window: {SEQ_LEN} bars x 5min = 8 hours of M5 data")

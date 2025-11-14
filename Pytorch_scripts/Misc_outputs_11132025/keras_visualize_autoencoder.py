#!/usr/bin/env python3
"""
Keras + visualkeras architecture visualizer for the Bowhead autoencoder.

Generates a layered diagram without running training. This is separate from the
PyTorch visualizer and does not require Graphviz; it uses PIL via visualkeras.

Outputs (to repo_root/plots):
- visualkeras_autoencoder_<timestamp>.png
- keras_model_summary.txt (optional text summary)

Requirements:
- TensorFlow (macOS Apple Silicon: `pip install tensorflow-macos tensorflow-metal`)
  or Intel macOS/CPU: `pip install tensorflow`
- visualkeras and pillow: `pip install visualkeras pillow`

Example:
  python Pytorch_scripts/keras_visualize_autoencoder.py \
      --height 121 --width 104 --channels 32 64 128 --latent 64 \
      --unet-skips --upsample-conv --refine

Notes:
- This script mirrors the high-level structure (3 down/ups, optional U-Net skips,
  Upsample+Conv decoder, refine head). It focuses on clean visualization rather
  than bit-exact layer mapping against the PyTorch variant.
- SE blocks are optional; for simplicity we apply them inside encoder convs when enabled.
"""
import argparse
import os
import sys
import time
from typing import Tuple
import re
import warnings
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")


def _import_tf_and_vk():
    """Import TensorFlow/Keras and visualkeras with helpful errors."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:
        print("[error] TensorFlow is not installed. Install one of:")
        print("  - Apple Silicon: pip install tensorflow-macos tensorflow-metal")
        print("  - Intel macOS/CPU: pip install tensorflow")
        print(f"Details: {e}")
        sys.exit(1)
    try:
        import visualkeras  # noqa: F401
    except Exception as e:
        print("[error] visualkeras is not installed. Install with:")
        print("  pip install visualkeras pillow")
        print(f"Details: {e}")
        sys.exit(1)
    from tensorflow import keras
    from tensorflow.keras import layers
    import visualkeras
    return keras, layers, visualkeras


def _shape_to_tuple(s):
    if s is None:
        return None
    # TensorShape or similar
    if hasattr(s, 'as_list'):
        try:
            lst = s.as_list()
            return tuple(lst) if lst is not None else None
        except Exception:
            pass
    try:
        return tuple(s)
    except Exception:
        return None


def _shape_from_tensor(t):
    try:
        return _shape_to_tuple(getattr(t, 'shape', None))
    except Exception:
        return None


def export_layer_details(model, out_csv_path: str, out_md_path: str):
    """Export a per-layer table with names, types, shapes, and key parameters."""
    rows = []
    for idx, lyr in enumerate(model.layers):
        name = getattr(lyr, 'name', f'layer_{idx}')
        cls = lyr.__class__.__name__

        # Input shape(s)
        in_shape = None
        try:
            in_shape = getattr(lyr, 'input_shape', None)
        except Exception:
            in_shape = None
        if in_shape is None:
            inp_t = getattr(lyr, 'input', None)
            if inp_t is not None:
                if isinstance(inp_t, (list, tuple)):
                    in_shape = [_shape_from_tensor(t) for t in inp_t]
                else:
                    in_shape = _shape_from_tensor(inp_t)

        # Output shape(s)
        out_shape = None
        try:
            out_shape = getattr(lyr, 'output_shape', None)
        except Exception:
            out_shape = None
        if out_shape is None:
            out_t = getattr(lyr, 'output', None)
            if out_t is not None:
                if isinstance(out_t, (list, tuple)):
                    out_shape = [_shape_from_tensor(t) for t in out_t]
                else:
                    out_shape = _shape_from_tensor(out_t)

        # Key params from config
        cfg = {}
        try:
            cfg = lyr.get_config()
        except Exception:
            cfg = {}

        details = []
        def add_detail(k):
            v = cfg.get(k, None)
            if v is not None:
                details.append(f"{k}={v}")

        # Common layer-specific attributes
        for key in ("filters", "units", "kernel_size", "strides", "padding", "activation", "pool_size", "interpolation"):
            add_detail(key)

        params = None
        try:
            params = int(lyr.count_params())
        except Exception:
            params = None

        rows.append({
            'index': idx,
            'name': name,
            'type': cls,
            'input_shape': in_shape,
            'output_shape': out_shape,
            'params': params,
            'details': "; ".join(details) if details else "",
        })

    # Write CSV
    try:
        with open(out_csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['index', 'name', 'type', 'input_shape', 'output_shape', 'params', 'details'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved layer details CSV: {out_csv_path}")
    except Exception as e:
        print(f"[info] Failed to write CSV: {e}")

    # Write Markdown
    try:
        with open(out_md_path, 'w') as fh:
            fh.write("| # | Name | Type | Input shape | Output shape | Params | Details |\n")
            fh.write("|---:|------|------|-------------|--------------:|------:|---------|\n")
            for r in rows:
                fh.write(f"| {r['index']} | {r['name']} | {r['type']} | {r['input_shape']} | {r['output_shape']} | {r['params']} | {r['details']} |\n")
        print(f"Saved layer details Markdown: {out_md_path}")
    except Exception as e:
        print(f"[info] Failed to write Markdown: {e}")


def se_block(x, reduction: int = 8):
    """Keras implementation of Squeeze-and-Excitation for channels-last tensors."""
    from tensorflow.keras import layers
    c = x.shape[-1]
    if c is None:
        # Fallback if static channel dim is unknown; skip SE for safety
        return x
    hidden = max(1, c // reduction)
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(hidden, activation="relu")(s)
    s = layers.Dense(int(c), activation="sigmoid")(s)
    s = layers.Reshape((1, 1, int(c)))(s)
    return layers.Multiply()([x, s])


def build_keras_autoencoder(height: int = 121,
                            width: int = 104,
                            channels: Tuple[int, int, int] = (32, 64, 128),
                            latent_dim: int = 64,
                            refine: bool = True,
                            unet_skips: bool = True,
                            upsample_conv: bool = True,
                            se_blocks: bool = False):
    """Construct a Keras autoencoder model mirroring the PyTorch one at a high level."""
    keras, layers, _vk = _import_tf_and_vk()  # ensure TF is present

    inputs = keras.Input(shape=(height, width, 1), name="input")

    # Encoder
    x = inputs
    # Enc1
    x1 = layers.Conv2D(channels[0], 3, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    if se_blocks:
        x1 = se_block(x1)
    p1 = layers.MaxPooling2D(2)(x1)  # (H/2, W/2)

    # Enc2
    x2 = layers.Conv2D(channels[1], 3, padding="same")(p1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    if se_blocks:
        x2 = se_block(x2)
    p2 = layers.MaxPooling2D(2)(x2)  # (H/4, W/4)

    # Enc3
    x3 = layers.Conv2D(channels[2], 3, padding="same")(p2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    if se_blocks:
        x3 = se_block(x3)
    p3 = layers.MaxPooling2D(2)(x3)  # (H/8, W/8)

    # Latent
    h8, w8 = height // 8, width // 8
    flat = layers.Flatten()(p3)
    z = layers.Dense(latent_dim * 2)(flat)
    z = layers.BatchNormalization()(z)
    z = layers.ReLU()(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(latent_dim, name="latent")(z)

    y = layers.Dense(channels[2] * h8 * w8)(z)
    y = layers.ReLU()(y)
    y = layers.Reshape((h8, w8, channels[2]))(y)

    # Decoder helper blocks
    def up_block(y_in, out_ch, skip_tensor=None, align_to=None):
        yb = y_in
        if upsample_conv:
            yb = layers.UpSampling2D(2, interpolation="bilinear")(yb)
            yb = layers.Conv2D(out_ch, 3, padding="same")(yb)
            yb = layers.BatchNormalization()(yb)
            yb = layers.ReLU()(yb)
        else:
            yb = layers.Conv2DTranspose(out_ch, 2, strides=2)(yb)
            yb = layers.BatchNormalization()(yb)
            yb = layers.ReLU()(yb)
        # Optional alignment to a specific spatial size before skip
        if align_to is not None:
            ah, aw = align_to
            yb = layers.Resizing(ah, aw, interpolation="bilinear")(yb)
        if unet_skips and skip_tensor is not None:
            yb = layers.Concatenate()([yb, skip_tensor])
            yb = layers.Conv2D(out_ch, 3, padding="same")(yb)
            yb = layers.BatchNormalization()(yb)
            yb = layers.ReLU()(yb)
        return yb

    # We use consistent U-Net-style skips for clean shapes:
    # up1 aligns with x3 (H/4, W/4), up2 aligns with x2 (H/2, W/2)
    y = up_block(y, channels[1], skip_tensor=x3 if unet_skips else None, align_to=(height // 4, width // 4))
    y = up_block(y, channels[0], skip_tensor=x2 if unet_skips else None, align_to=(height // 2, width // 2))

    # Final to 1 channel at exact HxW
    if upsample_conv:
        y = layers.Resizing(height, width, interpolation="bilinear")(y)
        out = layers.Conv2D(1, 3, padding="same", name="recon")(y)
    else:
        # Two transposed-conv ups already performed; last step to HxW may still be off
        # Use resize to enforce exact dims, then 1x1 conv
        y = layers.Resizing(height, width, interpolation="bilinear")(y)
        out = layers.Conv2D(1, 3, padding="same", name="recon")(y)

    if refine:
        r = layers.Conv2D(max(8, channels[0] // 2), 3, padding="same")(out)
        r = layers.ReLU()(r)
        r = layers.Conv2D(1, 3, padding="same")(r)
        out = layers.Add(name="refined")([out, r])

    return keras.Model(inputs, out, name="ImprovedAutoencoder_Keras")


def main():
    parser = argparse.ArgumentParser(description="Keras+visualkeras autoencoder diagram generator")
    parser.add_argument("--height", type=int, default=121)
    parser.add_argument("--width", type=int, default=104)
    parser.add_argument("--channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--unet-skips", dest="unet_skips", action="store_true")
    parser.add_argument("--upsample-conv", dest="upsample_conv", action="store_true")
    parser.add_argument("--se-blocks", dest="se_blocks", action="store_true")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--out", default=None, help="Output filename (png). Default: plots/visualkeras_autoencoder_<ts>.png")
    parser.add_argument("--mirror-pytorch", action="store_true", help="Read channels/latent/toggles from Bowhead_Train_Autoencoder_Fresh.py globals")
    parser.add_argument("--export-layers", action="store_true", help="Also export per-layer details (CSV + Markdown)")
    parser.add_argument("--no-colors", action="store_true", help="Disable color mapping in layered view")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Optionally mirror PyTorch globals from the training script
    if args.mirror_pytorch:
        pt_path = os.path.join(REPO_ROOT, 'Pytorch_scripts', 'Bowhead_Train_Autoencoder_Fresh.py')
        try:
            with open(pt_path, 'r') as f:
                src = f.read()
            def _re_bool(name):
                m = re.search(rf"^\s*{name}\s*=\s*(True|False)", src, re.M)
                return (m.group(1) == 'True') if m else None
            def _re_int(name):
                m = re.search(rf"^\s*{name}\s*=\s*(\d+)", src, re.M)
                return int(m.group(1)) if m else None
            def _re_list(name):
                m = re.search(rf"^\s*{name}\s*=\s*\[([^\]]+)\]", src, re.M)
                if not m:
                    return None
                try:
                    nums = [int(x.strip()) for x in m.group(1).split(',')]
                    return nums
                except Exception:
                    return None

            ch = _re_list('MODEL_CHANNELS')
            ld = _re_int('MODEL_LATENT_DIM')
            refine = _re_bool('USE_REFINEMENT_HEAD')
            unet = _re_bool('USE_UNET_SKIPS')
            up = _re_bool('USE_UPSAMPLE_CONV')
            se = _re_bool('USE_SE_BLOCKS')

            if ch and len(ch) == 3:
                args.channels = ch
            if ld is not None:
                args.latent = ld
            if refine is True:
                args.refine = True
            if unet is True:
                args.unet_skips = True
            if up is True:
                args.upsample_conv = True
            if se is True:
                args.se_blocks = True
            print(f"[info] Mirrored PyTorch globals from {pt_path}: channels={args.channels}, latent={args.latent}, refine={args.refine}, unet_skips={args.unet_skips}, upsample_conv={args.upsample_conv}, se_blocks={args.se_blocks}")
        except Exception as e:
            print(f"[info] Failed to mirror PyTorch globals: {e}. Proceeding with CLI defaults.")

    # Build Keras model
    model = build_keras_autoencoder(
        height=args.height,
        width=args.width,
        channels=tuple(args.channels),
        latent_dim=args.latent,
        refine=bool(args.refine),
        unet_skips=bool(args.unet_skips),
        upsample_conv=bool(args.upsample_conv),
        se_blocks=bool(args.se_blocks),
    )

    # Ensure model is built (some TF/Keras versions need explicit build for shapes)
    try:
        model.build((None, args.height, args.width, 1))
    except Exception:
        pass

    # Helper: monkey-patch missing output_shape for compatibility with visualkeras on Keras 3
    def _ensure_layer_output_shapes(m):
        try:
            import tensorflow as tf  # noqa: F401
        except Exception:
            return
        for lyr in m.layers:
            # Determine if output_shape attribute is usable
            has_attr = False
            try:
                has_attr = hasattr(lyr, 'output_shape') and getattr(lyr, 'output_shape') is not None
            except Exception:
                has_attr = False

            if has_attr:
                continue

            # Try to derive from layer.output, falling back to input properties for InputLayer
            try:
                def _shape_to_tuple(s):
                    if s is None:
                        return None
                    # TensorShape or similar
                    if hasattr(s, 'as_list'):
                        try:
                            lst = s.as_list()
                            return tuple(lst) if lst is not None else None
                        except Exception:
                            pass
                    try:
                        return tuple(s)
                    except Exception:
                        return None

                out = getattr(lyr, 'output', None)
                shape = None
                if out is not None:
                    if isinstance(out, (list, tuple)):
                        shape = [_shape_to_tuple(getattr(t, 'shape', None)) for t in out]
                    else:
                        shape = _shape_to_tuple(getattr(out, 'shape', None))
                if shape is None:
                    # Special handling for InputLayer
                    bi = getattr(lyr, 'batch_input_shape', None)
                    if bi is None:
                        bi = getattr(lyr, 'input_shape', None)
                    if bi is not None:
                        shape = tuple(bi)
                if shape is not None:
                    try:
                        setattr(lyr, 'output_shape', shape)
                    except Exception:
                        pass
            except Exception:
                continue

    _ensure_layer_output_shapes(model)

    # Try to render with visualkeras
    try:
        _keras, _layers, visualkeras = _import_tf_and_vk()
        # Suppress deprecation noise from visualkeras internals
        warnings.filterwarnings("ignore", message=r".*legend_text_spacing_offset.*")
        color_map = None
        if not args.no_colors:
            # Build a color map for clarity by layer type
            color_map = {
                _layers.InputLayer: '#FFFFFF',
                _layers.Conv2D: '#FFB347',
                _layers.Conv2DTranspose: '#FFB347',
                _layers.BatchNormalization: '#FF6961',
                _layers.ReLU: '#77DD77',
                _layers.MaxPooling2D: '#AEC6CF',
                _layers.UpSampling2D: '#AEC6CF',
                _layers.Resizing: '#AEC6CF',
                _layers.Flatten: '#CFCFC4',
                _layers.Dense: '#C23B22',
                _layers.Add: '#B39EB5',
                _layers.Concatenate: '#B39EB5',
                _layers.GlobalAveragePooling2D: '#FDFD96',
            }
        img = visualkeras.layered_view(
            model,
            legend=not args.no_legend,
            draw_volume=not args.no_volume,
            color_map=color_map,
        )
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = args.out or os.path.join(PLOTS_DIR, f"visualkeras_autoencoder_{ts}.png")
        img.save(out_path)
        print(f"Saved visualkeras diagram: {out_path}")
    except Exception as e:
        print(f"[info] layered_view failed: {e}. Trying graph_view fallback...")
        try:
            _keras, _layers, visualkeras = _import_tf_and_vk()
            # graph_view has a smaller, stable signature; don't pass unsupported kwargs
            img = visualkeras.graph_view(model)
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = args.out or os.path.join(PLOTS_DIR, f"visualkeras_autoencoder_graph_{ts}.png")
            img.save(out_path)
            print(f"Saved visualkeras graph diagram: {out_path}")
        except Exception as e2:
            print(f"[error] Failed to generate visualkeras diagram via graph_view as well: {e2}")

    # Optional: write a simple summary
    try:
        summary_path = os.path.join(PLOTS_DIR, "keras_model_summary.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda s: f.write(s + "\n"))
        print(f"Saved model summary: {summary_path}")
    except Exception as e:
        print(f"[info] Failed to write model summary: {e}")

    # Optional: export detailed per-layer table
    if args.export_layers:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(PLOTS_DIR, f"keras_layers_details_{ts}.csv")
        out_md = os.path.join(PLOTS_DIR, f"keras_layers_details_{ts}.md")
        try:
            export_layer_details(model, out_csv, out_md)
        except Exception as e:
            print(f"[info] Failed to export layer details: {e}")


if __name__ == "__main__":
    main()

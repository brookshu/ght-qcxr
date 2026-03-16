# --- 1. SILENCE LOGS (Must be at the very top) ---
import warnings
import logging
import os

warnings.simplefilter("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pydicom
from PIL import Image, ImageOps

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import mlx.core as mx

# Workaround load path
from mlx_embeddings.utils import get_model_path, load_model
from transformers import AutoProcessor


# --- CONFIGURATION ---
MODEL_ID = "mlx-community/medsiglip-448"
TEST_FILE = "cxr_test/rotated.dcm"
TEST_FILE = "../frommartin/rotated/1000049E.dcm"

def process_dicom(dicom_path: str) -> Optional[Image.Image]:
    """DICOM -> RGB Image with robust normalization + MONOCHROME1 invert."""
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)

        pmin, pmax = float(pixel_array.min()), float(pixel_array.max())
        if pmax > pmin:
            scaled = (pixel_array - pmin) / (pmax - pmin) * 255.0
        else:
            scaled = np.zeros_like(pixel_array, dtype=np.float32)

        img = Image.fromarray(np.uint8(np.clip(scaled, 0, 255)))

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = ImageOps.invert(img)

        return img.convert("RGB")
    except Exception as e:
        print(f"❌ DICOM Processing Error: {e}")
        return None


def load_medsiglip_fixed(model_id: str):
    """
    mlx-embeddings has a siglip repo-name regex assumption.
    Bypass load() and pass a fake repo-name that matches patch14-448.
    """
    print(f"🧠 Downloading model snapshot: {model_id}")
    model_path = get_model_path(model_id)

    fake_path_to_repo = "siglip-so400m-patch14-448"
    print("🧠 Loading model (workaround path_to_repo = siglip-so400m-patch14-448)")
    model = load_model(model_path, path_to_repo=fake_path_to_repo)

    processor = AutoProcessor.from_pretrained(model_path)

    print("✅ Model + processor loaded")
    return model, processor


def score_prompts(model, processor, image: Image.Image, prompts: List[str]) -> mx.array:
    """
    Returns probabilities (sigmoid(logits_per_image)) for each prompt, shape [N].
    Follows mlx-embeddings README: processor return_tensors='pt' then convert to mx. :contentReference[oaicite:2]{index=2}
    """
    try:
        import torch  # required by return_tensors="pt"
    except Exception:
        raise RuntimeError(
            "PyTorch is required because processor(..., return_tensors='pt') is used. "
            "Install it with: pip install torch"
        )

    # NOTE: padding="max_length" is what the README uses for SigLIP. :contentReference[oaicite:3]{index=3}
    inputs = processor(text=prompts, images=image, padding="max_length", return_tensors="pt")

    # Convert processor outputs to MLX arrays (channel-last pixel_values)
    pixel_values = mx.array(inputs.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)  # [B, H, W, C]
    input_ids = mx.array(inputs.input_ids)

    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    probs = mx.sigmoid(outputs.logits_per_image)[0]  # [N]
    return probs


def pick_label(model, processor, image: Image.Image, labels: List[str], prompts: List[str]) -> str:
    probs = score_prompts(model, processor, image, prompts)
    best_idx = int(mx.argmax(probs).item())
    return labels[best_idx]


def analyze_cxr_medsiglip(model, processor, image_path: str) -> Dict[str, Any]:
    image = process_dicom(image_path) if image_path.lower().endswith(".dcm") else None
    if image is None:
        raise RuntimeError(f"Could not process image: {image_path}")

    proj_labels = ["PA", "AP"]
    proj_prompts = ["chest x-ray PA projection", "chest x-ray AP projection"]

    insp_labels = ["Under-inflated", "Adequate", "Hyper-inflated"]
    insp_prompts = [
        "chest x-ray low lung volumes underinflated poor inspiration",
        "chest x-ray adequate inspiration normal lung volumes",
        "chest x-ray hyperinflated lungs high lung volumes",
    ]

    exp_labels = ["Under-exposed", "Adequate", "Over-exposed"]
    exp_prompts = [
        "underexposed chest x-ray low penetration too white",
        "adequately exposed chest x-ray appropriate penetration",
        "overexposed chest x-ray high penetration too dark",
    ]

    rot_labels = ["No", "Yes"]
    rot_prompts = [
        "chest x-ray not rotated symmetric clavicles",
        "chest x-ray rotated asymmetric clavicles",
    ]

    dq_labels = ["Non-diagnostic", "Suboptimal", "Optimal"]
    dq_prompts = [
        "nondiagnostic chest x-ray poor quality cannot interpret",
        "suboptimal chest x-ray limited quality borderline",
        "optimal chest x-ray high quality diagnostic",
    ]

    projection = pick_label(model, processor, image, proj_labels, proj_prompts)
    inspiration = pick_label(model, processor, image, insp_labels, insp_prompts)
    exposure = pick_label(model, processor, image, exp_labels, exp_prompts)
    rotation_str = pick_label(model, processor, image, rot_labels, rot_prompts)
    diagnostic_quality = pick_label(model, processor, image, dq_labels, dq_prompts)

    return {
        "rotation_detected": (rotation_str == "Yes"),
        "inspiration": inspiration,
        "projection": projection,
        "exposure": exposure,
        "diagnostic_quality": diagnostic_quality,
        "findings": (
            "MedSigLIP is an embedding model (not generative). "
            "For narrative findings, use a generative radiology VLM or retrieval."
        ),
    }


def pretty_print(image_path: str, data: Dict[str, Any], t: float) -> None:
    console = Console()
    console.print(f"\n[bold blue]👀 Analyzing Image:[/bold blue] {image_path}")

    table = Table(title="CXR Technical Quality (MedSigLIP Zero-shot)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Assessment", style="green")

    rotation_str = "[red]DETECTED[/red]" if data.get("rotation_detected") else "None"
    table.add_row("Projection", str(data.get("projection", "N/A")))
    table.add_row("Rotation", rotation_str)
    table.add_row("Inspiration", str(data.get("inspiration", "N/A")))
    table.add_row("Exposure", str(data.get("exposure", "N/A")))
    table.add_row("Diagnostic Quality", str(data.get("diagnostic_quality", "N/A")))

    console.print("\n")
    console.print(table)

    console.print(Panel(
        Text(str(data.get("findings", "")), justify="left"),
        title="[bold yellow]Clinical Findings[/bold yellow]",
        border_style="yellow"
    ))
    console.print(f"\n[dim]⏱️  Inference Time: {t:.2f}s[/dim]\n")


if __name__ == "__main__":
    print("🚀 GHT-qCXR MVP (MedSigLIP / MLX-Embeddings): Initializing on M4...")

    if not os.path.exists(TEST_FILE):
        print(f"⚠️  Test file '{TEST_FILE}' not found.")
        sys.exit(0)

    model, processor = load_medsiglip_fixed(MODEL_ID)

    start = time.time()
    result = analyze_cxr_medsiglip(model, processor, TEST_FILE)
    end = time.time()

    # Strict JSON (no markdown)
    print(json.dumps(result, ensure_ascii=False))

    # Pretty print like your original
    pretty_print(TEST_FILE, result, end - start)
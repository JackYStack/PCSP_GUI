# -*- coding: utf-8 -*-
"""Streamlit web app for Sandwich Panel Predictor.

Run:
  streamlit run scripts/streamlit_web_app.py
"""

from __future__ import annotations

import io
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


APP_TITLE = "Sandwich Panel Predictor (Web)"
EXCLUDED_BUILTIN_MODELS = {"composite_model.pkl"}

FEATURE_META: Dict[str, Dict[str, float | str]] = {
    "Width": {"symbol": "W", "unit": "mm", "min": 300.00, "max": 600.00},
    "LeafThickness": {"symbol": "L", "unit": "mm", "min": 50.00, "max": 70.00},
    "ConcreteStrength": {"symbol": "fc", "unit": "MPa", "min": 20.13, "max": 33.56},
    "RebarRatio": {"symbol": "R", "unit": "%", "min": 0.50, "max": 0.90},
    "InsulationThickness": {"symbol": "I", "unit": "mm", "min": 30.00, "max": 80.00},
    "S": {"symbol": "S", "unit": "mm", "min": 300.00, "max": 600.00},
    "ConnectorStiffness": {"symbol": "K", "unit": "N/mm", "min": 2000.00, "max": 15000.00},
    "ConnectorCapacity": {"symbol": "F", "unit": "kN", "min": 5000.00, "max": 25000.00},
}


def load_gui_feature_meta() -> Dict[str, Dict[str, float | str]]:
    """Load FEATURE_META from gui_predictor.py for behavior consistency.

    Falls back to local FEATURE_META if import fails.
    """
    gui_file = project_root() / "scripts" / "gui_predictor.py"
    if not gui_file.exists():
        return FEATURE_META

    try:
        spec = importlib.util.spec_from_file_location("gui_predictor_module", str(gui_file))
        if spec is None or spec.loader is None:
            return FEATURE_META
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = getattr(module, "FEATURE_META", None)
        if isinstance(meta, dict) and meta:
            return meta
    except Exception:
        return FEATURE_META

    return FEATURE_META


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_paths() -> None:
    root = project_root()
    for p in (root, root / "src"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _register_src_scaling_policy_compat() -> None:
    """Register minimal compatibility module for unpickling old/new models.

    Used when runtime cannot import `src.scaling_policy`.
    """
    import types

    if "src.scaling_policy" in sys.modules:
        return

    src_module = sys.modules.get("src")
    if src_module is None:
        src_module = types.ModuleType("src")
        sys.modules["src"] = src_module

    scaling_module = types.ModuleType("src.scaling_policy")

    class RegressorPipeline(Pipeline):
        def _more_tags(self):  # pragma: no cover
            tags = super()._more_tags()
            tags = dict(tags) if isinstance(tags, dict) else {}
            tags["estimator_type"] = "regressor"
            return tags

        def __sklearn_tags__(self):  # pragma: no cover
            tags = super().__sklearn_tags__()
            try:
                if isinstance(tags, dict):
                    new_tags = dict(tags)
                    new_tags["estimator_type"] = "regressor"
                    return new_tags
                tags.estimator_type = "regressor"
                return tags
            except Exception:
                return {"estimator_type": "regressor"}

    setattr(scaling_module, "RegressorPipeline", RegressorPipeline)
    setattr(src_module, "scaling_policy", scaling_module)
    sys.modules["src.scaling_policy"] = scaling_module


def read_features_from_config(config_path: Path) -> Tuple[List[str], List[str]]:
    if yaml is None or not config_path.exists():
        return [], []
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        features = cfg.get("features", {}).get("input_features", []) or []
        targets = cfg.get("features", {}).get("target_features", []) or []
        return list(features), list(targets)
    except Exception:
        return [], []


@st.cache_resource(show_spinner=False)
def load_model_from_file(model_file: str):
    ensure_paths()
    try:
        obj = joblib.load(model_file)
    except ModuleNotFoundError as exc:
        if "src" in str(exc):
            _register_src_scaling_policy_compat()
            obj = joblib.load(model_file)
        else:
            raise
    if isinstance(obj, dict):
        return {
            "model": obj.get("model"),
            "features": list(obj.get("features") or []),
            "targets": list(obj.get("targets") or []),
            "best_model": obj.get("best_model", ""),
        }
    return {"model": obj, "features": [], "targets": [], "best_model": ""}


def load_uploaded_model(uploaded_bytes: bytes):
    ensure_paths()
    buffer = io.BytesIO(uploaded_bytes)
    try:
        obj = joblib.load(buffer)
    except ModuleNotFoundError as exc:
        if "src" in str(exc):
            _register_src_scaling_policy_compat()
            buffer.seek(0)
            obj = joblib.load(buffer)
        else:
            raise
    if isinstance(obj, dict):
        return {
            "model": obj.get("model"),
            "features": list(obj.get("features") or []),
            "targets": list(obj.get("targets") or []),
            "best_model": obj.get("best_model", ""),
        }
    return {"model": obj, "features": [], "targets": [], "best_model": ""}


def ordered_union_features(feature_lists: List[List[str]]) -> List[str]:
    merged = []
    for f_list in feature_lists:
        for f in f_list:
            if f not in merged:
                merged.append(f)
    return merged


def display_name(feature: str) -> str:
    return "Spacement" if feature == "S" else feature


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    root = project_root()
    feature_meta = load_gui_feature_meta()
    models_dir = root / "models"
    cfg_features, cfg_targets = read_features_from_config(root / "configs" / "config_optimized.yaml")
    if not cfg_features:
        cfg_features, cfg_targets = read_features_from_config(root / "configs" / "config.yaml")

    built_in_models = sorted([p for p in models_dir.glob("*.pkl") if p.name not in EXCLUDED_BUILTIN_MODELS])

    with st.sidebar:
        st.header("Model Selection")
        model_names = [p.name for p in built_in_models]
        default_models = model_names[:1]
        selected_model_names = st.multiselect("Compare Models", model_names, default=default_models)
        uploaded = st.file_uploader("Import Custom Model (.pkl)", type=["pkl"])

    if not selected_model_names and not uploaded:
        st.info("Please select at least one model or upload one custom model.")
        return

    model_registry: Dict[str, Dict] = {}

    for name in selected_model_names:
        model_path = str(models_dir / name)
        try:
            model_registry[name] = load_model_from_file(model_path)
        except Exception as exc:
            st.error(f"Failed to load {name}: {exc}")

    if uploaded is not None:
        try:
            upload_name = uploaded.name or "uploaded_model.pkl"
            model_registry[upload_name] = load_uploaded_model(uploaded.getvalue())
        except Exception as exc:
            st.error(f"Failed to load uploaded model: {exc}")

    if not model_registry:
        st.error("No valid model could be loaded.")
        return

    model_feature_lists = [m.get("features") or [] for m in model_registry.values()]
    required_features = ordered_union_features(model_feature_lists)
    if not required_features:
        required_features = cfg_features

    target_names = []
    for m in model_registry.values():
        ts = m.get("targets") or []
        if ts:
            target_names = ts
            break
    if not target_names:
        target_names = cfg_targets or ["StiffnessComposite", "CapacityComposite"]

    st.caption(
        "Outputs: StiffnessComposite (%) and CapacityComposite (%)"
    )

    tab_manual, tab_csv = st.tabs(["Manual Input", "CSV Input"])

    input_df = None
    with tab_manual:
        st.subheader("Manual Input")
        cols = st.columns(2)
        values = {}
        for i, feature in enumerate(required_features):
            meta = feature_meta.get(feature, {})
            min_v = float(meta.get("min", 0.0))
            max_v = float(meta.get("max", 1.0))
            default_v = (min_v + max_v) / 2 if feature in feature_meta else 0.0
            unit = str(meta.get("unit", ""))
            symbol = str(meta.get("symbol", ""))
            label = f"{display_name(feature)} ({symbol}, {unit})" if unit else display_name(feature)
            with cols[i % 2]:
                values[feature] = st.number_input(label, value=float(default_v), format="%.4f")

        if st.button("Predict (Manual)", type="primary"):
            input_df = pd.DataFrame([values], columns=required_features)

    with tab_csv:
        st.subheader("CSV Input")
        st.markdown(
            "**CSV format requirement:** include all required numeric columns:  "
            + ", ".join(required_features)
        )
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        if uploaded_csv is not None:
            try:
                df_csv = pd.read_csv(uploaded_csv)
                st.dataframe(df_csv.head(20), use_container_width=True)
                missing = sorted(set(required_features) - set(df_csv.columns))
                if missing:
                    st.error(f"CSV missing required columns: {missing}")
                else:
                    if st.button("Predict (CSV Batch)", type="primary"):
                        input_df = df_csv.loc[:, required_features].copy()
            except Exception as exc:
                st.error(f"CSV read failed: {exc}")

    if input_df is None:
        return

    result_rows = []
    combined_df = input_df.copy()

    for model_name, payload in model_registry.items():
        model = payload.get("model")
        features = payload.get("features") or required_features
        targets = payload.get("targets") or target_names

        missing_for_model = sorted(set(features) - set(input_df.columns))
        if missing_for_model:
            st.warning(f"Skip {model_name}: missing columns {missing_for_model}")
            continue

        try:
            X_input = input_df[features].values
            preds = model.predict(X_input)
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            for i, t in enumerate(targets):
                combined_df[f"Predicted_{t}_{model_name}"] = preds[:, i]

            if preds.shape[0] == 1:
                stiff = float(preds[0, 0]) if preds.shape[1] > 0 else float("nan")
                cap = float(preds[0, 1]) if preds.shape[1] > 1 else float("nan")
            else:
                stiff = float(np.mean(preds[:, 0])) if preds.shape[1] > 0 else float("nan")
                cap = float(np.mean(preds[:, 1])) if preds.shape[1] > 1 else float("nan")

            result_rows.append(
                {
                    "Model": model_name,
                    "StiffnessComposite (%)": stiff,
                    "CapacityComposite (%)": cap,
                }
            )
        except Exception as exc:
            st.error(f"Prediction failed for {model_name}: {exc}")

    if not result_rows:
        st.error("No prediction results available.")
        return

    result_df = pd.DataFrame(result_rows)
    if len(result_df) > 1:
        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    [
                        {
                            "Model": "Average",
                            "StiffnessComposite (%)": float(result_df["StiffnessComposite (%)"].mean()),
                            "CapacityComposite (%)": float(result_df["CapacityComposite (%)"].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    st.subheader("Prediction Results")
    st.dataframe(result_df, use_container_width=True)

    csv_bytes = combined_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="predictions_web.csv",
        mime="text/csv",
    )

    st.caption("This GUI is developed by College of Civil Engineering, Hefei University of Technology, Hefei 230009, China")


if __name__ == "__main__":
    main()

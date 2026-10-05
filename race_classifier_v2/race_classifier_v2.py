#!/usr/bin/env python
# Race classifier v2: name (ethnicolr2) + face image (SigLIP2 + FairFace probe) -> LightGBM combiner.
# Drop-in successor to race_classifier_fbhgs.py.
#
# Usage:  python race_classifier_v2.py input_folder/ output_folder/ [--model unweighted] [--sort]
#
# input_folder holds images named firstname_lastname_id.jpg. For each image:
#   1. parse first/last name from the filename (every token between the first and the trailing numeric id is
#      kept as part of the surname, so multi-part surnames are no longer glued together or dropped)
#   2. ethnicolr2 Florida full-name model -> 4 probs + "other / no prediction" flag
#   3. SigLIP2 so400m embedding of a FairFace-style face chip (whole image if no face) -> FairFace 7-race
#      linear probe -> collapsed to 4 probs
#   4. the 9 features -> combiner model -> final 4-class probabilities + argmax label
#   5. one CSV row per image in output_folder/race_predictions.csv
#   6. --sort: also copy each image into output_folder/<predicted_label>/ (unreadable files into unreadable/)
#
# Unreadable image files: the sig_* features are passed to the combiner as NaN (never zeros). Its training set
# had 337 founders without a usable image, so LightGBM learned a direction for missing values at every sig_*
# split, and the row is scored on the name alone. Masking the image features of all 32,084 validation rows
# this way gives 86.1% overall / 50.1% Black accuracy (default model), close to a separately trained
# name-only model (87.7% / 41.2%). Such rows are marked basis=name_only in race_predictions.csv.
import os
import sys
import shutil
import argparse

import dlib  # must be imported before torch (pulled in by ethnicolr2 / transformers): a CUDA build of dlib
             # has to load its own CUDA libraries first
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
COMBINERS = {"sqrt-balanced": "combiner_model.joblib", "unweighted": "combiner_model_unweighted.joblib"}
SIGLIP = "google/siglip2-so400m-patch14-384"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSES = ["white", "asian", "black", "hispanic"]  # combiner output order
# FairFace 7 races -> 4 target classes (same collapse used to build the combiner's training features)
MAP4 = {"white": ["white", "middle_eastern"], "asian": ["east_asian", "southeast_asian", "indian"],
        "black": ["black"], "hispanic": ["latino_hispanic"]}
MAX_SIZE = 800  # FairFace/predict.py resizes the longer side to 800 before detection


# ---------------------------------------------------------------- 1. names
def parse_filename(fn):
    """firstname_lastname_id.jpg -> (first, last). Tokens between the first and the numeric id all belong to
    the surname (e.g. maria_de_la_cruz_123.jpg -> "maria", "de la cruz"). A name-only stem (kenma_123.jpg)
    is treated as a surname with an empty first name."""
    parts = [p for p in os.path.splitext(fn)[0].split("_") if p]
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], " ".join(parts[1:])


def name_features(df):
    """ethnicolr2 pred_fl_full_name -> eth_* columns."""
    from ethnicolr2 import pred_fl_full_name
    out = pred_fl_full_name(df[["first_name", "last_name"]].copy(), lname_col="last_name", fname_col="first_name")
    p = pd.DataFrame([{k: float(v) for k, v in d.items()} if isinstance(d, dict) else {} for d in out["probs"]],
                     index=out.index).reindex(columns=["nh_white", "asian", "nh_black", "hispanic"])
    f = pd.DataFrame({"eth_white": p.nh_white, "eth_asian": p.asian, "eth_black": p.nh_black,
                      "eth_hispanic": p.hispanic}, index=df.index)
    # 1 if ethnicolr2 gave no prediction or its argmax was "other"
    f["eth_other_or_missing"] = (out.preds.isna() | (out.preds == "other") | p.isna().any(axis=1)).astype(int)
    return f


# ---------------------------------------------------------------- 2. images
def face_or_image(path, det, sp):
    """FairFace-style chip (dlib CNN, 1 upsample, largest face, 300px, padding 0.25) of the image resized to a
    longer side of 800; the resized whole image when no face is found; None if the file can't be read."""
    try:
        img = dlib.load_rgb_image(path)
    except Exception as e:
        print(f"  could not read {os.path.basename(path)}: {e}", file=sys.stderr)
        return None, "error"
    h, w = img.shape[:2]
    nw, nh = (MAX_SIZE, int(MAX_SIZE * h / w)) if w > h else (int(MAX_SIZE * w / h), MAX_SIZE)
    img = dlib.resize_image(img, rows=nh, cols=nw)
    dets = det(img, 1)
    if len(dets) == 0:
        return img, "none"
    faces = dlib.full_object_detections()
    faces.append(sp(img, max(dets, key=lambda d: d.rect.area()).rect))
    return dlib.get_face_chips(img, faces, size=300, padding=0.25)[0], "cnn"


def image_features(paths, batch):
    """SigLIP2 embedding + FairFace probe -> sig_* columns (NaN for unreadable images) and face_detection."""

    import joblib
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor

    try:
        det = dlib.cnn_face_detection_model_v1(os.path.join(MODELS, "dlib", "mmod_human_face_detector.dat"))
    except RuntimeError as e:
        sys.exit(f"dlib face detector failed to start ({e}).\nA CUDA build of dlib needs a visible GPU; on a "
                 "CPU-only machine install the CPU build (pip install dlib).")
    sp = dlib.shape_predictor(os.path.join(MODELS, "dlib", "shape_predictor_5_face_landmarks.dat"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(SIGLIP, dtype=dtype).to(dev).eval()
    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    probe = joblib.load(os.path.join(MODELS, "siglip2_probe.joblib"))
    print(f"SigLIP2 on {dev}", flush=True)

    emb = np.full((len(paths), model.config.vision_config.hidden_size), np.nan, dtype=np.float32)
    how = []
    for s in range(0, len(paths), batch):
        imgs = []
        for i, p in enumerate(paths[s:s + batch]):
            im, d = face_or_image(p, det, sp)
            how.append(d)
            if im is not None:
                imgs.append((s + i, im))
        if imgs:
            px = proc(images=[Image.fromarray(im) for _, im in imgs], return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                f = model.get_image_features(pixel_values=px.to(dev, dtype))
                f = getattr(f, "pooler_output", f)
                # the probe was trained on float16-stored embeddings
                f = torch.nn.functional.normalize(f.float(), dim=-1).half().float().cpu().numpy()
            emb[[i for i, _ in imgs]] = f
        print(f"  images {min(s + batch, len(paths))}/{len(paths)}", flush=True)

    ok = ~np.isnan(emb).any(axis=1)
    p7 = pd.DataFrame(np.nan, index=range(len(paths)), columns=probe["classes"])
    if ok.any():
        P = probe["clf"].predict_proba(emb[ok])
        for j, c in enumerate(probe["clf"].classes_):
            p7.loc[ok, probe["classes"][c]] = P[:, j]
    f = pd.DataFrame({f"sig_{k}": p7[parts].sum(axis=1, min_count=1) for k, parts in MAP4.items()})
    f["face_detection"] = how
    return f


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Race classifier v2: ethnicolr2 + SigLIP2 -> LightGBM combiner")
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    ap.add_argument("--model", default="sqrt-balanced",
                    help="combiner: 'sqrt-balanced' (default, best per-person labels), 'unweighted' (better for "
                         "group shares), or a path to a .joblib")
    ap.add_argument("--sort", action="store_true", help="also copy each image into output_folder/<label>/")
    ap.add_argument("--names", help="optional CSV with columns filename,first_name,last_name to use instead of "
                                    "parsing names from filenames (e.g. to keep spaces in multi-part surnames)")
    ap.add_argument("--batch", type=int, default=16, help="SigLIP2 batch size")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.input_folder) if f.lower().endswith(IMG_EXT))
    if not files:
        sys.exit(f"no images found in {args.input_folder}")
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"{len(files)} images", flush=True)

    df = pd.DataFrame({"filename": files})
    df[["first_name", "last_name"]] = [parse_filename(f) for f in files]
    if args.names:
        n = pd.read_csv(args.names, dtype=str, keep_default_na=False).set_index("filename")
        has = df.filename.isin(n.index)
        df.loc[has, "first_name"] = df.filename[has].map(n.first_name)
        df.loc[has, "last_name"] = df.filename[has].map(n.last_name)
        print(f"names from {args.names} for {has.sum()} images", flush=True)

    print("1/3 ethnicolr2 (names)", flush=True)
    eth = name_features(df)
    print("2/3 SigLIP2 (images)", flush=True)
    sig = image_features([os.path.join(args.input_folder, f) for f in files], args.batch)

    print("3/3 combiner", flush=True)
    import joblib
    path = os.path.join(MODELS, COMBINERS[args.model]) if args.model in COMBINERS else args.model
    comb = joblib.load(path)
    X = pd.concat([eth, sig], axis=1)[comb["features"]]
    sig_cols = [c for c in X.columns if c.startswith("sig_")]
    no_img = X[sig_cols].isna().all(axis=1)
    # image features are all present or all NaN (whole-embedding failure), never partly missing
    assert (no_img == X[sig_cols].isna().any(axis=1)).all()
    assert (no_img == (sig.face_detection == "error")).all()
    P = comb["model"].predict_proba(X)
    probs = pd.DataFrame(P, columns=[f"prob_{c}" for c in comb["classes"]])

    out = pd.concat([df, probs[[f"prob_{c}" for c in CLASSES]]], axis=1)
    out["predicted_label"] = [comb["classes"][i] for i in P.argmax(axis=1)]
    # name_only = image unreadable, label comes from the name features alone (see the note at the top)
    out["basis"] = np.where(no_img, "name_only", "name+image")
    csv = os.path.join(args.output_folder, "race_predictions.csv")
    out.to_csv(csv, index=False)
    # inputs to the combiner, for auditing (no-face / unreadable images, missing name predictions)
    pd.concat([df.filename, X, sig.face_detection], axis=1).to_csv(
        os.path.join(args.output_folder, "race_features.csv"), index=False)

    if args.sort:
        # copy, never move: input images may be someone's only copy
        # unreadable files go to unreadable/, not a label folder: their label is name-only and the file can't
        # be viewed anyway
        for f, lab, b in zip(out.filename, out.predicted_label, out.basis):
            sub = "unreadable" if b == "name_only" else lab
            os.makedirs(os.path.join(args.output_folder, sub), exist_ok=True)
            shutil.copy2(os.path.join(args.input_folder, f), os.path.join(args.output_folder, sub, f))

    print(f"wrote {len(out)} rows -> {csv}  (model: {os.path.basename(path)})")
    print(out.predicted_label.value_counts().to_string())
    nf = (sig.face_detection == "none").sum()
    if nf:
        print(f"note: {nf} image(s) had no detected face (whole image used)")
    if no_img.any():
        print(f"WARNING: {no_img.sum()} image file(s) could not be read; labelled from the name only "
              f"(basis=name_only): {', '.join(out.filename[no_img])}")


if __name__ == "__main__":
    main()

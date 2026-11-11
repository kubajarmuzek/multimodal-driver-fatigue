import glob
import json
import os
import numpy as np
import pandas as pd


BASE_DIR = "nagrania"  # Folder z sesjami kierowców
OUTPUT_FILE = "dataset_ready_for_ml.csv"

WINDOW_SIZE_SEC = 60  # Długość okna (sekundy)
STEP_SIZE_SEC = 5  # Przesunięcie okna (krok predykcji)

MAX_MISSING_PCT_PER_WINDOW = 0.30  # Odrzuć okno jeśli >30% NaN (utrata śledzenia)
MAX_BLINK_DURATION_SEC = 1.5  # Zamknięte oczy >3s = artefakt / zasłonięcie
MAX_YAWN_DURATION_SEC = 6.0  # Otwarta buzia >8s = rozmowa / szum

EAR_THRESHOLD = 0.20  # Próg zamknięcia oka
MAR_THRESHOLD = 0.50  # Próg ziewnięcia


def get_driver_id_from_metadata(session_dir):
    """Odczytuje driver_id z pliku metadata.json w folderze sesji."""
    json_path = os.path.join(session_dir, "metadata.json")
    session_id = os.path.basename(session_dir)

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "driver_id" in data:
                    return str(data["driver_id"])
        except Exception as e:
            print(
                f"   [OSTRZEŻENIE] Błąd odczytu {json_path}: {e}. Używam fallbacku z nazwy folderu."
            )

    return session_id.split("_")[0] if "_" in session_id else session_id


def process_single_session(session_dir):
    """Przetwarza surowe pliki sesji (video_features.csv, hr_rr.csv, frame_labels.csv):

    - odczytuje driver_id z metadata.json,
    - czyści artefakty pomiarowe z wideo,
    - dzieli na okna czasowe i wylicza zagregowane cechy biometryczne,
    - synchronizuje dane sercowe z pliku hr_rr.csv oraz etykiety z frame_labels.csv.
    """
    csv_path = os.path.join(session_dir, "video_features.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    driver_id = get_driver_id_from_metadata(session_dir)
    session_id = os.path.basename(session_dir)

    hr_csv_path = os.path.join(session_dir, "hr_rr.csv")
    df_hr = None
    if os.path.exists(hr_csv_path):
        try:
            df_hr = pd.read_csv(hr_csv_path)
            if (
                "experiment_time_s" in df_hr.columns
                and "timestamp_sec" not in df_hr.columns
            ):
                df_hr["timestamp_sec"] = df_hr["experiment_time_s"]
        except Exception as e:
            print(f"   [OSTRZEŻENIE] Błąd wczytywania {hr_csv_path}: {e}")

    labels_csv_path = os.path.join(session_dir, "frame_labels.csv")
    df_labels = None
    if os.path.exists(labels_csv_path):
        try:
            df_labels = pd.read_csv(labels_csv_path)
        except Exception as e:
            print(f"   [OSTRZEŻENIE] Błąd wczytywania {labels_csv_path}: {e}")

    df["ear_avg"] = (df["ear_left"] + df["ear_right"]) / 2.0

    # -------------------------------------------------------------
    # KROK 1: Czyszczenie artefaktów na poziomie szeregu czasowego
    # -------------------------------------------------------------
    ear_closed = df["ear_avg"] < EAR_THRESHOLD
    mar_open = df["mar"] > MAR_THRESHOLD

    ear_group = (~ear_closed).cumsum()
    mar_group = (~mar_open).cumsum()

    fps = 1.0 / df["timestamp_sec"].diff().mean() if len(df) > 1 else 30.0

    ear_durations = df.groupby(ear_group)["ear_avg"].transform("count") / fps
    df.loc[ear_closed & (ear_durations > MAX_BLINK_DURATION_SEC), "ear_avg"] = (
        np.nan
    )

    mar_durations = df.groupby(mar_group)["mar"].transform("count") / fps
    df.loc[mar_open & (mar_durations > MAX_YAWN_DURATION_SEC), "mar"] = np.nan

    # -------------------------------------------------------------
    # KROK 2: Agregacja w Okna Czasowe
    # -------------------------------------------------------------
    max_time = df["timestamp_sec"].max()
    window_records = []
    current_start = 0.0

    while current_start + WINDOW_SIZE_SEC <= max_time:
        current_end = current_start + WINDOW_SIZE_SEC

        w_df = df[
            (df["timestamp_sec"] >= current_start)
            & (df["timestamp_sec"] < current_end)
        ]
        missing_pct = w_df["ear_avg"].isna().mean()

        if missing_pct <= MAX_MISSING_PCT_PER_WINDOW and len(w_df) > 0:
            valid_w = w_df.dropna(subset=["ear_avg", "mar", "head_pitch", "head_yaw"])

            if len(valid_w) > 0:
                perclos = (valid_w["ear_avg"] < EAR_THRESHOLD).mean()
                avg_ear = valid_w["ear_avg"].mean()
                std_ear = valid_w["ear_avg"].std()

                avg_mar = valid_w["mar"].mean()
                yawn_count = (valid_w["mar"] > MAR_THRESHOLD).sum()

                std_pitch = valid_w["head_pitch"].std()
                std_yaw = valid_w["head_yaw"].std()

                kss_raw = np.nan
                kss_binary = np.nan

                if df_labels is not None and "fatigue_label" in df_labels.columns:
                    w_labels = df_labels[
                        (df_labels["timestamp_sec"] >= current_start)
                        & (df_labels["timestamp_sec"] < current_end)
                    ]
                    if len(w_labels) > 0:
                        kss_raw = round(w_labels["fatigue_label"].mean(), 2)
                        kss_binary = 1 if kss_raw >= 7.0 else 0

                record = {
                    "driver_id": driver_id,
                    "session_id": session_id,
                    "window_start": current_start,
                    "window_end": current_end,
                    "perclos": round(perclos, 4),
                    "avg_ear": round(avg_ear, 4),
                    "std_ear": round(std_ear, 4) if not np.isnan(std_ear) else 0.0,
                    "avg_mar": round(avg_mar, 4),
                    "yawn_count": yawn_count,
                    "std_pitch": (
                        round(std_pitch, 4) if not np.isnan(std_pitch) else 0.0
                    ),
                    "std_yaw": round(std_yaw, 4) if not np.isnan(std_yaw) else 0.0,
                    "target_kss_raw": kss_raw,
                    "target_kss_binary": kss_binary,
                }

                if df_hr is not None and "timestamp_sec" in df_hr.columns:
                    w_hr = df_hr[
                        (df_hr["timestamp_sec"] >= current_start)
                        & (df_hr["timestamp_sec"] < current_end)
                    ]
                    if len(w_hr) > 0:
                        if "heart_rate_bpm" in w_hr.columns:
                            record["bpm_mean"] = round(w_hr["heart_rate_bpm"].mean(), 2)

                        if "rr_ms" in w_hr.columns:
                            rr_values = w_hr["rr_ms"].dropna()
                            if len(rr_values) > 1:
                                rr_diffs = np.diff(rr_values)
                                rmssd = np.sqrt(np.mean(rr_diffs**2))
                                record["rmssd"] = round(rmssd, 2)

                window_records.append(record)

        current_start += STEP_SIZE_SEC

    return pd.DataFrame(window_records)


def apply_person_specific_normalization(df):
    """Normalizacja Z-Score dla cech z podziałem na kierowców."""
    features_to_scale = ["avg_ear", "avg_mar", "std_pitch", "std_yaw"]

    for hr_col in ["bpm_mean", "rmssd"]:
        if hr_col in df.columns:
            features_to_scale.append(hr_col)

    for col in features_to_scale:
        df[f"{col}_norm"] = df.groupby("driver_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
    return df


def run_pipeline():
    if not os.path.exists(BASE_DIR):
        print(f"BŁĄD: Folder bazowy '{BASE_DIR}' nie istnieje!")
        return

    subfolders = sorted(os.listdir(BASE_DIR))
    all_sessions_dfs = []

    print("=== ROZPOCZĘCIE PRZYGOTOWANIA DANYCH (DATA PIPELINE) ===")

    for item in subfolders:
        session_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(session_path) and not item.startswith("IGNORE"):
            print(f" Przetwarzanie sesji: {item}")
            session_df = process_single_session(session_path)

            if session_df is not None and not session_df.empty:
                all_sessions_dfs.append(session_df)

    if not all_sessions_dfs:
        print("BŁĄD: Nie przetworzono żadnej sesji!")
        return

    master_df = pd.concat(all_sessions_dfs, ignore_index=True)
    master_df = apply_person_specific_normalization(master_df)

    master_df.to_csv(OUTPUT_FILE, index=False)
    print(
        f"\n[SUKCES] Wygenerowano plik '{OUTPUT_FILE}' z {len(master_df)} próbkami okienkowymi!"
    )
    print(f"Liczba unikalnych kierowców: {master_df['driver_id'].nunique()}")
    print(f"Liczba sesji: {master_df['session_id'].nunique()}")


if __name__ == "__main__":
    run_pipeline()
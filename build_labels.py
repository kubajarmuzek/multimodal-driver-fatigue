import json
import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def process_single_session(session_dir):
    """Przetwarza jedną sesję i zapisuje plik frame_labels.csv w jej folderze."""
    print(f"\n Rozpoczynam przetwarzanie: {session_dir}")

    video_files = [
        f
        for f in os.listdir(session_dir)
        if f.lower().endswith(".mp4") and not f.startswith("._")
    ]
    if not video_files:
        print(f"   [OSTRZEŻENIE] Brak pliku .mp4 w {session_dir}. Pomijam.")
        return

    video_path = os.path.join(session_dir, video_files[0])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps == 0 or total_frames == 0:
        print(
            f"   [OSTRZEŻENIE] Błąd odczytu wideo {video_path} (FPS=0 lub klatki=0). Pomijam."
        )
        return

    print(
        f"   Wideo: {video_files[0]} | FPS: {fps:.2f} | Klatek łącznie: {total_frames}"
    )

    events_path = os.path.join(session_dir, "events.csv")
    if not os.path.exists(events_path):
        print(f"   [OSTRZEŻENIE] Brak pliku events.csv w {session_dir}. Pomijam.")
        return

    df_events = pd.read_csv(events_path)

    df_fatigue = df_events[df_events["event"] == "FATIGUE"].copy()
    if df_fatigue.empty:
        print(
            f"   [OSTRZEŻENIE] Brak zdarzeń FATIGUE w {events_path}. Pomijam."
        )
        return

    df_fatigue["value"] = df_fatigue["value"].astype(float)
    times = df_fatigue["experiment_time_s"].values
    ratings = df_fatigue["value"].values

    if times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
        ratings = np.insert(ratings, 0, ratings[0])

    end_event = df_events[df_events["event"] == "END"]
    if not end_event.empty:
        end_time = float(end_event["experiment_time_s"].iloc[0])
        if times[-1] < end_time:
            times = np.append(times, end_time)
            ratings = np.append(ratings, ratings[-1])

    pchip = PchipInterpolator(times, ratings)

    frame_indices = np.arange(total_frames)
    timestamps_sec = frame_indices / fps

    interpolated_labels = pchip(timestamps_sec)
    interpolated_labels = np.clip(interpolated_labels, 1.0, 10.0)

    df_result = pd.DataFrame(
        {
            "frame_idx": frame_indices,
            "timestamp_sec": np.round(timestamps_sec, 4),
            "fatigue_label": np.round(interpolated_labels, 4),
        }
    )

    output_path = os.path.join(session_dir, "frame_labels.csv")
    df_result.to_csv(output_path, index=False)
    print(f"   [SUKCES] Zapisano '{output_path}' ({len(df_result)} wierszy)")


def process_all_nagrania(base_dir="nagrania"):
    """Przechodzi po wszystkich podfolderach w folderze nagrania z pominięciem IGNORE..."""
    if not os.path.exists(base_dir):
        print(f" BŁĄD: Folder '{base_dir}' nie istnieje!")
        return

    subfolders = sorted(os.listdir(base_dir))

    processed_count = 0
    ignored_count = 0

    for item in subfolders:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            if item.startswith("IGNORE"):
                print(f"\n Pomijam folder: {item} (zaczyna się od IGNORE)")
                ignored_count += 1
                continue

            # Przetwarzamy poprawną sesję
            process_single_session(item_path)
            processed_count += 1

    print(
        f"\n Zakończono! Przetworzono sesji: {processed_count}, pominięto folderów IGNORE: {ignored_count}."
    )


if __name__ == "__main__":
    process_all_nagrania(base_dir="nagrania")
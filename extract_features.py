import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

IGNORED_VIDEO_PREFIXES = ("IGNORE", "TEMP", "._", "test_")
IGNORED_EXACT_FILES = ["corrupted_video.mp4", "backup.mp4"]


def calculate_ear(landmarks, eye_indices):
    pts = [landmarks[i] for i in eye_indices]

    d_v1 = np.linalg.norm(np.array([pts[1].x, pts[1].y]) - np.array([pts[5].x, pts[5].y]))
    d_v2 = np.linalg.norm(np.array([pts[2].x, pts[2].y]) - np.array([pts[4].x, pts[4].y]))
    d_h = np.linalg.norm(np.array([pts[0].x, pts[0].y]) - np.array([pts[3].x, pts[3].y]))

    if d_h == 0:
        return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)


def calculate_mar(landmarks):
    # Punkty: 78 (lewy kącik), 308 (prawy kącik), 13 (górna warga środek), 14 (dolna warga środek)
    p_left = landmarks[78]
    p_right = landmarks[308]
    p_top = landmarks[13]
    p_bottom = landmarks[14]

    d_vertical = np.linalg.norm(np.array([p_top.x, p_top.y]) - np.array([p_bottom.x, p_bottom.y]))
    d_horizontal = np.linalg.norm(np.array([p_left.x, p_left.y]) - np.array([p_right.x, p_right.y]))

    if d_horizontal == 0:
        return 0.0
    return d_vertical / d_horizontal


def get_head_pose(landmarks, image_shape):
    img_h, img_w, _ = image_shape

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nos
        (0.0, -330.0, -65.0),        # Broda
        (-225.0, 170.0, -135.0),     # Lewe oko
        (225.0, 170.0, -135.0),      # Prawe oko
        (-150.0, -150.0, -125.0),    # Lewe usta
        (150.0, -150.0, -125.0)      # Prawe usta
    ], dtype=np.float64)

    image_points = np.array([
        [landmarks[1].x * img_w, landmarks[1].y * img_h],     # Nos
        [landmarks[152].x * img_w, landmarks[152].y * img_h], # Broda
        [landmarks[263].x * img_w, landmarks[263].y * img_h], # Lewe oko
        [landmarks[33].x * img_w, landmarks[33].y * img_h],   # Prawe oko
        [landmarks[291].x * img_w, landmarks[291].y * img_h], # Lewe usta
        [landmarks[61].x * img_w, landmarks[61].y * img_h]    # Prawy kącik
    ], dtype=np.float64)

    model_points = np.ascontiguousarray(model_points)
    image_points = np.ascontiguousarray(image_points)

    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return np.nan, np.nan

    rmat, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)

    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return float(pitch_deg), float(yaw_deg)


def process_video_features(session_dir):
    all_videos = [f for f in os.listdir(session_dir) if f.lower().endswith(".mp4")]

    valid_videos = []
    for video_name in all_videos:
        if video_name.startswith(IGNORED_VIDEO_PREFIXES):
            print(f"   [IGNOROWANIE PLIKU] Pomijam po prefiksie: {video_name}")
            continue

        if video_name in IGNORED_EXACT_FILES:
            print(f"   [IGNOROWANIE PLIKU] Pomijam po nazwie: {video_name}")
            continue

        valid_videos.append(video_name)

    if not valid_videos:
        print(f"   [INFO] Brak prawidłowych plików .mp4 do przetworzenia w {session_dir}")
        return

    video_path = os.path.join(session_dir, valid_videos[0])
    print(f"\n Rozpoczynam ekstrakcję cech wideo dla: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        print(f"   [BŁĄD] Nie można odczytać klatek z {video_path}. Pomijam.")
        cap.release()
        return

    features = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        ear_l, ear_r, mar, pitch, yaw = np.nan, np.nan, np.nan, np.nan, np.nan

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            ear_l = calculate_ear(landmarks, LEFT_EYE)
            ear_r = calculate_ear(landmarks, RIGHT_EYE)
            mar = calculate_mar(landmarks)

            try:
                pitch, yaw = get_head_pose(landmarks, frame.shape)
            except Exception as e:
                pitch, yaw = np.nan, np.nan

        features.append({
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "ear_left": round(ear_l, 4) if not np.isnan(ear_l) else None,
            "ear_right": round(ear_r, 4) if not np.isnan(ear_r) else None,
            "mar": round(mar, 4) if not np.isnan(mar) else None,
            "head_pitch": round(pitch, 4) if not np.isnan(pitch) else None,
            "head_yaw": round(yaw, 4) if not np.isnan(yaw) else None
        })

        if frame_idx > 0 and frame_idx % 5000 == 0:
            print(f"   Postęp: {frame_idx}/{total_frames} klatek ({(frame_idx/total_frames)*100:.1f}%)")

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(features)
    output_path = os.path.join(session_dir, "video_features.csv")
    df.to_csv(output_path, index=False)
    print(f"   [SUKCES] Zapisano '{output_path}' ({len(df)} wierszy)")


def process_all(base_dir="nagrania"):
    if not os.path.exists(base_dir):
        print(f" BŁĄD: Główny folder '{base_dir}' nie istnieje!")
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

            process_video_features(item_path)
            processed_count += 1

    print(
        f"\n Zakończono przetwarzanie wideo! Sesji przetworzonych: {processed_count}, pominiętych folderów IGNORE: {ignored_count}."
    )


if __name__ == "__main__":
    process_all(base_dir="nagrania")
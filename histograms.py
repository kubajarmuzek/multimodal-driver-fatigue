import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def find_project_root():
    current = os.getcwd()
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "nagrania")):
            return current
        current = os.path.dirname(current)
    return os.getcwd()

def process_and_plot_all_sessions(base_dir_name="nagrania"):
    root_dir = find_project_root()
    base_dir = os.path.join(root_dir, base_dir_name)
    
    if not os.path.exists(base_dir):
        print(f"BŁĄD: Folder główny '{base_dir}' nie istnieje!")
        return

    csv_files = sorted(glob.glob(os.path.join(base_dir, "*", "video_features.csv")))
    
    valid_csv_files = [f for f in csv_files if not os.path.basename(os.path.dirname(f)).startswith("IGNORE")]
    
    if not valid_csv_files:
        print(f"Nie znaleziono żadnych plików 'video_features.csv' w podkatalogach '{base_dir_name}'!")
        return

    print(f"Znaleziono {len(valid_csv_files)} sesji do przetworzenia.")
    
    summary_metrics = []

    for csv_path in valid_csv_files:
        session_id = os.path.basename(os.path.dirname(csv_path))
        print(f"\n---> Generowanie wykresów diagnostycznych dla: {session_id}")
        
        df = pd.read_csv(csv_path)
        total_frames = len(df)
        
        if total_frames == 0:
            print(f"   [OSTRZEŻENIE] Plik dla {session_id} jest pusty. Pomijam.")
            continue
            
        df['ear_avg'] = (df['ear_left'] + df['ear_right']) / 2.0
        
        missing_frames = df['ear_left'].isna().sum()
        missing_pct = (missing_frames / total_frames) * 100.0
        
        clean_df = df.dropna(subset=['ear_avg', 'mar', 'head_pitch', 'head_yaw'])
        avg_ear = clean_df['ear_avg'].mean() if len(clean_df) > 0 else np.nan
        avg_mar = clean_df['mar'].mean() if len(clean_df) > 0 else np.nan
        
        status = "DOBRA"
        if missing_pct > 20.0 or (not np.isnan(avg_ear) and (avg_ear > 0.45 or avg_ear < 0.15)):
            status = "NIEWIARYGODNA (Odrzuć)"
        elif missing_pct > 10.0:
            status = "ŚREDNIA (Czyszczenie)"

        summary_metrics.append({
            "Sesja": session_id,
            "Klatki": total_frames,
            "Brak detekcji (%)": round(missing_pct, 2),
            "Średni EAR": round(avg_ear, 4) if not np.isnan(avg_ear) else None,
            "Średni MAR": round(avg_mar, 4) if not np.isnan(avg_mar) else None,
            "Status": status
        })

        if len(clean_df) == 0:
            print(f"   [OSTRZEŻENIE] Brak poprawnych klatek po usunięciu NaN dla {session_id}.")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Analiza Rozkładów Cech – {session_id} (Status: {status})", fontsize=15, fontweight='bold')

        sns.histplot(clean_df['ear_avg'], kde=True, bins=40, color='royalblue', ax=axes[0, 0])
        axes[0, 0].axvline(x=0.20, color='red', linestyle='--', linewidth=1.5, label='Próg zamknięcia (0.20)')
        axes[0, 0].set_title("Rozkład EAR (Oczy)")
        axes[0, 0].set_xlabel("EAR")
        axes[0, 0].legend()

        sns.histplot(clean_df['mar'], kde=True, bins=40, color='darkorange', ax=axes[0, 1])
        axes[0, 1].axvline(x=0.50, color='red', linestyle='--', linewidth=1.5, label='Próg ziewnięcia (0.50)')
        axes[0, 1].set_title("Rozkład MAR (Usta)")
        axes[0, 1].set_xlabel("MAR")
        axes[0, 1].legend()

        sns.histplot(clean_df['head_pitch'], kde=True, bins=40, color='forestgreen', ax=axes[1, 0])
        axes[1, 0].axvline(x=0.0, color='gray', linestyle=':', label='Pozycja 0°')
        axes[1, 0].set_title("Rozkład Head Pitch (Góra/Dół)")
        axes[1, 0].set_xlabel("Kąt Pitch (°)")
        axes[1, 0].legend()

        sns.histplot(clean_df['head_yaw'], kde=True, bins=40, color='purple', ax=axes[1, 1])
        axes[1, 1].axvline(x=0.0, color='gray', linestyle=':', label='Pozycja 0°')
        axes[1, 1].set_title("Rozkład Head Yaw (Prawo/Lewo)")
        axes[1, 1].set_xlabel("Kąt Yaw (°)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    report_df = pd.DataFrame(summary_metrics)
    print("\n=======================================================")
    print("        ZBIORCZY RAPORT WIARYGODNOŚCI WSZYSTKICH SESJI")
    print("=======================================================")
    print(report_df)

    plt.figure(figsize=(12, 5))
    bar_colors = [
        'crimson' if 'NIEWIARYGODNA' in s else 'orange' if 'ŚREDNIA' in s else 'mediumseagreen' 
        for s in report_df["Status"]
    ]
    plt.bar(report_df["Sesja"], report_df["Brak detekcji (%)"], color=bar_colors)
    plt.axhline(y=15.0, color='red', linestyle='--', label='Próg odrzucenia sesji (15% utraty)')
    plt.ylabel("% brakujących klatek (NaN)")
    plt.title("Zbiór Sesji: Odsetek Utraty Śledzenia Twarzy przez MediaPipe", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Uruchomienie pętli przetwarzającej
process_and_plot_all_sessions(base_dir_name="nagrania")
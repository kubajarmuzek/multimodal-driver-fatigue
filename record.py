import asyncio
import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from bleak import BleakClient, BleakScanner

POLAR_NAME = "Polar H9"

HEART_RATE_CHARACTERISTIC = (
    "00002a37-0000-1000-8000-00805f9b34fb"
)

TEST_DURATION_SECONDS = 3 * 60 * 60
FATIGUE_INTERVAL_SECONDS = 20 * 60

experiment_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

EXPERIMENT_DIR = Path(
    "experiments"
) / f"test_{experiment_id}"

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HR_RR_FILE = EXPERIMENT_DIR / "hr_rr.csv"
EVENTS_FILE = EXPERIMENT_DIR / "events.csv"
METADATA_FILE = EXPERIMENT_DIR / "metadata.json"


experiment_running = False
experiment_start_perf = None
experiment_start_utc = None

polar_connected = False

hr_rr_file = None
hr_rr_writer = None

events_file = None
events_writer = None

last_fatigue_prompt = 0


def experiment_time():

    if experiment_start_perf is None:
        return None

    return time.perf_counter() - experiment_start_perf


def save_event(event, value=""):

    if events_writer is None:
        return

    t = experiment_time()

    events_writer.writerow([
        f"{t:.6f}" if t is not None else "",
        datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        event,
        value
    ])

    events_file.flush()


def save_hr_rr(timestamp, heart_rate, rr_ms):

    if hr_rr_writer is None:
        return

    t = experiment_time()

    hr_rr_writer.writerow([
        f"{t:.6f}" if t is not None else "",
        timestamp,
        heart_rate,
        f"{rr_ms:.2f}"
    ])

    hr_rr_file.flush()


def heart_rate_handler(sender, data):

    global polar_connected

    flags = data[0]

    if flags & 0x01:

        heart_rate = int.from_bytes(
            data[1:3],
            byteorder="little"
        )

        index = 3

    else:

        heart_rate = data[1]

        index = 2

    rr_present = bool(
        flags & 0x10
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )

    if rr_present:

        while index + 1 < len(data):

            rr_raw = int.from_bytes(
                data[index:index + 2],
                byteorder="little"
            )

            rr_seconds = rr_raw / 1024.0

            rr_ms = rr_seconds * 1000.0

            if experiment_running:

                save_hr_rr(
                    timestamp,
                    heart_rate,
                    rr_ms
                )

            print(
                f"{timestamp} | "
                f"HR: {heart_rate:3d} BPM | "
                f"RR: {rr_ms:7.2f} ms"
            )

            index += 2

    else:

        print(
            f"{timestamp} | "
            f"HR: {heart_rate:3d} BPM | "
            f"RR: brak"
        )


polar_loop = None


def start_polar_thread():

    global polar_loop

    def run():

        asyncio.run(
            polar_main()
        )

    thread = threading.Thread(
        target=run,
        daemon=True
    )

    thread.start()


async def polar_main():

    global polar_connected

    print()
    print("========================================")
    print("POLAR")
    print("========================================")
    print("Szukam Polar H9...")
    print()

    devices = await BleakScanner.discover(
        timeout=10
    )

    polar = None

    for device in devices:

        if (
            device.name
            and POLAR_NAME.lower()
            in device.name.lower()
        ):

            polar = device
            break

    if polar is None:

        print(
            "BŁĄD: Nie znaleziono Polar H9."
        )

        root.after(
            0,
            lambda: polar_status.config(
                text="POLAR: NIE ZNALEZIONO"
            )
        )

        return

    print(
        f"Znaleziono: {polar.name}"
    )

    print(
        f"Adres: {polar.address}"
    )

    print()

    try:

        async with BleakClient(
            polar.address
        ) as client:

            polar_connected = True

            print(
                "Połączono z Polar H9!"
            )

            root.after(
                0,
                lambda: polar_status.config(
                    text="POLAR: POŁĄCZONY"
                )
            )

            await client.start_notify(
                HEART_RATE_CHARACTERISTIC,
                heart_rate_handler
            )

            print(
                "Oczekiwanie na START eksperymentu..."
            )

            while True:

                await asyncio.sleep(1)

            await client.stop_notify(
                HEART_RATE_CHARACTERISTIC
            )

    except Exception as e:

        polar_connected = False

        print()
        print(
            "BŁĄD POLAR:"
        )

        print(e)

        root.after(
            0,
            lambda: polar_status.config(
                text="POLAR: BŁĄD"
            )
        )


def prepare_files():

    global hr_rr_file
    global hr_rr_writer

    global events_file
    global events_writer

    hr_rr_file = open(
        HR_RR_FILE,
        "w",
        newline="",
        encoding="utf-8"
    )

    hr_rr_writer = csv.writer(
        hr_rr_file
    )

    hr_rr_writer.writerow([
        "experiment_time_s",
        "timestamp_utc",
        "heart_rate_bpm",
        "rr_ms"
    ])


    events_file = open(
        EVENTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    )

    events_writer = csv.writer(
        events_file
    )

    events_writer.writerow([
        "experiment_time_s",
        "timestamp_utc",
        "event",
        "value"
    ])


def save_metadata():

    metadata = {

        "experiment_id":
            experiment_id,

        "camera":
            "Logi C270 HD WebCam",

        "camera_resolution":
            "1280x720",

        "camera_fps":
            30,

        "test_duration_seconds":
            TEST_DURATION_SECONDS,

        "fatigue_interval_seconds":
            FATIGUE_INTERVAL_SECONDS,

        "polar":
            POLAR_NAME,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


def start_experiment():

    global experiment_running
    global experiment_start_perf
    global experiment_start_utc

    if experiment_running:
        return

    if not polar_connected:

        messagebox.showwarning(
            "Polar",
            "Polar H9 nie jest połączony."
        )

        return

    answer = messagebox.askyesno(
        "START",
        "Czy Aparat już nagrywa i kamera jest zasłonięta?"
    )

    if not answer:
        return

    prepare_files()
    save_metadata()

    experiment_start_perf = time.perf_counter()

    experiment_start_utc = datetime.now(
        timezone.utc
    )

    experiment_running = True

    save_event(
        "START"
    )


    root.configure(
        bg="white"
    )

    title_label.config(
        text="START",
        bg="white",
        fg="black"
    )

    timer_label.config(
        text="ODSŁOŃ KAMERĘ",
        bg="white",
        fg="black"
    )

    root.update()

    print()
    print("========================================")
    print("START EKSPERYMENTU")
    print("========================================")
    print(
        f"UTC: {experiment_start_utc.isoformat()}"
    )
    print(
        "ODSŁOŃ KAMERĘ"
    )


    root.after(
        2000,
        normal_screen
    )


    update_timer()

    root.after(
        FATIGUE_INTERVAL_SECONDS * 1000,
        fatigue_prompt
    )


def normal_screen():

    if not experiment_running:
        return

    root.configure(
        bg="black"
    )

    title_label.config(
        bg="black",
        fg="white"
    )

    timer_label.config(
        bg="black",
        fg="white"
    )

def update_timer():

    if not experiment_running:
        return

    elapsed = experiment_time()

    remaining = (
        TEST_DURATION_SECONDS
        - elapsed
    )

    if remaining <= 0:

        finish_experiment()

        return

    minutes = int(
        remaining // 60
    )

    seconds = int(
        remaining % 60
    )

    timer_label.config(
        text=f"POZOSTAŁO {minutes:02d}:{seconds:02d}"
    )

    root.after(
        500,
        update_timer
    )


def fatigue_prompt():

    if not experiment_running:
        return

    save_event(
        "FATIGUE_PROMPT"
    )

    window = tk.Toplevel(root)

    window.title("Ocena zmęczenia")
    window.geometry("600x350")

    window.attributes("-topmost", True)

    label = tk.Label(
        window,
        text="Wpisz poziom zmęczenia (0–10):",
        font=("Arial", 24)
    )

    label.pack(pady=40)

    entry = tk.Entry(
        window,
        font=("Arial", 28),
        justify="center",
        width=8
    )

    entry.pack(pady=10)

    entry.focus_set()

    def submit():

        value = entry.get().strip()

        try:
            fatigue = int(value)

            if fatigue < 0 or fatigue > 10:
                raise ValueError

        except ValueError:

            messagebox.showwarning(
                "Błędna wartość",
                "Wpisz liczbę całkowitą od 0 do 10."
            )

            entry.focus_set()
            return

        save_event(
            "FATIGUE",
            fatigue
        )

        print()
        print(
            f"ZMĘCZENIE: {fatigue}/10"
        )

        window.destroy()

        if experiment_running:

            root.after(
                FATIGUE_INTERVAL_SECONDS * 1000,
                fatigue_prompt
            )

    button = tk.Button(
        window,
        text="ZATWIERDŹ",
        command=submit,
        font=("Arial", 20),
        width=15,
        height=2
    )

    button.pack(pady=25)

    window.bind(
        "<Return>",
        lambda event: submit()
    )

def finish_experiment():

    global experiment_running

    if not experiment_running:
        return

    experiment_running = False

    save_event(
        "END"
    )

    if hr_rr_file is not None:

        hr_rr_file.flush()
        hr_rr_file.close()

    if events_file is not None:

        events_file.flush()
        events_file.close()


    root.configure(
        bg="black"
    )

    title_label.config(
        text="KONIEC",
        bg="black",
        fg="white"
    )

    timer_label.config(
        text="EKSPERYMENT ZAKOŃCZONY",
        bg="black",
        fg="white"
    )

    print()
    print("========================================")
    print("KONIEC EKSPERYMENTU")
    print("========================================")

    print(
        f"Dane: {EXPERIMENT_DIR.absolute()}"
    )

    messagebox.showinfo(
        "Koniec",
        "Eksperyment zakończony.\n\n"
        f"Dane zapisano w:\n"
        f"{EXPERIMENT_DIR.absolute()}"
    )


def emergency_stop():

    if not experiment_running:

        root.destroy()

        return

    answer = messagebox.askyesno(
        "STOP",
        "Czy na pewno zakończyć eksperyment?"
    )

    if answer:

        save_event(
            "EMERGENCY_STOP"
        )

        finish_experiment()


root = tk.Tk()

root.title(
    "Driver Monitoring Experiment"
)

root.geometry(
    "1000x700"
)

root.configure(
    bg="black"
)

root.protocol(
    "WM_DELETE_WINDOW",
    emergency_stop
)


title_label = tk.Label(
    root,
    text="EKSPERYMENT",
    font=("Arial", 40, "bold"),
    bg="black",
    fg="white"
)

title_label.pack(
    pady=60
)


polar_status = tk.Label(
    root,
    text="POLAR: ŁĄCZENIE...",
    font=("Arial", 20),
    bg="black",
    fg="white"
)

polar_status.pack(
    pady=10
)


timer_label = tk.Label(
    root,
    text="",
    font=("Arial", 28),
    bg="black",
    fg="white"
)

timer_label.pack(
    pady=20
)


start_button = tk.Button(
    root,
    text="START",
    command=start_experiment,
    font=("Arial", 25),
    width=15,
    height=2
)

start_button.pack(
    pady=30
)


stop_button = tk.Button(
    root,
    text="STOP",
    command=emergency_stop,
    font=("Arial", 18),
    width=10
)

stop_button.pack()


# ============================================================
# START
# ============================================================

print("========================================")
print("DRIVER MONITORING EXPERIMENT")
print("========================================")
print()
print(
    f"Katalog: {EXPERIMENT_DIR.absolute()}"
)
print()
print(
    f"Czas testu: "
    f"{TEST_DURATION_SECONDS // 60} min"
)
print(
    "Ocena zmęczenia: "
    f"co {FATIGUE_INTERVAL_SECONDS // 60} min"
)
print()
print("Uruchamiam Polar...")
print()


start_polar_thread()

root.mainloop()

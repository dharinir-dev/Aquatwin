
import os
import time
import joblib
import pandas as pd
import serial

MODEL_PATH = os.path.join("models", "aquatwin_rf_model.joblib")
FEATURES = ["temp_c", "humidity_pct", "water_level_pct", "delta_level", "delta_humidity"]

SERIAL_PORT = "COM5"   # <-- change this to your Arduino's port
BAUD_RATE = 9600

LABEL_TO_CODE = {"Normal": "N", "Evaporation": "E", "Leak": "L"}


def main():
    model = joblib.load(MODEL_PATH)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # allow Arduino to reset after serial connect

    print(f"Listening on {SERIAL_PORT}... (Ctrl+C to stop)")
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split(",")]
            if len(values) != len(FEATURES):
                print(f"Ignoring malformed line: {line}")
                continue
        except ValueError:
            print(f"Ignoring malformed line: {line}")
            continue

        row = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(row)[0]
        code = LABEL_TO_CODE[prediction]

        print(f"Reading: {values} -> {prediction}")
        ser.write(code.encode("utf-8"))


if __name__ == "__main__":
    main()

# new_run_4cams_yolov11_with_relay_debug.py (use yes) 
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import serial
import serial.tools.list_ports

# ============== CONFIG ==============
MODEL_PATH = r"C:\Users\ASUS\Downloads\best (3).pt"
SERIAL_PORT = "COM4"
BAUDRATE = 115200

CAM_INDEXES_TRY = list(range(0, 8))
USE_CAM_INDEXES = None
CONF_THRESH = 0.35
WEED_CLASS_ID = 0
IMG_SIZE = 640
DISPLAY_SCALE = 0.8

# Debounce / cooldown / hold
DETECT_FRAMES = 3
MIN_INTERVAL = 1.0
HOLD_TIME = 2.0

# Explicit mapping: CAMERA_INDEX -> RELAY_CHANNEL (1..4)
# ปรับให้ตรงกับตำแหน่งกล้องจริงของคุณ
CAM_INDEXES_TRY = [0, 1, 3, 4]
USE_CAM_INDEXES = [0, 1, 3, 4]   # บังคับใช้เฉพาะ index เหล่านี้
CAMERA_TO_RELAY = {
    0: 1,   # camera index 0 -> relay channel 1
    1: 2,   # camera index 1 -> relay channel 2
    2: 3,   # camera index 2 -> relay channel 3
    3: 4,   # camera index 3 -> relay channel 4
}

# ถ้าอยากจำลองการส่ง (ไม่ต้องต่อ Arduino) ให้ True
SIMULATE = False
# ====================================

device = 0 if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

print("Loading model...")
model = YOLO(MODEL_PATH)

def list_serial_ports():
    ports = list(serial.tools.list_ports.comports())
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device} - {p.description} - HWID={p.hwid}")
    return ports

def open_serial(port_name=None, baud=115200):
    if SIMULATE:
        print("[SIM] Serial simulation enabled — not opening real port.")
        return None
    ports = list_serial_ports()
    if port_name:
        try:
            ser = serial.Serial(port_name, baud, timeout=0.5)
            time.sleep(1)
            print(f"Opened serial on {port_name}")
            return ser
        except Exception as e:
            print("⚠️  ไม่สามารถเปิด Serial port:", e)

    for p in ports:
        try:
            ser = serial.Serial(p.device, baud, timeout=0.5)
            time.sleep(1)
            print(f"Auto-opened serial on {p.device}")
            return ser
        except Exception as e:
            print(f"can't open {p.device}: {e}")
    print("No serial port available or accessible.")
    return None

ser = open_serial(SERIAL_PORT, BAUDRATE)

def serial_ping(s):
    if SIMULATE:
        print("[SIM] PING -> PONG")
        return True
    if not s:
        return False
    try:
        s.reset_input_buffer()
        s.write(b"PING\n")
        time.sleep(0.2)
        if s.in_waiting:
            line = s.readline().decode(errors='ignore').strip()
            print("Arduino:", line)
            return True
    except Exception as e:
        print("serial_ping error:", e)
    return False

if ser or SIMULATE:
    serial_ping(ser)

def find_cameras(try_indexes):
    cams = []
    for idx in try_indexes:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            try:
                cap.release()
            except: pass
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            print(f"✅ เปิดกล้อง {idx} สำเร็จ")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cams.append((idx, cap))
        else:
            print(f"❌ ไม่สามารถเปิดกล้อง {idx}")
    return cams

all_cams = find_cameras(CAM_INDEXES_TRY)
if USE_CAM_INDEXES:
    cams = [(i,c) for (i,c) in all_cams if i in USE_CAM_INDEXES]
else:
    cams = all_cams[:4]

if len(cams) == 0:
    raise RuntimeError("ไม่พบกล้องที่เปิดได้แม้แต่ตัวเดียว!")

print("\nใช้กล้อง (index):", [i for i,_ in cams])
print("\nเริ่มรันระบบ... (กด q เพื่อออก)\n")

# states
prev_states = {idx: False for idx,_ in cams}
detect_counters = {idx: 0 for idx,_ in cams}
last_change_time = {idx: 0.0 for idx,_ in cams}
relay_hold_until = {idx: 0.0 for idx,_ in cams}

def send_cmd_and_wait_ack(cmd, expected_ack=None, timeout=1.0, retries=2):
    """
    ส่งคำสั่งไปยัง Arduino แล้วรอ ACK (expected_ack เป็น substring ที่คาดว่าจะเห็น เช่น 'OK 1 1')
    คืนค่า True ถ้าได้รับ ack, False ถ้าไม่รับ
    """
    if SIMULATE:
        print(f"[SIM SEND] {cmd.strip()}  -> [SIM ACK] {expected_ack}")
        return True

    if ser is None:
        print("[Serial] ser is None - no serial connection")
        return False

    for attempt in range(1, retries+1):
        try:
            ser.reset_input_buffer()
            print(f"[SEND attempt {attempt}] {cmd.strip()}")
            ser.write(cmd.encode())
        except Exception as e:
            print("Serial write error:", e)
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode(errors='ignore').strip()
                    if line:
                        print("Arduino:", line)
                        if expected_ack is None or expected_ack in line:
                            return True
            except Exception as e:
                print("Serial read error:", e)
                break
        print(f"No ACK for '{cmd.strip()}' (attempt {attempt})")
    return False

def set_relay_channel(channel, value):
    """channel: int (1..), value: 0/1"""
    cmd = f"SET {channel} {value}\n"
    expected = f"OK {channel} {value}"
    ok = send_cmd_and_wait_ack(cmd, expected_ack=expected, timeout=1.0, retries=2)
    if not ok:
        print(f"[WARN] No ACK from Arduino for {cmd.strip()}. Check wiring / sketch / COM port.")
    return ok

def set_relay_for_camera(cam_idx, weed_detected):
    # mapping camera index -> relay channel via CAMERA_TO_RELAY
    if cam_idx in CAMERA_TO_RELAY:
        channel = CAMERA_TO_RELAY[cam_idx]
        print(f"-> Sending relay command: camera {cam_idx} -> channel {channel} -> {'ON' if weed_detected else 'OFF'}")
        set_relay_channel(channel, 1 if weed_detected else 0)
    else:
        print(f"[WARN] No relay mapping for camera index {cam_idx}. Configure CAMERA_TO_RELAY.")

# main loop
while True:
    frames = []
    now = time.time()

    for idx, cap in cams:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)
            continue

        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, device=device, verbose=False)
        annotated = results[0].plot()

        weed_detected_frame = any(int(box.cls[0]) == WEED_CLASS_ID for box in results[0].boxes)

        # debounce counter
        if weed_detected_frame:
            detect_counters[idx] += 1
        else:
            detect_counters[idx] = 0

        candidate_state = detect_counters[idx] >= DETECT_FRAMES
        time_since_last = now - last_change_time[idx]

        if candidate_state != prev_states[idx] and time_since_last >= MIN_INTERVAL:
            print(f"[{time.strftime('%H:%M:%S')}] CAM {idx} state changed: {prev_states[idx]} -> {candidate_state} (cnt={detect_counters[idx]})")
            prev_states[idx] = candidate_state
            last_change_time[idx] = now
            set_relay_for_camera(idx, candidate_state)
            if HOLD_TIME and candidate_state:
                relay_hold_until[idx] = now + HOLD_TIME

        # handle hold time auto-off
        if HOLD_TIME and relay_hold_until[idx] > 0 and now >= relay_hold_until[idx]:
            if prev_states[idx]:
                print(f"[{time.strftime('%H:%M:%S')}] CAM {idx} hold_time elapsed -> OFF")
                prev_states[idx] = False
                set_relay_for_camera(idx, False)
            relay_hold_until[idx] = 0

        color = (0, 0, 255) if prev_states[idx] else (0, 255, 0)
        text = f"CAM {idx} | {'WEED' if prev_states[idx] else 'OK'} | cnt={detect_counters[idx]}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frames.append(annotated)

    # display grid
    if len(frames) >= 4:
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top, bottom))
    elif len(frames) == 3:
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], np.zeros_like(frames[2])))
        grid = np.vstack((top, bottom))
    elif len(frames) == 2:
        grid = np.hstack((frames[0], frames[1]))
    else:
        grid = frames[0]

    grid = cv2.resize(grid, (int(grid.shape[1]*DISPLAY_SCALE), int(grid.shape[0]*DISPLAY_SCALE)))
    cv2.imshow("YOLOv11 - 4 Cameras (Press q to exit)", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
for _, cap in cams:
    cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()   
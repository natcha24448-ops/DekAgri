# new_run_4cams_yolov11_with_relay_debug.py
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

# กล้องที่ต้องการสแกน/ใช้ (บังคับใช้เฉพาะตัวนี้)
CAM_INDEXES_TRY = [0, 1, 3, 4]
USE_CAM_INDEXES = [0, 1, 3, 4]

CONF_THRESH = 0.35          # เกณฑ์ความเชื่อมั่นสำหรับสั่งงาน
WEED_CLASS_ID = 0           # class id ที่ต้องการ (model.names แสดงว่า 0 = 'No' เป็นต้น)
IMG_SIZE = 640
DISPLAY_SCALE = 0.8

# Debounce / cooldown / hold
DETECT_FRAMES = 3
MIN_INTERVAL = 1.0
HOLD_TIME = 2.0

# แม็ปกล้อง (index) -> รีเลย์ (channel) (ตั้งชัดเจนให้ cam index 3 -> relay 3)
# แก้ตามฮาร์ดแวร์ของคุณ หากต้องการ mapping อื่นก็แก้ตรงนี้
CAMERA_TO_RELAY = {
    0: 1,   # cam index 0 -> relay 1
    1: 2,   # cam index 1 -> relay 2
    3: 3,   # cam index 3 -> relay 3  <-- สำคัญ
    4: 4,   # cam index 4 -> relay 4
}

SIMULATE = False  # True = ไม่เชื่อม Arduino, แค่จำลอง
# ====================================

device = 0 if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
print("Loading model...")
model = YOLO(MODEL_PATH)

def list_serial_ports():
    return list(serial.tools.list_ports.comports())

def open_serial(port_name=None, baud=115200):
    if SIMULATE:
        print("[SIM] Serial simulation enabled — not opening real port.")
        return None
    ports = list_serial_ports()
    if port_name:
        try:
            s = serial.Serial(port_name, baud, timeout=0.5)
            time.sleep(1)
            print(f"Opened serial on {port_name}")
            return s
        except Exception as e:
            print("⚠️ cannot open Serial port:", e)
    for p in ports:
        try:
            s = serial.Serial(p.device, baud, timeout=0.5)
            time.sleep(1)
            print(f"Auto-opened serial on {p.device}")
            return s
        except Exception:
            pass
    print("No serial port available or accessible.")
    return None

ser = open_serial(SERIAL_PORT, BAUDRATE)

def send_cmd_and_wait_ack(cmd, expected_ack=None, timeout=1.0, retries=2):
    if SIMULATE:
        print(f"[SIM SEND] {cmd.strip()} -> [SIM ACK] {expected_ack}")
        return True
    if ser is None:
        print("[Serial] No serial connection")
        return False
    for attempt in range(1, retries+1):
        try:
            ser.reset_input_buffer()
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
        # no ack in this attempt
    return False

def set_relay_channel(channel, value):
    cmd = f"SET {channel} {value}\n"
    expected = f"OK {channel} {value}"
    ok = send_cmd_and_wait_ack(cmd, expected_ack=expected, timeout=1.0, retries=2)
    if not ok:
        print(f"[WARN] No ACK for {cmd.strip()}")
    return ok

def set_relay_for_camera(cam_idx, on):
    if cam_idx not in CAMERA_TO_RELAY:
        print(f"[WARN] No mapping for camera {cam_idx} -> skipping relay send")
        return False
    channel = CAMERA_TO_RELAY[cam_idx]
    # on=True => send 1 (ON); on=False => send 0 (OFF)
    print(f"Relay cmd: cam {cam_idx} -> channel {channel} -> {'ON' if on else 'OFF'}")
    return set_relay_channel(channel, 1 if on else 0)

def find_cameras(try_indexes):
    cams = []
    for idx in try_indexes:
        if USE_CAM_INDEXES and idx not in USE_CAM_INDEXES:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            try:
                cap.release()
            except:
                pass
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cams.append((idx, cap))
    return cams

all_cams = find_cameras(CAM_INDEXES_TRY)
if len(all_cams) == 0:
    raise RuntimeError("No cameras available")

# Use only cameras that exist in mapping to ensure cam_idx -> relay channel consistent
cams = [(i, c) for (i, c) in all_cams if i in CAMERA_TO_RELAY]
print("Using camera indices:", [i for i,_ in cams])
print("Camera->Relay mapping:", CAMERA_TO_RELAY)
print("\nStart (press q to exit)\n")

# state bookkeeping
prev_states = {idx: False for idx,_ in cams}
detect_counters = {idx: 0 for idx,_ in cams}
last_change_time = {idx: 0.0 for idx,_ in cams}
relay_hold_until = {idx: 0.0 for idx,_ in cams}

while True:
    frames = []
    now = time.time()

    for idx, cap in cams:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)
            continue

        # run detection with CONF_THRESH to avoid showing thousands of low-conf boxes
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, device=device, verbose=False)
        annotated = results[0].plot()

        # check detections: only consider boxes with conf >= CONF_THRESH
        target_detected = False
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            # boxes.cls and boxes.conf are tensors; iterate them
            try:
                cls_tensor = boxes.cls
                conf_tensor = boxes.conf
                for ci, cf in zip(cls_tensor, conf_tensor):
                    class_idx = int(ci.cpu().numpy()) if hasattr(ci, "cpu") else int(ci)
                    conf_val = float(cf.cpu().numpy()) if hasattr(cf, "cpu") else float(cf)
                    if class_idx == WEED_CLASS_ID and conf_val >= CONF_THRESH:
                        target_detected = True
                        break
            except Exception:
                # fallback: try reading attributes directly
                for b in boxes:
                    try:
                        class_idx = int(getattr(b, "cls", b[5]))
                        conf_val = float(getattr(b, "conf", 0.0))
                        if class_idx == WEED_CLASS_ID and conf_val >= CONF_THRESH:
                            target_detected = True
                            break
                    except Exception:
                        pass

        # debounce logic
        if target_detected:
            detect_counters[idx] += 1
        else:
            # reset counter and if previously ON, turn OFF immediately
            detect_counters[idx] = 0
            if prev_states.get(idx, False):
                prev_states[idx] = False
                set_relay_for_camera(idx, False)

        candidate_state = detect_counters[idx] >= DETECT_FRAMES
        time_since_last = now - last_change_time[idx]

        if candidate_state != prev_states[idx] and time_since_last >= MIN_INTERVAL:
            prev_states[idx] = candidate_state
            last_change_time[idx] = now
            set_relay_for_camera(idx, candidate_state)
            if HOLD_TIME and candidate_state:
                relay_hold_until[idx] = now + HOLD_TIME

        # auto-off after hold time
        if HOLD_TIME and relay_hold_until[idx] > 0 and now >= relay_hold_until[idx]:
            if prev_states.get(idx, False):
                prev_states[idx] = False
                set_relay_for_camera(idx, False)
            relay_hold_until[idx] = 0

        color = (0, 0, 255) if prev_states[idx] else (0, 255, 0)
        text = f"CAM {idx} | {'TARGET' if prev_states[idx] else 'OK'} | cnt={detect_counters[idx]}"
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

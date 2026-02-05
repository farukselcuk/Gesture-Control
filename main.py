print("🚀 Program basladi")

import cv2
import mediapipe as mp
import time
import webbrowser
import keyboard
import os
try:
    import winsound
except Exception:
    winsound = None
from collections import deque
from datetime import datetime
import threading

# =====================
# GLOBAL
# =====================
mode = 1
COOLDOWN = 2.0
last_time = {}
pending = {}  # for sustained (2s) gesture confirmation
snap_history = {"Left": deque(), "Right": deque()}  # (t, px_dist)
pos_history = {"Left": deque(), "Right": deque()}  # (t, x_px) for swipe detection

# Heart gesture
HEART_THRESHOLD = 40        # px
HEART_DISPLAY_TIME = 1.5    # seconds
heart_display_until = 0.0

# Flag video (place file named 'flag.mp4' in project folder)

# Swipe (screenshot) params
SWIPE_WINDOW = 0.5
SWIPE_DELTA_PX = 120
    
# Siren (surrender) sound
SIREN_FILE = "siren.mp4"  # place a WAV or mp4 file in project root for local playback
# No YouTube fallback anymore; prefer local `siren` file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flag video (place file named 'flag.mp4' in the same folder as main.py)
FLAG_VIDEO = os.path.join(BASE_DIR, "flag.mp4")

# Screenshots folder (under project dir)
SS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SS_DIR, exist_ok=True)

# Photo files for mode 3 (place photo1.jpg..photo4.jpg in project folder)
PHOTO_FILES = {
    1: os.path.join(BASE_DIR, "foto1.jpeg"),
    2: os.path.join(BASE_DIR, "foto2.jpeg"),
    3: os.path.join(BASE_DIR, "foto3.jpeg"),
    4: os.path.join(BASE_DIR, "foto4.jpeg"),
}

def show_image(path, duration=3.0):
    """Show an image in a separate window for `duration` seconds then close.
    Resize image to fit within a sensible max size while keeping aspect ratio.
    """
    def _show():
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot load image: {path}")
            return
        # maximum display size to avoid oversized windows
        max_w, max_h = 800, 600
        h_img, w_img = img.shape[:2]
        scale = min(1.0, max_w / float(w_img), max_h / float(h_img))
        if scale < 1.0:
            nw = int(w_img * scale)
            nh = int(h_img * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        win = f"Photo: {os.path.basename(path)}"
        start = time.time()
        while time.time() - start < duration:
            cv2.imshow(win, img)
            if cv2.waitKey(100) & 0xFF == 27:
                break
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass

    t = threading.Thread(target=_show, daemon=True)
    t.start()

def can_run(key):
    now = time.time()
    if key not in last_time or now - last_time[key] > COOLDOWN:
        last_time[key] = now
        return True
    return False

# =====================
# LINKS
# =====================
EDIS_SONG = "https://music.youtube.com/watch?v=oLqZ7OMWzQs"
MANIFEST_SNAP = "https://music.youtube.com/watch?v=4rupInb9c78"
TARKAN_OPUCEM = "https://music.youtube.com/watch?v=VMn4mN7TeAk"
ARDA_GULER = "https://www.google.com/search?q=arda+g%C3%BCler+hareketi"
MANIFEST_ARIYOR = "https://music.youtube.com/watch?v=69IoUJcdsGU"

def open_link(link):
    """Open a URL or local file robustly on Windows, fallback to webbrowser."""
    try:
        if os.path.exists(link):
            # local file (video)
            os.startfile(link)
            return
        if isinstance(link, str) and link.startswith("http"):
            # prefer OS open on Windows for default behavior
            try:
                os.startfile(link)
                return
            except Exception:
                pass
        webbrowser.open(link, new=2)
    except Exception as e:
        try:
            webbrowser.open(link, new=2)
        except Exception:
            print(f"Cannot open link: {link} -> {e}")


def play_local_video(path, duration=5.5):
    """Play a local video file in a separate thread for `duration` seconds then close."""
    def _play():
        capv = cv2.VideoCapture(path)
        win = f"Video: {os.path.basename(path)}"
        start = time.time()
        while capv.isOpened() and time.time() - start < duration:
            ret, framev = capv.read()
            if not ret:
                break
            cv2.imshow(win, framev)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        capv.release()
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass

    t = threading.Thread(target=_play, daemon=True)
    t.start()

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)

def finger_open(tip, pip):
    return tip.y < pip.y

def px_distance(p1, p2, w, h):
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    return (dx*dx + dy*dy) ** 0.5

SNAP_WINDOW = 0.35
SNAP_DELTA_PX = 30
SUSTAIN_TIME = 2.0

def detect_snap(history_deque):
    # history_deque: deque of (t, dist_px) ordered by time
    if len(history_deque) < 3:
        return False
    arr = list(history_deque)
    dists = [x[1] for x in arr]
    times = [x[0] for x in arr]
    min_idx = dists.index(min(dists))
    if min_idx == 0 or min_idx == len(dists) - 1:
        return False
    before_max = max(dists[:min_idx]) if min_idx > 0 else 0
    after_max = max(dists[min_idx+1:]) if min_idx < len(dists)-1 else 0
    if before_max - dists[min_idx] >= SNAP_DELTA_PX and after_max - dists[min_idx] >= SNAP_DELTA_PX:
        t_before = times[dists.index(before_max)]
        t_after = times[min_idx+1 + dists[min_idx+1:].index(after_max)]
        if t_after - t_before <= SNAP_WINDOW:
            return True
    return False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # MODE SWITCH
        if keyboard.is_pressed("1"):
            mode = 1
        elif keyboard.is_pressed("2"):
            mode = 2
        elif keyboard.is_pressed("3"):
            mode = 3

        cv2.putText(
            frame,
            f"MODE {mode}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        h, w = frame.shape[:2]

        if res.multi_hand_landmarks and res.multi_handedness:
            # count surrender candidates per frame
            surrender_count = 0
            for hand_lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm = hand_lms.landmark
                hand_label = handed.classification[0].label  # Left / Right

                index = finger_open(lm[8], lm[6])
                middle = finger_open(lm[12], lm[10])
                ring = finger_open(lm[16], lm[14])
                pinky = finger_open(lm[20], lm[18])

                thumb_index_px = px_distance(lm[4], lm[8], w, h)
                thumb_middle_px = px_distance(lm[4], lm[12], w, h)

                # simple thumb-open heuristic
                thumb_open = thumb_index_px > 50

                now = time.time()

                # HEART: only in mode 1 — thumb + index touch -> display message
                if mode == 1 and thumb_index_px < HEART_THRESHOLD:
                    heart_display_until = now + HEART_DISPLAY_TIME

                # track wrist x for swipe detection
                wrist_x = lm[0].x * w
                ph = pos_history.get(hand_label)
                ph.append((now, wrist_x))
                while ph and now - ph[0][0] > SWIPE_WINDOW:
                    ph.popleft()

                # palm open heuristic
                palm_open = index and middle and ring and pinky and thumb_open

                # detect right-swipe (open palm move right quickly)
                if palm_open and len(ph) >= 2:
                    first_x = ph[0][1]
                    last_x = ph[-1][1]
                    if last_x - first_x > SWIPE_DELTA_PX:
                        if can_run("screenshot"):
                            fname = os.path.join(SS_DIR, f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                            cv2.imwrite(fname, frame)
                            print(f"📸 Screenshot saved: {fname}")
                        ph.clear()

                # WOLF / FLAG gesture: index + pinky up, others down (only mode 1)
                wolf = index and pinky and not middle and not ring
                if wolf and mode == 1:
                    if can_run("flag"):
                        print("🇹🇷 Flag gesture detected - opening flag video")
                        if os.path.exists(FLAG_VIDEO):
                            try:
                                os.startfile(FLAG_VIDEO)
                            except Exception:
                                try:
                                    open_link(FLAG_VIDEO)
                                except Exception:
                                    print(f"Cannot open flag video: {FLAG_VIDEO}")
                        else:
                            print(f"Flag video not found: {FLAG_VIDEO}")

                # surrender candidate detection (both hands up with open palm) — for Mode 2
                wrist_y = lm[0].y * h
                raised = wrist_y < (h * 0.45)
                if mode == 2 and palm_open and raised:
                    surrender_count += 1

                # =====================
                # MODE 2 - sustained (2s) gestures
                # =====================
                if mode == 2:
                    # Left: thumb + pinky open, others closed -> Manifest arıyorum
                    left_call = (
                        hand_label == "Left" and thumb_open and pinky and not index and not middle and not ring
                    )

                    if left_call:
                        key = "manifest_left"
                        now = time.time()
                        if key not in pending:
                            pending[key] = now
                        elif now - pending[key] >= SUSTAIN_TIME:
                            if can_run(key):
                                print("📞 Manifest Arıyorum (Left)")
                                open_link(MANIFEST_ARIYOR)
                            pending.pop(key, None)
                    else:
                        pending.pop("manifest_left", None)

                    # Right: thumb + pinky open, others closed -> Edis song
                    right_call = (
                        hand_label == "Right" and thumb_open and pinky and not index and not middle and not ring
                    )

                    if right_call:
                        key = "manifest_right"
                        now = time.time()
                        if key not in pending:
                            pending[key] = now
                        elif now - pending[key] >= SUSTAIN_TIME:
                            if can_run(key):
                                print("📞 Edis Şarkısı (Right)")
                                open_link(EDIS_SONG)
                            pending.pop(key, None)
                    else:
                        pending.pop("manifest_right", None)

                    # Arda Güler: only RIGHT hand, index up alone
                    arda_gesture = (
                        hand_label == "Right" and index and not middle and not ring and not pinky
                    )
                    if arda_gesture:
                        key = "arda"
                        now = time.time()
                        if key not in pending:
                            pending[key] = now
                        elif now - pending[key] >= SUSTAIN_TIME:
                            if can_run(key):
                                print("⚽ Arda Güler")
                                open_link(ARDA_GULER)
                            pending.pop(key, None)
                    else:
                        pending.pop("arda", None)

                # =====================
                # MODE 3 - snap detection
                # =====================
                if mode == 3:
                    # append thumb-middle distance history for this hand
                    hist = snap_history.get(hand_label)
                    now = time.time()
                    hist.append((now, thumb_middle_px))
                    # prune old
                    while hist and now - hist[0][0] > SNAP_WINDOW:
                        hist.popleft()

                    if detect_snap(hist):
                        if can_run("snap"):
                            print("🤌 Manifest Snap")
                            open_link(MANIFEST_SNAP)
                        hist.clear()

                    # Mode 3: finger-count photo display (1..4)
                    finger_count = int(index) + int(middle) + int(ring) + int(pinky)
                    if finger_count in (1, 2, 3, 4):
                        key = f"photo{finger_count}"
                        if can_run(key):
                            img_path = PHOTO_FILES.get(finger_count)
                            if img_path and os.path.exists(img_path):
                                print(f"🖼️ Showing photo{finger_count}")
                                show_image(img_path, duration=3.0)
                            else:
                                print(f"Photo not found for {finger_count}: {img_path}")

                    # Tarkan kiss -- suspended (face-based) -> keep commented for now
                    # kiss_gesture = thumb_index_px < 30
                    # if kiss_gesture and can_run("tarkan"):
                    #     print("💋 Tarkan Öpücem")
                    #     webbrowser.open(TARKAN_OPUCEM)

            # after processing hands in this frame, check surrender (both hands)
            if 'surrender_count' in locals() and surrender_count >= 2 and mode == 2:
                if can_run("siren"):
                    print("🚨 Surrender detected — playing siren")
                    if os.path.exists(SIREN_FILE):
                        ext = os.path.splitext(SIREN_FILE)[1].lower()
                        if ext == '.wav' and winsound:
                            try:
                                winsound.PlaySound(SIREN_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
                            except Exception:
                                print(f"Could not play WAV via winsound: {SIREN_FILE}")
                        else:
                            try:
                                os.startfile(SIREN_FILE)
                            except Exception:
                                try:
                                    play_local_video(SIREN_FILE, duration=5.5)
                                except Exception:
                                    print(f"Cannot play siren file: {SIREN_FILE}")
                    else:
                        print(f"Siren file not found: {SIREN_FILE}")
                # overlay text briefly
                cv2.putText(frame, "TESLIM OL!", (w//2 - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # draw heart message if active
        if time.time() < heart_display_until:
            cv2.putText(frame, "Cok guzelsin", (w//2 - 160, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

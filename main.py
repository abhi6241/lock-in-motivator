import cv2
from mediapipe import solutions as mp
import numpy as np
import time
import subprocess
from pathlib import Path

# -------------------- macOS video control --------------------

def osascript(script: str):
    subprocess.run(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

def play_video(video_path: Path):
    script = f'''
    tell application "QuickTime Player"
        activate
        open POSIX file "{video_path}"
        play document 1
    end tell
    '''
    osascript(script)

def close_all_videos():
    script = '''
    tell application "QuickTime Player"
        close every document saving no
    end tell
    '''
    osascript(script)

# -------------------- UI --------------------

def draw_warning(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 10, 30), -1)
    cv2.putText(
        frame,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (0, 255, 200),
        3,
        cv2.LINE_AA,
    )

# -------------------- Head Pose --------------------

FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),   # Chin
    (-43.3, 32.7, -26.0),  # Left eye corner
    (43.3, 32.7, -26.0),   # Right eye corner
    (-28.9, -28.9, -24.1), # Left mouth
    (28.9, -28.9, -24.1),  # Right mouth
])

FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def estimate_pitch(landmarks, frame_shape):
    h, w = frame_shape[:2]

    face_2d = []
    for idx in FACE_LANDMARK_IDS:
        lm = landmarks[idx]
        face_2d.append((lm.x * w, lm.y * h))

    face_2d = np.array(face_2d, dtype=np.float64)

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])

    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_POINTS,
        face_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pitch = np.degrees(np.arctan2(-rot_mat[2, 1], rot_mat[2, 2]))
    return pitch

# -------------------- MAIN --------------------

def main():
    face_mesh = mp.face_mesh.FaceMesh()
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    video_path = (Path(__file__).parent / "motivid.mp4").resolve()

    EMA_ALPHA = 0.2
    DOWN_PITCH_THRESHOLD = -15.0   # degrees
    REQUIRED_TIME = 2.0            # seconds

    smoothed_pitch = None
    down_time = 0.0
    last_time = time.time()
    video_playing = False

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        now = time.time()
        dt = now - last_time
        last_time = now

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            pitch = estimate_pitch(landmarks, frame.shape)

            if pitch is not None:
                if smoothed_pitch is None:
                    smoothed_pitch = pitch
                else:
                    smoothed_pitch = (
                        EMA_ALPHA * pitch +
                        (1 - EMA_ALPHA) * smoothed_pitch
                    )

                if smoothed_pitch < DOWN_PITCH_THRESHOLD:
                    down_time += dt
                else:
                    down_time = max(0, down_time - dt * 2)
        else:
            down_time = 0

        if down_time >= REQUIRED_TIME and not video_playing:
            play_video(video_path)
            video_playing = True

        if down_time < 0.3 and video_playing:
            close_all_videos()
            video_playing = False

        if video_playing:
            draw_warning(frame, "DOOMSCROLLING DETECTED")

        cv2.putText(
            frame,
            f"Pitch: {smoothed_pitch:.1f}" if smoothed_pitch else "Pitch: --",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("lock in", frame)
        if cv2.waitKey(1) == 27:
            break

    close_all_videos()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

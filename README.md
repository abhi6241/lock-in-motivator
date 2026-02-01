# 🔒 lock in

Detects when you're doomscrolling by tracking your head pose via webcam. The moment you've been looking down long enough, it plays a motivational video to snap you out of it.

## How it works

The app uses MediaPipe's Face Mesh to track 6 facial landmarks in real time. From those landmarks, it estimates your head pitch using PnP (Perspective-n-Point) solving — the same technique used in AR. An exponential moving average smooths out the noisy pitch values so it doesn't flicker.

If your smoothed pitch stays below **-15°** (i.e., you're looking down) for **2 continuous seconds**, it fires off a motivational video in QuickTime Player and slaps a warning banner on the webcam feed. Look back up and it closes everything automatically.

## Requirements

- **macOS** (uses AppleScript to control QuickTime Player)
- **Python 3.9 or 3.10** — this project does **not** support Python 3.13 or later due to compatibility issues with the vision dependencies. Python 3.9 or 3.10 is strongly recommended.
- A webcam

## Installation

Clone the repo:

```bash
git clone https://github.com/abhi6241/lock-in-motivator.git
cd lock-in-motivator
```

Create and activate a virtual environment (use Python 3.9 or 3.10):

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

Place your motivational video in the project root and name it `motivid.mp4`:

```
lock-in/
├── main.py
├── motivid.mp4
└── README.md
```

## Usage

```bash
python main.py
```

A webcam window labeled **"lock in"** will open. Your current head pitch is displayed at the bottom of the frame. Just browse normally — if you start doomscrolling, you'll know.

Press **Esc** to exit.

## Configuration

All tuning lives at the top of `main()` in `main.py`:

| Variable | Default | What it does |
|---|---|---|
| `EMA_ALPHA` | `0.2` | Smoothing factor for pitch. Lower = smoother but slower to react. |
| `DOWN_PITCH_THRESHOLD` | `-15.0°` | How far down you need to look to trigger detection. |
| `REQUIRED_TIME` | `2.0s` | How long you must hold that angle before the video fires. |

## Contributing

Got an idea or a fix? Fork the repo, make your changes, and open a pull request. Happy to review.

## License

MIT

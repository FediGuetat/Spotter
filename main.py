"""Pull-up counter.

    python main.py                    (webcam)
    python main.py --source clip.mp4 --output result.mp4

click a person = track only them, A = back to auto, space = pause, Q = quit
"""

import argparse
import ctypes
import functools
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from counter import Point, RepCounter, hands_overhead, nearest_pose, pick_athlete, shoulder_center

MODEL = Path(__file__).with_name("pose_landmarker_lite.task")
WINDOW = "Pull-up counter"
# skip face points (0-10), only draw the body
BODY = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS if c.start > 10 and c.end > 10]

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, GREEN, ORANGE, YELLOW = (255, 255, 255), (80, 220, 100), (0, 165, 255), (0, 220, 255)


@functools.cache  # screen doesn't change, ask Windows once
def screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except AttributeError:  # not Windows
        return 1920, 1080


def fit_half_screen(frame):
    # about half the screen wide
    sw, sh = screen_size()
    h, w = frame.shape[:2]
    scale = min(sw * 0.5 / w, sh * 0.85 / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def draw_figure(frame, landmarks: list[Point]) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    thickness = max(2, w // 300)
    for a, b in BODY:
        cv2.line(frame, pts[a], pts[b], GREEN, thickness, cv2.LINE_AA)
    for i in range(11, len(pts)):
        cv2.circle(frame, pts[i], thickness + 2, WHITE, -1, cv2.LINE_AA)


def draw_bar_target(frame, bar_y: float) -> None:
    h, w = frame.shape[:2]
    y = int(bar_y * h)
    for x in range(0, w, 30):  # dashed
        cv2.line(frame, (x, y), (min(x + 15, w), y), YELLOW, 2, cv2.LINE_AA)
    # on the right so the panel doesn't cover it
    (tw, _), _ = cv2.getTextSize("chin target", FONT, 0.6, 1)
    cv2.putText(frame, "chin target", (w - tw - 10, y - 8), FONT, 0.6, YELLOW, 1, cv2.LINE_AA)


def draw_panel(frame, counter: RepCounter, locked: bool) -> None:
    # (text, size, color, thickness)
    gray = (200, 200, 200)
    lines = [
        (f"PULL-UPS  {counter.count}", 1.3, WHITE, 2),
        (f"set {counter.set_no}: {counter.set_count} reps", 0.6, WHITE, 1),
        (f"state: {counter.state}   tracking: {'locked' if locked else 'auto'}", 0.55, gray, 1),
    ]
    if counter.reps:
        lines.append((f"form: {counter.clean_reps}/{len(counter.reps)} clean reps", 0.7, WHITE, 1))
        last = counter.reps[-1]
        if last.issues:
            lines.append((f"rep {last.number}:", 0.7, ORANGE, 1))
            lines += [(f"  - {issue}", 0.6, ORANGE, 1) for issue in last.issues]
        else:
            lines.append((f"rep {last.number}: good form", 0.7, GREEN, 1))
        if last.pull_s is not None:
            down = f" / down {last.lower_s:.1f}s" if last.lower_s is not None else ""
            lines.append((f"speed: up {last.pull_s:.1f}s{down}", 0.6, WHITE, 1))
    slowdown = counter.slowdown()
    if slowdown is not None:
        color = ORANGE if slowdown > 0.25 else WHITE
        lines.append((f"fatigue: pulls {slowdown:+.0%} vs first reps", 0.6, color, 1))

    pad, y = 14, 14
    sizes = [cv2.getTextSize(t, FONT, s, th) for t, s, _, th in lines]
    width = max(sz[0][0] for sz in sizes) + 2 * pad
    height = sum(sz[0][1] + sz[1] + 10 for sz in sizes) + pad
    # darken only the panel area, copying the whole frame for this is wasteful
    roi = frame[10:10 + height, 10:10 + width]
    dark = roi.copy()
    dark[:] = 20
    cv2.addWeighted(dark, 0.65, roi, 0.35, 0, dst=roi)
    for (text, scale, color, th), ((_, th_h), base) in zip(lines, sizes):
        y += th_h + 6
        cv2.putText(frame, text, (10 + pad, y), FONT, scale, color, th, cv2.LINE_AA)
        y += base + 4

    help_text = "click: select person   A: auto   space: pause   Q: quit"
    cv2.putText(frame, help_text, (12, frame.shape[0] - 12), FONT, 0.5, gray, 1, cv2.LINE_AA)


def print_summary(counter: RepCounter) -> None:
    print(f"\nTotal pull-ups: {counter.count}")
    if not counter.reps:
        return
    print(f"Clean reps: {counter.clean_reps}/{len(counter.reps)}")
    slowdown = counter.slowdown()
    if slowdown is not None:
        print(f"Fatigue (last set): last 3 pulls {slowdown:+.0%} vs first 3")
    for rep in counter.reps:
        up = f"{rep.pull_s:.1f}s" if rep.pull_s is not None else "-"
        down = f"{rep.lower_s:.1f}s" if rep.lower_s is not None else "-"
        form = "good" if not rep.issues else "; ".join(rep.issues)
        print(f"  set {rep.set_no}  rep {rep.number:>2}  up {up:>5}  down {down:>5}  {form}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index or video file path")
    parser.add_argument("--output", help="save the annotated video here (.mp4)")
    parser.add_argument("--no-window", action="store_true", help="run without a preview window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3,  # the landmark model runs once per person, more = slower
    )
    counter = RepCounter()
    athlete_center = None
    locked = False  # True after the user clicks on someone
    away_frames = 0  # frames the tracked person hasn't been holding the bar
    paused = False
    writer = None

    # mouse clicks come in as pixel positions in the shown frame
    ui = {"click": None, "size": (1, 1)}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = ui["size"]
            ui["click"] = (x / w, y / h)

    if not args.no_window:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        quit_ = False
        while not quit_:
            ok, raw = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(frame_index * 1000 / fps))
            t = frame_index / fps
            frame_index += 1

            poses = [[Point(p.x, p.y, p.visibility) for p in pose] for pose in result.pose_landmarks]
            chosen = pick_athlete(poses, athlete_center, locked=locked)
            if chosen is not None:
                athlete_center = shoulder_center(poses[chosen])
                counter.update(poses[chosen], t)

                # in auto mode let go of someone who stopped holding the bar for 1s,
                # otherwise we end up following whoever walks past that spot
                away_frames = 0 if hands_overhead(poses[chosen], 0.5) else away_frames + 1
                if not locked and away_frames > fps:
                    athlete_center, away_frames = None, 0

            if args.no_window and not args.output:
                continue  # nobody sees the frame, skip drawing

            # runs once per frame, or keeps redrawing while paused
            while True:
                if ui["click"] is not None:
                    i, d = nearest_pose(poses, ui["click"])
                    ui["click"] = None
                    if i is not None and d < 0.25:
                        chosen, locked = i, True
                        athlete_center = shoulder_center(poses[i])

                frame = fit_half_screen(raw)
                ui["size"] = (frame.shape[1], frame.shape[0])
                if counter.bar_y is not None:
                    draw_bar_target(frame, counter.bar_y)
                if chosen is not None:
                    draw_figure(frame, poses[chosen])
                draw_panel(frame, counter, locked)

                if args.no_window:
                    break
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key == ord("q"):
                    quit_ = True
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("a"):
                    locked = False
                if quit_ or not paused:
                    break

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)

    counter.finish()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print_summary(counter)


if __name__ == "__main__":
    main()

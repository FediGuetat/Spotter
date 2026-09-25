# Pull-up counting + form checks from MediaPipe landmarks.
# Coordinates are 0-1 and y goes DOWN, so "higher" means smaller y.

from dataclasses import dataclass, field
import math

NOSE = 0
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

REQUIRED = (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
)


@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


def angle(a: Point, b: Point, c: Point) -> float:
    # angle at b in degrees
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    norm = math.hypot(*ab) * math.hypot(*cb)
    if norm == 0:
        return 0.0
    cos = (ab[0] * cb[0] + ab[1] * cb[1]) / norm
    # rounding can give 1.0000001 and acos crashes on that
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def shoulder_center(landmarks: list[Point]) -> tuple[float, float]:
    ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    return (ls.x + rs.x) / 2, (ls.y + rs.y) / 2


def hands_overhead(landmarks: list[Point], min_visibility: float) -> bool:
    # wrists above shoulders = holding the bar. Compared to shoulders, not the
    # nose, because at the top of a rep the nose goes above the wrists.
    pairs = ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER))
    return all(
        landmarks[w].visibility >= min_visibility and landmarks[w].y < landmarks[s].y
        for w, s in pairs
    )


def nearest_pose(poses: list[list[Point]], point: tuple[float, float]) -> tuple[int | None, float]:
    # index of the person whose shoulders are closest to point, and the distance
    if not poses:
        return None, math.inf
    dist = [math.dist(shoulder_center(p), point) for p in poses]
    i = min(range(len(poses)), key=dist.__getitem__)
    return i, dist[i]


def pick_athlete(
    poses: list[list[Point]],
    previous: tuple[float, float] | None,
    max_jump: float = 0.15,
    min_visibility: float = 0.5,
    locked: bool = False,
) -> int | None:
    # Returns index of the person doing pull-ups (or None).
    # Stick with last frame's person if someone is still close to where they were,
    # otherwise pick whoever holds the bar, closest to the middle.
    # locked = user clicked on someone, never switch to another person on our own
    if previous is not None:
        i, d = nearest_pose(poses, previous)
        if d <= max_jump:
            return i
        if locked:
            return None
    candidates = [i for i, p in enumerate(poses) if hands_overhead(p, min_visibility)]
    if not candidates:
        return None
    return min(candidates, key=lambda i: abs(shoulder_center(poses[i])[0] - 0.5))


@dataclass
class Rep:
    number: int
    set_no: int = 1
    issues: list[str] = field(default_factory=list)
    pull_s: float | None = None  # bottom to chin over bar
    lower_s: float | None = None  # back down to straight arms


@dataclass
class _Cycle:
    # stats collected during one rep, used for the form check
    max_elbow: float = 0.0
    min_mouth_y: float = 1.0
    hip_x_min: float = 1.0
    hip_x_max: float = 0.0
    torso: float = 0.0
    max_arm_gap: float = 0.0


class RepCounter:
    # states:
    #   down - hanging, arms straight
    #   up   - nose above the bar (rep counted here)
    #   off  - hands not on the bar (dropped / standing), frames ignored

    def __init__(
        self,
        down_angle: float = 150.0,
        lockout_angle: float = 160.0,
        max_swing: float = 0.25,
        max_arm_gap: float = 25.0,
        min_visibility: float = 0.5,
    ):
        self.down_angle = down_angle
        self.lockout_angle = lockout_angle
        self.max_swing = max_swing
        self.max_arm_gap = max_arm_gap
        self.min_visibility = min_visibility
        self.count = 0
        self.state = "unknown"
        self.bar_y: float | None = None
        self.reps: list[Rep] = []
        self.set_no = 1
        self._off_since = 0.0
        self._cycle: _Cycle | None = None
        self._pending: Rep | None = None
        self._last_down_t = 0.0  # last time arms were straight
        self._up_t = 0.0  # when the current rep reached the top

    def update(self, landmarks: list[Point], t: float) -> int:
        # t = video time in seconds
        if any(landmarks[i].visibility < self.min_visibility for i in REQUIRED):
            return self.count

        wrist_y = (landmarks[LEFT_WRIST].y + landmarks[RIGHT_WRIST].y) / 2
        left = angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
        right = angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        elbow = (left + right) / 2
        hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
        hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        torso = abs(hip_y - shoulder_center(landmarks)[1])

        # a dead hang tells us where the bar is. Done again after every break,
        # the camera may have moved in between sets
        if self.bar_y is None or self.state == "off":
            if not (hands_overhead(landmarks, self.min_visibility) and elbow >= self.down_angle):
                return self.count
            self.bar_y = wrist_y
            # back on the bar after 5s+ off = new set. Shorter gaps are usually
            # just a few frames where the hands weren't detected
            if (self.state == "off" and t - self._off_since > 5
                    and self.reps and self.reps[-1].set_no == self.set_no):
                self.set_no += 1

        # hands left the bar. Tolerance is relative to torso size so it works
        # no matter how far the person is from the camera
        if abs(wrist_y - self.bar_y) > 0.3 * torso:
            self._close_rep()
            self._cycle = None
            if self.state != "off":
                self._off_since = t
            self.state = "off"
            return self.count

        if self._cycle is None:
            self._cycle = _Cycle()
        c = self._cycle
        c.max_elbow = max(c.max_elbow, elbow)
        mouth = [landmarks[i] for i in (MOUTH_LEFT, MOUTH_RIGHT) if landmarks[i].visibility >= self.min_visibility]
        if mouth:
            c.min_mouth_y = min(c.min_mouth_y, min(p.y for p in mouth))
        c.hip_x_min, c.hip_x_max = min(c.hip_x_min, hip_x), max(c.hip_x_max, hip_x)
        c.torso = max(c.torso, torso)
        c.max_arm_gap = max(c.max_arm_gap, abs(left - right))

        if elbow >= self.down_angle:
            if self.state == "up":
                if self._pending:
                    self._pending.lower_s = t - self._up_t
                self._close_rep()
                self._cycle = _Cycle(max_elbow=elbow, hip_x_min=hip_x, hip_x_max=hip_x, torso=torso)
            self.state = "down"
            self._last_down_t = t
            # slowly adjust bar height so one bad frame can't move it much
            self.bar_y = 0.9 * self.bar_y + 0.1 * wrist_y
        elif landmarks[NOSE].y < self.bar_y and self.state == "down":
            self.state = "up"
            self.count += 1
            self._pending = Rep(self.count, self.set_no, pull_s=t - self._last_down_t)
            self._up_t = t
        return self.count

    def finish(self) -> None:
        # video ended, last rep still needs its form check
        self._close_rep()

    def _close_rep(self) -> None:
        rep, c = self._pending, self._cycle
        if rep is None or c is None:
            return
        if c.max_elbow < self.lockout_angle:
            rep.issues.append("Arms not fully extended at bottom")
        if self.bar_y is not None and c.min_mouth_y > self.bar_y:
            rep.issues.append("Chin not clearly over bar")
        if c.torso and (c.hip_x_max - c.hip_x_min) > self.max_swing * c.torso:
            rep.issues.append("Body swinging / kipping")
        if c.max_arm_gap > self.max_arm_gap:
            rep.issues.append("Uneven pull (one arm leads)")
        self.reps.append(rep)
        self._pending = None

    @property
    def clean_reps(self) -> int:
        return sum(1 for r in self.reps if not r.issues)

    @property
    def set_count(self) -> int:
        # reps in the current set, including the one still in progress
        done = sum(1 for r in self.reps if r.set_no == self.set_no)
        return done + (self._pending is not None)

    def slowdown(self) -> float | None:
        # how much slower the last 3 pulls of the current set are than its
        # first 3 (0.4 = 40% slower). Pull time only, that's where fatigue shows first
        times = [r.pull_s for r in self.reps if r.pull_s and r.set_no == self.set_no]
        if len(times) < 6:
            return None
        first, last = sum(times[:3]) / 3, sum(times[-3:]) / 3
        return last / first - 1

# tests for counter.py, uses fake skeletons instead of a video

from counter import Point, RepCounter, angle, nearest_pose, pick_athlete


def pose(
    nose_y: float,
    elbow_offset: float,
    visibility: float = 1.0,
    right_elbow_offset: float | None = None,
    hip_dx: float = 0.0,
    wrist_y: float = 0.2,
) -> list[Point]:
    # fake front-view skeleton, bar at y=0.2
    # elbow_offset=0 -> straight arms, bigger -> more bent
    if right_elbow_offset is None:
        right_elbow_offset = elbow_offset
    lm = [Point(0.5, 0.5, visibility) for _ in range(33)]
    lm[0] = Point(0.5, nose_y, visibility)
    lm[9] = Point(0.48, nose_y + 0.03, visibility)
    lm[10] = Point(0.52, nose_y + 0.03, visibility)
    for side, sign, offset in ((11, -1, elbow_offset), (12, 1, right_elbow_offset)):
        shoulder = Point(0.5 + sign * 0.1, nose_y + 0.1, visibility)
        wrist = Point(0.5 + sign * 0.1, wrist_y, visibility)
        elbow = Point(shoulder.x + sign * offset, (shoulder.y + wrist.y) / 2, visibility)
        lm[side], lm[side + 2], lm[side + 4] = shoulder, elbow, wrist
        lm[side + 12] = Point(shoulder.x + hip_dx, shoulder.y + 0.3, visibility)  # hip
    return lm


HANG = pose(nose_y=0.6, elbow_offset=0.0)
TOP = pose(nose_y=0.15, elbow_offset=0.15)
MID = pose(nose_y=0.35, elbow_offset=0.12)


def run(frames: list[list[Point]]) -> RepCounter:
    c = RepCounter()
    for i, frame in enumerate(frames):
        c.update(frame, t=i * 0.5)
    c.finish()
    return c


def test_angle_straight_and_right():
    assert round(angle(Point(0, 0), Point(1, 0), Point(2, 0))) == 180
    assert round(angle(Point(0, 0), Point(1, 0), Point(1, 1))) == 90


def test_counts_full_reps():
    c = run([HANG, MID, TOP, MID, HANG, MID, TOP, HANG])
    assert c.count == 2
    assert c.bar_y is not None and abs(c.bar_y - 0.2) < 1e-9


def test_no_count_without_starting_from_hang():
    assert run([TOP]).count == 0


def test_staying_at_top_counts_once():
    assert run([HANG, TOP, TOP, MID, TOP]).count == 1


def test_half_rep_not_counted():
    assert run([HANG, MID, HANG, MID]).count == 0


def test_low_visibility_frames_ignored():
    assert run([HANG, pose(0.15, 0.15, visibility=0.1), HANG]).count == 0


def test_dropping_off_bar_at_end_adds_no_reps():
    # he lets go and stands after the set, this used to count extra reps
    standing_bent = pose(nose_y=0.5, elbow_offset=0.12, wrist_y=0.8)
    standing_straight = pose(nose_y=0.5, elbow_offset=0.0, wrist_y=0.9)
    c = run([HANG, TOP, HANG, standing_straight, standing_bent, standing_straight, standing_bent])
    assert c.count == 1
    assert c.state == "off"


def test_nose_must_pass_bar_not_just_bend_arms():
    below_bar = pose(nose_y=0.25, elbow_offset=0.15)
    assert run([HANG, below_bar, HANG, below_bar]).count == 0


def test_clean_rep_has_no_issues():
    c = run([HANG, MID, TOP, MID, HANG])
    assert [r.issues for r in c.reps] == [[]]
    assert c.clean_reps == 1


def test_form_flags_partial_lockout():
    # ~155 deg at the bottom, counts but not a full hang
    bent_hang = pose(nose_y=0.6, elbow_offset=0.055)
    c = run([bent_hang, TOP, bent_hang])
    assert c.count == 1
    assert "Arms not fully extended at bottom" in c.reps[0].issues


def test_form_flags_swing_and_uneven_arms():
    swinging_top = pose(nose_y=0.15, elbow_offset=0.15, right_elbow_offset=0.02, hip_dx=0.1)
    c = run([HANG, swinging_top, HANG])
    assert c.count == 1
    assert "Body swinging / kipping" in c.reps[0].issues
    assert "Uneven pull (one arm leads)" in c.reps[0].issues


def shifted(landmarks: list[Point], dx: float) -> list[Point]:
    return [Point(p.x + dx, p.y, p.visibility) for p in landmarks]


def standing(dx: float) -> list[Point]:
    # someone standing next to the bar, arms down
    lm = [Point(0.5 + dx, 0.5) for _ in range(33)]
    lm[0] = Point(0.5 + dx, 0.3)
    for i in (11, 12, 13, 14, 15, 16):
        lm[i] = Point(0.5 + dx + (0.05 if i % 2 == 0 else -0.05), 0.4 + (i - 11) // 2 * 0.15)
    return lm


def test_picks_hanging_person_over_standing_people():
    assert pick_athlete([standing(-0.3), HANG, standing(0.3)], None) == 1


def test_no_athlete_when_everyone_stands():
    assert pick_athlete([standing(-0.2), standing(0.2)], None) is None


def test_prefers_center_when_several_hang():
    assert pick_athlete([shifted(HANG, -0.3), shifted(HANG, 0.05)], None) == 1


def test_keeps_previous_athlete_when_off_center():
    # athlete at x=0.2 last frame, someone else hanging in the middle
    poses = [shifted(HANG, 0.0), shifted(TOP, -0.3)]
    assert pick_athlete(poses, previous=(0.2, 0.25)) == 1


def test_athlete_at_top_of_rep_still_qualifies():
    assert pick_athlete([standing(-0.3), TOP], None) == 1


def test_locked_athlete_is_never_swapped():
    # user clicked someone at x=0.2 who is now gone, don't jump to the guy in the middle
    assert pick_athlete([HANG], previous=(0.2, 0.25), locked=True) is None
    assert pick_athlete([HANG], previous=(0.2, 0.25)) == 0


def test_nearest_pose():
    poses = [shifted(HANG, -0.3), HANG]
    assert nearest_pose(poses, (0.5, 0.7))[0] == 1
    assert nearest_pose([], (0.5, 0.5))[0] is None


def test_new_bar_height_after_break():
    # camera moved between sets, bar is now lower in the picture
    lower = 0.3
    hang2 = [Point(p.x, p.y + lower, p.visibility) for p in HANG]
    top2 = [Point(p.x, p.y + lower, p.visibility) for p in TOP]
    standing_straight = pose(nose_y=0.5, elbow_offset=0.0, wrist_y=0.9)
    c = run([HANG, TOP, HANG, standing_straight, hang2, top2, hang2])
    assert c.count == 2
    assert abs(c.bar_y - 0.5) < 1e-9


def test_rep_timing():
    # frames are 0.5s apart: HANG t=0, MID 0.5, TOP 1.0, MID 1.5, HANG 2.0
    c = run([HANG, MID, TOP, MID, HANG])
    assert c.reps[0].pull_s == 1.0
    assert c.reps[0].lower_s == 1.0


def test_slowdown():
    fast = [HANG, TOP]
    slow = [HANG, MID, MID, MID, TOP]
    c = run(fast * 3 + slow * 3 + [HANG])
    assert c.count == 6
    assert abs(c.slowdown() - 3.0) < 1e-9  # 0.5s -> 2.0s per pull
    assert run(fast * 3 + [HANG]).slowdown() is None  # not enough reps


def test_break_starts_new_set():
    # frames are 0.5s apart, so 12 standing frames = 6s break
    standing_straight = pose(nose_y=0.5, elbow_offset=0.0, wrist_y=0.9)
    c = run([HANG, TOP, HANG, TOP, HANG] + [standing_straight] * 12 + [HANG, TOP, HANG])
    assert [r.set_no for r in c.reps] == [1, 1, 2]


def test_short_gap_is_same_set():
    standing_straight = pose(nose_y=0.5, elbow_offset=0.0, wrist_y=0.9)
    c = run([HANG, TOP, HANG, standing_straight, HANG, TOP, HANG])
    assert [r.set_no for r in c.reps] == [1, 1]


def test_set_count_includes_rep_in_progress():
    c = RepCounter()
    for i, frame in enumerate([HANG, TOP, HANG, TOP]):
        c.update(frame, t=i * 0.5)
    assert c.count == 2
    assert c.set_count == 2

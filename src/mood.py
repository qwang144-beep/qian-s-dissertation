"""
theta -> [MOOD] mapping, faithful to Kim & Choi (2024), Figure 1.

The paper places valence on x and arousal on y, then takes the angle
    theta = atan2(arousal, valence)   in radians, range (-pi, pi]
and assigns a mood label from the [MOOD] table in Figure 1.

The printed boundaries (0.17pi, 0.33pi, 0.67pi, 0.83pi, ...) are rounded
multiples of pi/6: the circle is partitioned into 12 equal 30-degree sectors.
We therefore key off exact pi/6 sectors rather than the rounded decimals, so
values that land near a boundary are classified consistently.

Sector layout (theta increasing from -pi to pi), each of width pi/6:
    [-pi   , -5pi/6) SAD        [0     ,  pi/6 ) PLEASED
    [-5pi/6, -4pi/6) BORED      [ pi/6 , 2pi/6 ) HAPPY
    [-4pi/6, -3pi/6) SLEEPY     [2pi/6 , 3pi/6 ) EXCITED
    [-3pi/6, -2pi/6) CALM       [3pi/6 , 4pi/6 ) ANNOYING
    [-2pi/6, -1pi/6) PEACEFUL   [4pi/6 , 5pi/6 ) ANGRY
    [-1pi/6,  0    ) RELAXED    [5pi/6 ,  pi   ) NERVOUS

Lower bound inclusive, upper bound exclusive (matching Figure 1's notation).
theta == pi exactly (pure negative valence, zero arousal) wraps to SAD.
"""
import math

# label for sector index k = floor(theta / (pi/6)), k in [-6, 5]
_SECTOR = {
    -6: "SAD",     -5: "BORED",   -4: "SLEEPY",  -3: "CALM",
    -2: "PEACEFUL", -1: "RELAXED",  0: "PLEASED",  1: "HAPPY",
     2: "EXCITED",  3: "ANNOYING",  4: "ANGRY",    5: "NERVOUS",
}


def valence_arousal_to_theta(valence: float, arousal: float) -> float:
    """theta = atan2(y, x) with x=valence, y=arousal. Range (-pi, pi]."""
    return math.atan2(arousal, valence)


def theta_to_mood(theta: float) -> str:
    """Map an angle (radians) to a Figure 1 [MOOD] label."""
    k = math.floor(theta / (math.pi / 6.0))
    if k == 6:          # theta == pi -> negative x-axis, wrap into SAD
        k = -6
    if k not in _SECTOR:
        raise ValueError(f"theta={theta} out of range (k={k})")
    return _SECTOR[k]


def mood(valence: float, arousal: float) -> str:
    """Convenience: valence/arousal -> mood label."""
    return theta_to_mood(valence_arousal_to_theta(valence, arousal))


if __name__ == "__main__":
    # 1) Paper's worked example (real Deezer values for Muse - Time Is Running Out)
    v, a = -1.04838590843, 0.335195092328
    th = valence_arousal_to_theta(v, a)
    print(f"Muse example: valence={v:.4f} arousal={a:.4f} "
          f"-> theta={th/math.pi:.2f}pi -> {theta_to_mood(th)}")
    assert abs(th / math.pi - 0.90) < 0.005, "theta should be ~0.90pi"
    assert theta_to_mood(th) == "NERVOUS", "paper says 'nervous mood'"

    # 2) One representative point in the middle of each of the 12 sectors
    print("\nSector centres (should walk through all 12 labels):")
    for k in range(-6, 6):
        centre = (k + 0.5) * (math.pi / 6.0)
        print(f"  theta={centre/math.pi:+.3f}pi -> {theta_to_mood(centre)}")

    # 3) Cardinal sanity checks
    print("\nCardinal directions:")
    for name, (vv, aa) in {
        "pure +valence (pleased/relaxed boundary=0)": (1.0, 0.0),
        "pure +arousal (excited/annoying boundary)":  (0.0, 1.0),
        "pure -valence (theta=pi -> SAD)":            (-1.0, 0.0),
        "pure -arousal (sleepy/calm boundary)":       (0.0, -1.0),
    }.items():
        print(f"  {name:44s} -> {mood(vv, aa)}")
    print("\nAll assertions passed.")

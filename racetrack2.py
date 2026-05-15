import numpy as np
import random
import matplotlib.pyplot as plt


# =============================================================================
# TRACK DEFINITIONS
# Cell values: 0=wall, 1=track, 2=start, 3=finish
# =============================================================================

def _build_simple():
    """Track 1: Simple L-shaped (original course assignment track)
    34x18 | ~6-8 cell wide | 1 turn | 298 drivable cells"""
    return np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])


def _build_medium():
    """Track 2: Medium S-shaped
    46x26 | ~3 cell wide | 2 turns | 199 drivable cells
    Path: start bottom → up → right → up → right → up → finish top-right"""
    g = np.zeros((46, 26), dtype=int)
    for r in range(10, 44):        # Segment 1: straight up
        for c in range(4, 7):
            g[r, c] = 1
    g[44, 4:7] = 2                 # start line
    for r in range(8, 12):         # Corner 1: turn right
        for c in range(4, 14):
            g[r, c] = 1
    for r in range(5, 12):         # Segment 2: straight up
        for c in range(11, 14):
            g[r, c] = 1
    for r in range(3, 7):          # Corner 2: turn right
        for c in range(11, 23):
            g[r, c] = 1
    for r in range(1, 7):          # Segment 3: straight up
        for c in range(20, 23):
            g[r, c] = 1
    g[1:4, 23] = 3                 # finish line
    return g


def _build_complex():
    """Track 3: Complex - Suzuka-inspired circuit
    52x38 | ~3 cell wide | 5+ turns | 515 drivable cells
    Features: long straight, S-curves (esses), hairpin, chicane, back straight"""
    g = np.zeros((52, 38), dtype=int)
    W = 3

    g[50, 3:3+W] = 2               # start line

    # Segment A: long straight UP (left side)
    for r in range(6, 50):
        g[r, 3:3+W] = 1

    # Corner A: top-left, turn right
    for r in range(4, 8):
        for c in range(3, 15):
            g[r, c] = 1

    # Segment B: short straight up
    for r in range(2, 8):
        g[r, 12:12+W] = 1

    # Esses (S-curves)
    for c in range(12, 21):        # Esse 1 horizontal
        g[2:2+W, c] = 1
    for r in range(2, 9):          # Esse 2 vertical
        g[r, 18:18+W] = 1
    for c in range(18, 29):        # Esse 3 horizontal
        g[6:6+W, c] = 1
    for r in range(2, 9):          # Esse 4 vertical
        g[r, 26:26+W] = 1
    for c in range(26, 35):        # Esse 5 horizontal
        g[2:2+W, c] = 1

    # Finish line (top-right)
    g[1:4, 35] = 3
    g[1:4, 34] = 1

    # Hairpin (top-right)
    for r in range(2, 13):         # right vertical
        g[r, 32:32+W] = 1
    for r in range(10, 14):        # hairpin base
        for c in range(28, 35):
            g[r, c] = 1

    # Segment C: long straight DOWN (right side)
    for r in range(12, 43):
        g[r, 28:28+W] = 1

    # Chicane (bottom-right)
    for r in range(40, 44):        # wide section
        for c in range(20, 30):
            g[r, c] = 1
    for r in range(38, 48):        # narrow exit
        g[r, 18:18+W] = 1

    # Back straight (bottom connector)
    for r in range(46, 50):
        for c in range(3, 21):
            g[r, c] = 1

    return g


TRACK_SIMPLE  = _build_simple()
TRACK_MEDIUM  = _build_medium()
TRACK_COMPLEX = _build_complex()


# =============================================================================
# TRACK SELECTION
# =============================================================================

def get_track(name="simple"):
    """
    Returns (track, starts, actions) for the specified track.
    Args:
        name: "simple", "medium", or "complex"
    """
    tracks = {"simple": TRACK_SIMPLE, "medium": TRACK_MEDIUM, "complex": TRACK_COMPLEX}
    if name not in tracks:
        raise ValueError(f"Unknown track '{name}'. Choose from: {list(tracks.keys())}")
    track   = tracks[name]
    starts  = tuple(map(tuple, np.argwhere(track == 2)))
    actions = tuple((ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1])
    return track, starts, actions


def get_racetrack_data():
    """Backward-compatible: returns the original Simple track."""
    return get_track("simple")


# =============================================================================
# ENVIRONMENT FUNCTIONS (identical to original course code)
# =============================================================================

def reset_env(starts, start_pos=None, start_index=None):
    """Returns initial env state: (y, x, vy, vx, start_y, start_x)"""
    if start_pos is not None:
        start_y, start_x = start_pos
    elif start_index is not None:
        start_y, start_x = starts[start_index % len(starts)]
    else:
        start_y, start_x = random.choice(starts)
    return (start_y, start_x, 0, 0, start_y, start_x)


def get_agent_state(env_state):
    """Extracts Q-table key: (y, x, vy, vx)."""
    return env_state[:4]


def bresenham_line(y0, x0, y1, x1):
    """All grid cells on the line from (y0,x0) to (y1,x1)."""
    points = []
    dy, dx = abs(y1-y0), abs(x1-x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dy - dx
    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dx:
            err -= dx; y0 += sy
        if e2 < dy:
            err += dy; x0 += sx
    return points


def step_env(track, env_state, action):
    """Apply action; return (new_env_state, reward, done)."""
    y, x, vy, vx, start_y, start_x = env_state
    ax, ay = action
    if random.random() < 0.1:
        ax, ay = 0, 0
    new_vx = max(0, min(4, vx + ax))
    new_vy = max(0, min(4, vy + ay))
    if new_vx == 0 and new_vy == 0:
        new_vx = max(0, min(4, vx + ax))
    new_y = y - new_vy
    new_x = x + new_vx
    for p_y, p_x in bresenham_line(y, x, new_y, new_x):
        if (p_y < 0 or p_y >= track.shape[0] or
                p_x < 0 or p_x >= track.shape[1] or
                track[p_y, p_x] == 0):
            return reset_env(None, start_pos=(start_y, start_x)), -1, False
        if track[p_y, p_x] == 3:
            return (p_y, p_x, new_vy, new_vx, start_y, start_x), 0, True
    return (new_y, new_x, new_vy, new_vx, start_y, start_x), -1, False


def get_action(Q, agent_state, actions, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.choice(actions)
    q_values = [Q[agent_state][a] for a in actions]
    max_q    = max(q_values)
    best     = [a for a, q in zip(actions, q_values) if q == max_q]
    return random.choice(best)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_trajectory(track, actions, Q, title, start_pos, color, ax):
    """Run greedy episode and plot trajectory on ax."""
    env_state = reset_env(None, start_pos=start_pos)
    path_y, path_x = [env_state[0]], [env_state[1]]
    done = False
    steps = 0
    while not done and steps < 300:
        agent_state = get_agent_state(env_state)
        best_a = actions[np.argmax([Q[agent_state][a] for a in actions])]
        next_state, _, done = step_env(track, env_state, best_a)
        if abs(next_state[0]-env_state[0]) > 5 or abs(next_state[1]-env_state[1]) > 5:
            break
        env_state = next_state
        path_y.append(env_state[0])
        path_x.append(env_state[1])
        steps += 1
    result = "reached finish" if done else "did not finish"
    ax.imshow(track, cmap='tab20c')
    ax.plot(path_x, path_y, marker='o', color=color, markersize=4, linewidth=2)
    ax.set_title(f"{title}\n{steps} steps — {result}", fontsize=10)
    ax.set_xticks(np.arange(-.5, track.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, track.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def preview_tracks():
    """Plot all three tracks side by side."""
    items = [
        ("Simple (L-shape)",       TRACK_SIMPLE),
        ("Medium (S-shape)",       TRACK_MEDIUM),
        ("Complex (Suzuka-style)", TRACK_COMPLEX),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.suptitle("Racetrack Complexity Levels", fontsize=16, fontweight='bold')
    for ax, (name, track) in zip(axes, items):
        n = np.sum(track > 0)
        ax.imshow(track, cmap='tab20c')
        ax.set_title(f"{name}\n{track.shape[0]}x{track.shape[1]}  |  {n} drivable cells", fontsize=11)
        ax.set_xticks(np.arange(-.5, track.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, track.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preview_tracks()

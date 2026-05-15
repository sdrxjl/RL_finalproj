# Does Algorithm Choice Matter?
## How Racetrack Complexity Modulates the Performance Gap Between SARSA and Q-Learning

**Group 16**  
GitHub: https://github.com/sdrxjl/RL_finalproj

---

# Introduction

This project uses the classic Racetrack environment to ask a focused question:

> *Does the choice between on-policy (SARSA) and off-policy (Q-Learning) temporal-difference control become more consequential as the environment grows more complex?*

Both algorithms are tabular TD methods that share the same $\varepsilon$-greedy exploration mechanism, differing only in which next-state value they bootstrap from.

- **SARSA** updates using the action actually taken, making it sensitive to exploratory noise.
- **Q-Learning** bootstraps from the greedy maximum, treating future behavior as if exploration never occurs.

This single structural difference—innocuous on simple tasks—produces dramatically divergent outcomes once the environment demands sustained precision across multiple turns.

---

# Problem Setup and Methodology

## Environment

The Racetrack simulator represents the driving surface as a 2-D NumPy array.

Cell values encode:

| Value | Meaning |
|---|---|
| 0 | Wall |
| 1 | Drivable track |
| 2 | Starting line |
| 3 | Finish line |

A state is represented as:

$$
(row, col, vel_{row}, vel_{col})
$$

Velocity components are non-negative and bounded below 5.

Each of the 9 actions applies a:

- $-1$
- $0$
- $+1$

increment to each velocity axis.

The reward is:

$$
-1
$$

at every step.

Additional environment rules:

- Hitting a wall resets the car to a random start with zero velocity while the episode continues.
- With probability $0.1$, the intended acceleration is ignored (stochastic noise).
- Episodes terminate at the finish line or after 1,000 steps.

---

## Track Complexity

Track complexity is defined prior to training by three structural properties:

1. Number of turns
2. Corridor width
3. Overall layout

This definition is independent of algorithmic outcomes.

| Track | Shape | Turns | Width | Drivable Cells | Grid Size |
|---|---|---|---|---|---|
| Simple | L-shape | 1 | Wide (~4) | 298 | 34 × 18 |
| Medium | S-shape | 2 | Narrow (~3) | 199 | 46 × 26 |
| Complex | Multi-turn circuit | 5+ | Narrow (~2–3) | 515 | 52 × 38 |

---

## Algorithms

### SARSA (On-Policy)

A single $\varepsilon$-greedy policy:

$$
\varepsilon = 0.1
$$

both generates experience and is evaluated.

Update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl[r + \gamma Q(s',a') - Q(s,a)\Bigr]
$$

where:

- $a'$ is the action actually selected by the behavior policy.

Because exploration noise contaminates the bootstrap target, SARSA learns to anticipate its own random actions, producing conservative trajectories that avoid states near walls.

---

### Q-Learning (Off-Policy)

Q-Learning replaces $Q(s',a')$ with:

$$
\max_a Q(s',a)
$$

Update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr]
$$

Q-Learning can therefore discover shorter, more aggressive paths, and its value estimates remain uncontaminated by exploratory deviations.

---

## Evaluation Metrics

Both algorithms were trained for:

$$
40,000
$$

episodes under identical hyperparameters:

| Parameter | Value |
|---|---|
| $\varepsilon$ | 0.1 (fixed) |
| $\gamma$ | 1.0 |
| $\alpha$ | 0.5 |
| Seed | 0 |

Policy quality was assessed at:

$$
800
$$

greedy evaluation checkpoints.

Evaluation starts from velocity state:

$$
(v_y = 2,\ v_x = 0)
$$

Metrics reported:

1. **Finish rate (%fin)**  
   Fraction of checkpoints where the greedy policy reached the finish line.

2. **Average steps (avg_steps)**  
   Mean steps on successful runs.

3. **Looping rate**  
   Fraction of checkpoints where the agent entered oscillation cycles without finishing.

---

# Results

| Track | Algorithm | Finish Rate | Loop Rate | Avg Steps | Gap (Q − SARSA) |
|---|---|---|---|---|---|
| Simple | SARSA | 90.1% | 9.9% | 16.9 | — |
| Simple | Q-Learning | 94.0% | 6.0% | 14.3 | +3.9 pp |
| Medium | SARSA | 85.9% | 14.1% | 25.2 | — |
| Medium | Q-Learning | 94.1% | 5.9% | 18.9 | +8.2 pp |
| Complex | SARSA | 65.2% | 34.8% | 44.2 | — |
| Complex | Q-Learning | 89.2% | 10.8% | 28.6 | +24.0 pp |

The most striking finding is that the performance gap between Q-Learning and SARSA grows monotonically with track complexity:

- **Simple:** +3.9 pp
- **Medium:** +8.2 pp
- **Complex:** +24.0 pp

On the Simple track the two algorithms are nearly interchangeable.

On the Complex track:

- SARSA fails to finish more than one-third of evaluation episodes.
- Q-Learning remains above 89%.

The divergence is driven primarily by looping.

SARSA looping rates:

- 9.9% → 34.8%

Q-Learning looping rates:

- 6.0% → 10.8%

This suggests SARSA’s conservatism becomes a liability as environments demand sustained directional commitment.

---

# Analysis: Why Complexity Hurts SARSA More

The performance gap can be traced through a causal chain of three compounding mechanisms.

---

## Step 1 — Corners Amplify Exploration Risk

Near a wall or turn, a random exploratory action has a much higher probability of causing a collision than the same action on a straight segment.

SARSA’s Q-values absorb this collision cost because updates use the action actually taken, including exploratory actions.

As a result:

- SARSA penalizes speed near corners.
- Q-Learning ignores exploratory deviations in the bootstrap target.

---

## Step 2 — Speed Makes Errors Worse

The Racetrack environment contains momentum.

A car traveling at:

$$
(v_y = 3,\ v_x = 2)
$$

cannot stop in one step.

Approaching a corner with excessive speed therefore causes collisions even if the next action is correct.

SARSA internalizes this through exploratory failures and learns extremely conservative approach velocities.

Q-Learning, by contrast, maintains higher target velocities.

---

## Step 3 — Slowing Down Triggers Paralysis

A near-zero-velocity car at a corner faces a difficult state:

- Accelerating risks collision.
- Not accelerating prevents progress.

SARSA therefore assigns similarly negative Q-values to many actions, producing oscillation loops.

Q-Learning avoids this because the bootstrap target always reflects the best possible outcome, preserving a directional value gradient toward the finish.

Policy-map inspection supports this interpretation:

- SARSA maps contain stuck states and oscillation loops.
- Q-Learning maps preserve coherent directional flow.

---

# Discussion and Limitations

These findings generalize beyond racetracks.

Any environment where exploratory actions carry asymmetric costs may amplify the SARSA–Q-Learning gap, including:

- Narrow corridors
- Cliff-edge tasks
- Precision assembly problems

However:

- SARSA’s conservatism may be preferable when exploratory mistakes are merely inefficient rather than catastrophic.
- Q-Learning may be superior when correctness matters more than caution, such as:
  - Autonomous driving
  - Robotic manipulation
  - High-frequency trading

### Limitations

1. Policy quality is evaluated at only one velocity state:

$$
(v_y = 2,\ v_x = 0)
$$

2. Looping is inferred from trajectories rather than directly from Q-value structure.

3. Fixed $\varepsilon$ may exaggerate SARSA conservatism relative to annealed exploration schedules.

Future work should evaluate:

- $\varepsilon$ annealing
- Full state-space coverage

---

# Conclusion

In simple environments, algorithm choice barely matters:

- SARSA and Q-Learning differ by less than 4 percentage points.

In complex environments:

- The gap grows to 24 percentage points.

This divergence is driven by SARSA’s tendency to translate exploration risk into excessive conservatism and oscillatory behavior.

Q-Learning’s optimistic bootstrap preserves a reliable value gradient regardless of track complexity, allowing it to maintain:

$$> 89 \%  $$

finish rates even on the most demanding layouts.

The practical implication is clear:

> In environments with narrow margins for error, off-policy control is the more robust choice.

---

# References

Sutton, R. S., & Barto, A. G. (2018).  
*Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
http://incompleteideas.net/book/the-book-2nd.html
# Gate 0 — Invent the Ruler

## Question

Can a learner that never sees a world's generating variables invent a compact coordinate system in which held-out interventional dynamics are easier to predict than in raw sensor coordinates?

## Why this gate exists

The old style of law extractor usually receives candidate observables chosen by us. If the world was generated from `phi`, a Laplacian and an energy-like expression, then asking symbolic regression to rediscover combinations of those quantities is useful engineering but weak evidence of concept discovery.

Gate 0 moves the boundary backward. The scientist receives an observation stream whose channels have no privileged semantic labels. It must build its own internal ruler.

## Separation of roles

### World

`lawextractor/worlds/hidden_rotor.py`

Owns the generating state and sensor rendering. It is allowed to know the truth.

### Collector

`lawextractor/data.py`

Runs interventions and writes two files. The public file contains only observable transitions. The private file is an evaluator receipt.

### Scientist

`lawextractor/scientist.py`

Imports no world code. It learns candidate coordinates from `obs, action, next_obs` only.

### Evaluator

`lawextractor/evaluate.py`

First compares public predictive performance. Only afterward, optionally, it opens the private receipt and measures whether the invented coordinate is linearly related to the generating state.

## The deliberate bias

Gate 0 is **not ontology-free**.

It defines one notion of simplicity:

```text
z(t+1) = small linear law(z(t), intervention(t))
```

The encoder is free to invent a nonlinear coordinate system, but the law family is fixed. This makes Gate 0 a test of **coordinate invention under a simplicity prior**, not general law invention.

That limitation is the point of the next gate.

## Receipts

A candidate is useful only if it survives all of these:

1. **Held-out episodes** — not training reconstruction.
2. **Stronger held-out interventions** — test kicks are larger than train kicks.
3. **Open-loop rollout** — initialize once, then stop correcting the model from reality.
4. **Raw-coordinate baseline** — a ridge-fitted linear law receives the same public observations and interventions.
5. **Complexity accounting** — report the number of law parameters, not just error.
6. **Private alignment only after training** — hidden state is an evaluator diagnostic, never a target.

## Kill conditions

Gate 0 should be considered dead, not rescued by prettier plots, if across seeds:

- the raw sensor-space baseline matches or beats the invented coordinate on open-loop interventional rollouts at comparable complexity;
- the latent model only wins one-step prediction and diverges when rolled forward;
- results disappear when the sensor rendering is changed;
- success requires choosing the true latent dimensionality by hand;
- the scientist accidentally receives `private_truth.npz` or imports the world implementation.

The current seed-0 development run passes the first implementation check. It does **not** establish robustness across worlds or seeds.

## Why the 3-D winner is encouraging

The hidden generating state is 2-D, but the selected Gate 0 ruler is 3-D.

That prevents the cleanest possible self-deception: we did not simply tell the system "there are two real variables" and celebrate when it returned two. The third coordinate may be redundant, may make the nonlinear sensor chart easier to decode, or may reflect an optimization artifact. Gate 0 does not yet know.

That ambiguity is useful. A real law maker must be allowed to invent coordinates that are not one-to-one with a human's preferred hidden variables.

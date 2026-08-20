# LawExtractor

**Invent the ruler before fitting the law.**

Most automated law discovery starts after a human has already done the hardest part. We hand it variables called position, velocity, energy, frequency, graph mode, pressure, or voltage and ask it to find a compact equation.

LawExtractor starts one step earlier:

> **Given only awkward observations and the ability to intervene, can a machine invent a measurement language in which the world becomes simpler?**

The long-term target is not symbolic regression over a vocabulary we supplied. It is an artificial scientist that can:

```text
WORLD
  ↓
raw observation
  ↓
invent candidate observables / coordinates
  ↓
intervene
  ↓
keep the coordinates that make counterfactual prediction simple
  ↓
compress the surviving regularity into an executable law
  ↓
only then try to translate it into human mathematics
```

This repository begins with **Gate 0: Invent the Ruler**.

## Gate 0 — Invent the Ruler

Gate 0 contains a tiny hidden world with a deliberately awkward 12-channel sensorium. The true world has a compact internal state and a simple controlled update, but that state is never exposed to the scientist.

Collection produces two physically separate files:

```text
public.npz
    obs
    action
    next_obs
    split
    episode

private_truth.npz
    hidden
    next_hidden
    split
    episode
```

The discovery code reads **only `public.npz`**.

It trains several candidate rulers with different latent dimensionalities. Each ruler is a learned nonlinear encoder/decoder wrapped around a deliberately tiny latent law:

```text
awkward sensors
      ↓
 learned ruler
      ↓
      z
      ↓
 small executable law + intervention
      ↓
     z'
      ↓
 learned ruler
      ↓
predicted sensors
```

The candidate is judged on held-out episodes and interventions. A one-step fit is not enough: the harder receipt is an **open-loop rollout** in which the model gets no correction from reality for 30 steps.

Only after discovery is finished may the evaluator open `private_truth.npz` and ask whether the invented coordinates happen to line up with the hidden state that generated the world.

## Development receipt

A deterministic development run (`seed=0`, 450 epochs, latent dimensions 1–4) produced:

```text
raw sensor-space linear law
    law parameters:          168
    held-out one-step MSE:   0.1494
    open-loop 30-step MSE:   0.4482

invented-coordinate law
    chosen latent dimension: 3
    law parameters:          15
    held-out one-step MSE:   0.1157
    open-loop 30-step MSE:   0.0819

30-step rollout improvement: ~5.47x
private evaluator R²(z → true hidden state): ~0.87
```

The interesting part is not that the hidden world is difficult. It is intentionally tiny. The receipt is that a machine given only the ugly sensors found a small internal coordinate system in which a much smaller law survives stronger, unseen interventions substantially better over time.

Also note the imperfection: the hidden world has two true variables, while the winning scientist used three. Gate 0 is not recovering a privileged answer by construction.

## Run it

Python 3.10+ with NumPy and PyTorch:

```bash
pip install -e .
python -m lawextractor.cli gate0 --epochs 450 --dims 1,2,3,4 --seed 0
```

Or run the stages separately:

```bash
python -m lawextractor.cli collect --out runs/gate0/data
python -m lawextractor.cli discover \
    --data runs/gate0/data/public.npz \
    --out runs/gate0/discovery \
    --dims 1,2,3,4 \
    --epochs 450
python -m lawextractor.cli evaluate \
    --data runs/gate0/data/public.npz \
    --private runs/gate0/data/private_truth.npz \
    --model runs/gate0/discovery/candidate_z3.pt
```

Run tests with:

```bash
pytest -q
```

## What Gate 0 is *not*

It is not yet the machine described in the opening paragraph.

The largest cheat is explicit: Gate 0 tells the scientist that a desirable law is **linear in the invented coordinates**. The neural network is therefore learning a ruler that makes one particular kind of mathematics simple. That is already more honest than handing it the true variables, but mathematics is not yet the last observer through the door.

The next gates should remove that assumption rather than decorate this one.

See [`docs/GATE0.md`](docs/GATE0.md) for the kill conditions and [`HANDOFF.md`](HANDOFF.md) for the next hunches.

## Design rules

1. **World truth stays private.** A scientist may not import a simulator's implementation or hidden state.
2. **Prediction must survive intervention.** Correlation alone is not a law receipt.
3. **Long rollouts matter.** One-step prediction can reward locally convenient nonsense.
4. **The observer may invent coordinates.** Do not pre-name energy, modes, objects, frequencies, or other observables unless a gate is explicitly testing that bias.
5. **A failure is a result.** If a raw baseline wins, preserve the receipt and change the hypothesis.
6. **Mathematics comes after the representation survives.** Symbolic naming is an interpretation stage, not permission to redesign the world.

## Where this is intended to go

The same public interface should eventually accept things that are not toy worlds:

- a cellular or particle simulation whose source code is hidden from the scientist;
- a trained neural network exposed only through activations and interventions;
- a software system exposed through traces and perturbations;
- experimental time series with controlled perturbations;
- learned worlds where even the useful *objects* are not known in advance.

The ambition is simple to say and difficult to earn:

> **Find the coordinate system in which the thing becomes stupid. Then find out whether the stupidity survives when we poke it.**

# LawExtractor

**Point an instrument at something before deciding what the thing is.**

Live demo: https://anttiluode.github.io/LawExtractor/

`main/index.html` is now the clearest statement of what this repo became: a browser-only webcam instrument that sends the same raw field through many incompatible mathematical **rulers**, watches for peculiar simplicity, proposes cautious stories, and prints the experiment that would kill each story.

```text
raw phenomenon
      ↓
measurements
      ↓
many rulers
      ↓
peculiarity / GLINT
      ↓
educated guesses
      ↓
falsifiers
      ↓
probe again
```

The mathematics is deliberately **not** the first architect through the door. A lens is a proposed way of looking, not a claim about what is really there.

## What main does

The page downsamples a camera, screen share, or video into a small spatial field and keeps a rolling temporal history. Twenty-seven lenses then inspect the same observations from very different viewpoints, from ancient ratios, moments, triangulation, reciprocals and exhaustion to Fourier spectra, PCA modes, graph frequency, Koopman-like dynamics, recurrence, information flow, microstates, optical flow and rPPG.

The important output is not merely a dashboard of numbers. Three ideas matter:

### GLINT — one ruler suddenly works

Each lens keeps a history of its own score. A glint is logged when the world becomes unusually simple through that ruler compared with what that ruler normally sees. The current faceplate is still heuristic — native lens scores are different currencies and the bar heights should not be read as a rigorous cross-lens ranking.

The research direction is to make this a common evidence currency: compare each ruler against a proper null or its own calibrated baseline, then spend attention where the null unexpectedly fails.

### RESIDUAL — none of the current stories work

The instrument removes leading modes and asks whether what remains is still temporally structured. Large error is usually boring. **Error with repeatable timing is a possible missing cause.**

That is where a future LawExtractor should invent ruler 28 rather than merely choose among the existing 27.

### PROBE — known excitation → unknown medium

The page can drive a known 127-step grey-level sequence into the room and measure what comes back through the camera: correlation, delay, gain and asymmetry. This is the generalized `HeadAsResonator` move:

```text
known thing I do
      ↓
unknown physical path
      ↓
thing I can measure
```

The path may include the display, light, surfaces, air, lens, sensor and the camera's own exposure control. The instrument does not need to know that ontology in advance to measure a transfer relationship.

## The epistemic rule

Every generated story carries a **killer**: the next cheap observation that should make the story fail if it is wrong.

Examples include covering skin to challenge an rPPG interpretation, locking exposure to challenge a camera-control interpretation, occluding the middle of the frame to challenge propagation geometry, or turning the probe away to distinguish optical coupling from a shared electronic clock.

So the intended loop is:

```text
OBSERVE
  ↓
TRY MANY WAYS OF SEEING
  ↓
NOTICE A GLINT OR STRUCTURED RESIDUAL
  ↓
TELL A SMALL STORY
  ↓
ASK WHAT WOULD KILL IT
  ↓
INTERVENE
  ↓
KEEP / KILL / INVENT A NEW RULER
```

## Gate 0 still matters

The older runnable branch **`gate0-invent-the-ruler`** attacks a complementary problem. There the scientist sees only awkward observations plus interventions, while the generating truth is kept physically separate. It learns a coordinate system in which a deliberately small predictive law survives held-out counterfactual rollouts; only afterward is the hidden state opened as an evaluator receipt.

That branch asks:

> Can the scientist invent a better ruler for **states**?

The live main page asks:

> Can many rulers compete over raw observations, notice where one becomes useful, notice where all fail, and suggest the next experiment?

The eventual project is the combination: **an artificial observer that can inherit rulers, calibrate them, invent new ones, and use intervention to discover which descriptions of an unfamiliar system actually buy predictive leverage.**

## Caveats

This is an experimental thinking instrument, not an autonomous scientist. The current guesses are hand-authored combinations of measured cues. Correlation-built graphs are not anatomy. Camera auto-exposure and white balance are part of the measurement. Nothing above the measured Nyquist rate is established. A glint is a reason to investigate, not evidence that a proposed ontology is true.

That caution is part of the project rather than a disclaimer bolted onto it:

> **The world is allowed to be uglier than the mathematics we hoped to find.**

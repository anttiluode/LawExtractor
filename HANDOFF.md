# Handoff — after Gate 0

Gate 0 works well enough to attack its own assumption.

## What exists

A hidden simulator emits awkward observations. A collector keeps generating truth physically separate from public observations. A neural scientist learns a nonlinear coordinate system around a small linear intervention-aware law. The evaluator compares it with a raw-coordinate baseline and then, only as a receipt, checks alignment with hidden state.

On the development seed the invented coordinate uses 15 law parameters versus 168 for the raw law and is about 5.5x better on a 30-step open-loop held-out-intervention rollout.

## Do not spend the next branch polishing Gate 0

The obvious danger is to make better autoencoders, bigger sweeps, nicer plots and a GUI. That would turn LawExtractor into another representation-learning demo.

The live question is earlier.

## Gate 1 hunch — let the scientist invent what "simple law" means

Gate 0 says:

> Find coordinates that make a linear controlled dynamical law work.

Gate 1 should say something closer to:

> Find coordinates **and** a small executable update program whose combined description predicts new interventions.

Do not begin with a symbolic vocabulary such as `sin`, `x²`, gradients, Laplacians, energy, Fourier modes, etc.

A possible route is a tiny differentiable program machine with generic primitives that are not named as physics: copy, add, multiply, gate, delay, compare, route, local memory. The extractor must pay for every primitive and every invented coordinate. Competing programs are tested on counterfactual rollouts.

The success condition is not whether the discovered program looks elegant to us. It is whether it predicts unseen interventions with a shorter executable description than the alternatives.

## Gate 2 hunch — the scientist chooses the experiment

Gate 0 receives random interventions.

A law extractor should eventually ask:

> Which poke would make my competing explanations disagree most?

Give it a fixed intervention budget. Let it maintain several candidate rulers/laws. Reward interventions that eliminate hypotheses per unit cost. This is where the project becomes an experimental scientist rather than a passive representation learner.

## Gate 3 hunch — remove the known object boundary

The current public observation is a vector with stable channels. That is already a gift.

Render a world as pixels or event clouds. Do not tell the scientist where one persistent thing ends and another begins. Ask whether it can invent a decomposition that improves intervention prediction.

This is the first serious test of the sentence:

> The hard part may be discovering what is worth measuring.

## First external target after the toys

A small frozen transformer is unusually attractive because we can intervene almost everywhere while still hiding our preferred vocabulary from the scientist.

Do **not** initially expose names such as attention head, MLP, induction circuit or residual stream as explanatory objects. Record activations and permit controlled perturbations through a thin adapter. Ask LawExtractor to invent a lower-dimensional set of state variables/programs that predicts the consequences of those perturbations.

If it merely rediscovers architectural boundaries, that is a calibration result. If stable predictive objects cut across our modules, that is where the project becomes genuinely interesting.

## One rule to carry forward

**Never modify the world because the discovered law is ugly.**

The scientist is allowed to fail. The world is not obliged to become mathematically convenient.

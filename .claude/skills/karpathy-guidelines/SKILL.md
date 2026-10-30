---
name: karpathy-guidelines
description: Behavioral guidelines to reduce common LLM coding mistakes (inspired by Andrej Karpathy's observations). Use at the start of every coding task in this repo -- writing, reviewing, debugging or refactoring code.
---

# Karpathy coding guidelines

LLMs tend to make wrong assumptions silently, overcomplicate code, touch code they were not asked to touch,
and declare victory without checking. These four rules counter that. They favour caution over speed;
for truly trivial tasks, use judgment.

## 1. Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If a request has several interpretations, present them -- don't pick one silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

## 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for code that is used once.
- No "flexibility" or "configurability" that nobody requested.
- No error handling for scenarios that cannot happen.
- If you wrote 200 lines and it could be 50, rewrite it.

Test: would a senior engineer call this overcomplicated? If yes, simplify.

## 3. Surgical changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments or formatting.
- Don't refactor things that aren't broken.
- Match the existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it -- don't delete it.
- Remove imports/variables/functions that YOUR change made unused; leave pre-existing dead code alone.

Test: every changed line should trace directly to the user's request.

## 4. Goal-driven execution

Define success criteria. Loop until verified.

Turn tasks into verifiable goals:
- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan with a check per step:

```
1. [step] -> verify: [check]
2. [step] -> verify: [check]
```

"It ran without crashing" is not the same as "it works": check the output actually shows the intended effect
(e.g. a training run whose grad_norm is 0 at every step did not train).

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Basketball Analytics & Computer Vision Specifics

**Respect coordinate spaces, hardware constraints, and tensor geometry.**

- **Coordinate Integrity:** Always clarify what space coordinates are in (`pixel xy`, `normalized 0-1`, or `court feet`). Never guess or blend them silently.
- **Tensor Device Safety:** When editing model logic, match the existing hardware device target (`device="mps"`). Do not let tensors accidentally drop back to CPU mid-pipeline.
- **Fail Gracefully on Occlusions:** In sports tracking, detections *will* drop. Assume missing keypoints, missing bounding boxes, and missing track IDs are normal scenarios, not exceptional errors. Fall back safely to bounding box centers or `-1` IDs without crashing the stream.
- **Leverage Math Libraries:** Do not write scratch loops for geometric calculations. Use `numpy` or `opencv` (`cv2`) for matrix transformations and Euclidean distance checks.
- **Strict Frame-Rate Awareness:** If adding temporal logic or post-processing filters, always check the frame rate assumptions (e.g., assuming 30 FPS means 60 frames = 2 seconds).

## 6. Project Goal & Data Architecture

**This is a headless data extraction pipeline for basketball analytics.**

- **The Objective:** Process raw court video to output structured tracking data (`_dynamic.json`) containing player bounding boxes, skeletons, and court-space feet coordinates.
- **No UI/Rendering Bias:** The pipeline runs headless. Do not add `cv2.imshow()`, visual overlays, or video-writing logic unless explicitly requested. Optimize purely for extraction speed and data accuracy.
- **Downstream Dependency:** The JSON structure is strict. Every row must represent a flat ledger entry matching the existing schema (`frame`, `player_id`, `court_coordinates_feet`, `court_side`, `bbox`, `skeleton`, `paint_corners`). Never alter these keys or nest them differently.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
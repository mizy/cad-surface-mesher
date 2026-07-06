# Code Principles

## Core Goal: Vehicle CAX Tools

Keep each capability easy to find by responsibility:

- `predict-vehicle-cd`: fast geometry-similarity Cd prediction from vehicle meshes.
- `cad-surface-mesher`: target-driven CAD surface mesh audit and visual QA.
- `mesh-repair`: automatic watertight surface mesh repair and diagnostics.
- `cad-tessellation`: CAD import and tessellation embedding.

## Boundaries

- Keep this repository personal and standalone. Do not depend on private work repositories.
- Do not commit credentials, service endpoints, customer data, or proprietary meshes.
- Packaged reference data must be public, synthetic, or explicitly allowed for personal use.
- Prefer local CLI tools and file artifacts over hidden service calls.

## Architecture

- Group code and docs by owned responsibility, not artifact type.
- A feature owns its assets, scripts, references, generated examples, and validation notes.
- Shared utilities should be introduced only when at least two real features need them.
- Keep generated caches, mesh outputs, screenshots, and temporary reports out of git unless they are intentional fixtures.
- Agent skills live under `.agents/skills/<skill-name>` and own their scripts, references, assets, and `SKILL.md` there.

## Style

- Keep scripts direct and inspectable.
- Use clear names such as `predict-vehicle-cd`, `mesh-repair`, `cad-tessellation`, `assets`, `references`, and `scripts`.
- Avoid wrapper layers that only rename or forward calls.
- Document assumptions around units, axes, mesh validity, and output quality.

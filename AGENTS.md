# Code Principles

## Core Goal: CAD Surface Meshing

Keep each capability easy to find by responsibility and colocated inside its skill:

- `predict-vehicle-cd`: fast geometry-similarity Cd prediction from vehicle meshes.
- `mesh-watertight-repair`: build a geometry-driven watertight exterior shell from a mesh input.
- `surface-tessellation`: convert CAD or mesh inputs to triangle-only VTP surface meshes.
- `cad-surface-mesher`: target-driven CAD surface mesh audit and visual QA; orchestrates sibling skills for tessellation and watertight repair.

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
- Agent skills live under `.agents/skills/<skill-name>` and own their scripts, references, assets, and `SKILL.md` there. No business logic lives outside `.agents/skills/`.

## Style

- Keep scripts direct and inspectable.
- Use clear names such as `cad-surface-mesher`, `predict-vehicle-cd`, `mesh-repair`, `cad-tessellation`, `assets`, `references`, and `scripts`.
- Avoid wrapper layers that only rename or forward calls.
- Document assumptions around units, axes, mesh validity, and output quality.

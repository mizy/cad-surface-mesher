# CAD Surface Mesher

Personal toolkit for CAD-to-watertight-surface meshing, repair planning, tessellation, and fast vehicle geometry analysis.

All logic lives inside **agent skills** under `.agents/skills/<skill-name>`. Each skill owns its `SKILL.md`, scripts, references, assets, and documentation.

## Skills

### predict-vehicle-cd

Fast geometry-similarity Cd prediction from vehicle meshes.

```bash
cd .agents/skills/predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --output-dir /tmp/cd_estimate
```

### mesh-watertight-repair

Build, audit, and validate a geometry-driven watertight exterior shell from a mesh input.

```bash
cd .agents/skills/mesh-watertight-repair
python scripts/two_stage_watertight_remesh.py /path/to/input.vtp \
  --output-dir /tmp/watertight-output
```

### surface-tessellation

Convert CAD (STEP, IGES, BREP) or mesh (STL, OBJ, VTP, VTK, GLB, GLTF) into triangle-only VTP surface meshes.

```bash
cd .agents/skills/surface-tessellation
python scripts/cad_tessellate.py tessellate /path/to/model.step \
  --output-dir /tmp/tessellation \
  --mesh-size 0.05 \
  --angle-deg 18 \
  --chord 0.005
```

### cad-surface-mesher

Target-driven CAD surface mesh audit and visual QA.  
Orchestrates `surface-tessellation` (CAD/mesh → VTP), `mesh-watertight-repair` (mesh → watertight shell), and its own `scripts/audit_surface_mesh.py` for deterministic audit and screenshot generation.

```bash
cd .agents/skills/cad-surface-mesher
python scripts/audit_surface_mesh.py /path/to/car.vtp --output-dir /tmp/cad-surface-audit
```

See each skill's `SKILL.md` for detailed documentation.

## Dependencies

```bash
python -m pip install -r requirements.txt
```

Optional `trimesh` improves GLB/GLTF loading.

## Repository Layout

```text
.agents/skills/predict-vehicle-cd/          Cd prediction (standalone)
.agents/skills/mesh-watertight-repair/      Mesh → watertight exterior shell
.agents/skills/surface-tessellation/        CAD/mesh → VTP surface mesh
.agents/skills/cad-surface-mesher/          CAD surface meshing orchestrator
  ├── SKILL.md                              AI agent entrypoint
  ├── scripts/                              Shared scripts (audit, etc.)
  └── docs/                                 Roadmap and cross-feature notes
```

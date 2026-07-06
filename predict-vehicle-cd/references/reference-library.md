# Reference Library Schema

`assets/reference-library/reference_cars.json` is the packaged prediction library. It contains generated side-view artifacts and Cd labels.

## Fields

- `version`: library schema version. Version 2 stores the side silhouette in the depth PNG alpha channel.
- `image_width`, `image_height`: artifact dimensions.
- `view`: currently `side_-y`.
- `cars[]`: reference vehicle records.
- `cars[].id`: stable identifier.
- `cars[].display_name`: human-readable name.
- `cars[].category`: broad body class.
- `cars[].cd`: drag coefficient label used for prediction.
- `cars[].cd_source`: provenance and relevant test/source condition for the Cd value.
- `cars[].cd_confidence`: `seed`, `derived`, or `authoritative`.
- `cars[].mesh_source`: optional source URL or project path for third-party/local mesh provenance.
- `cars[].mesh_license`: optional mesh license, required for third-party assets.
- `cars[].mesh_author`: optional mesh author or publisher attribution.
- `cars[].mesh_note`: optional mesh/model-year caveat.
- `cars[].artifacts.depth`: relative path to side-view depth PNG; alpha channel is the side-view mask.
- `cars[].artifacts.normal`: relative path to side-view normal PNG.
- `cars[].artifacts.mask`: optional legacy relative path to side-view mask PNG for version 1 libraries.
- `seed_manifest.pending_cars[]`: Cd-vetted records waiting for a PyVista-readable mesh; these are not packaged into `reference_cars.json`.

## Editing Rules

- Keep `cd` numeric for prediction records.
- Set `cd_confidence` to `authoritative` only for CFD, wind-tunnel, OEM/public manufacturer specification, or project-approved labels.
- Record enough source context to distinguish different rear-end, cooling, wheel, ground, and model-year conditions.
- Move a `pending_cars[]` record into `cars[]` only after a readable mesh is present and mesh provenance is recorded.
- Regenerate artifacts with `build-library` after changing a reference mesh or rasterization settings.
- Keep generated artifact paths relative to `reference_cars.json`.

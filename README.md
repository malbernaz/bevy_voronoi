![demo](https://github.com/malbernaz/bevy_voronoi/raw/main/static/demo.png)

# `bevy_voronoi`

A low-level **Bevy** plugin for generating **Voronoi diagrams** from 2D meshes and textures.

## **Features**

- Generates **Voronoi diagrams** from any `Mesh2d`.
- Supports **alpha masks** for transparency and occlusion.
- Uses the **Jump Flood Algorithm (JFA)** for efficient computation.
- Attaches a **VoronoiTexture** component to the view entity in the render world with the **fragment coordinates** for the diagram and the original alpha mask.

## Usage

See the examples folder for usage details.


## Compatibility

| bevy   | bevy_voronoi |
| ------ | ------------ |
| `0.16` | `0.3`        |
| `0.15` | `0.1..0.2`   |

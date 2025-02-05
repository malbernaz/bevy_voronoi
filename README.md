# `bevy_voronoi`

A low-level **Bevy** plugin for generating **Voronoi diagrams** from 2D meshes.

## **Features**

- Generates **Voronoi diagrams** from any `Mesh2d`.
- Supports **alpha masks** for transparency and occlusion.
- Uses the **Jump Flood Algorithm (JFA)** for efficient computation.
- Outputs a **VoronoiTexture** in the render world with **UV coordinates** for the diagram.

## Usage

See the examples folder for usage details.


## Compatibility

| bevy   | bevy_voronoi |
| ------ | ------------ |
| `0.15` | `0.1`        |

<br />

![demo](https://github.com/malbernaz/bevy_voronoi/raw/main/static/demo.png)

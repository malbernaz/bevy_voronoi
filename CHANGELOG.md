# Changelog

## 0.4

## Breaking
- replace property `down_sample` for `scale` for clarity.

## Fix
- make sure the alpha mask is loaded

## 0.3.0

## Features
- Bevy 0.16 support
- Introduces a mandatory `VoronoiCamera` that can control the `VoronoiTexture` sampling factor
- Expose original alpha mask on the alpha channel

## Breaking
- Since the original alpha mask is already exposed the blue channel is not used anymore

## 0.2.0

## Features
- Proper signed distances (`VoronoiTexture`'s blue channel indicates if distance is inside mask)
- Additional JFA pass for improved accuracy

## Breaking
- `VoronoiTexture` output fragment instead of UV coordinates

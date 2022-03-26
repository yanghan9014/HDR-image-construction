# VFX Project 1: HDR-image-construction
Image alignment, High Dynamic Range image construction and tone mapping
---
B08901054 楊學翰 B08901058 陳宏恩

## basic usage
```
python3 main.py --input [input directory] --key [photographic tonemapping key]
# example: 
# python3 main.py --input test --key 0.9
```
## Project overview
- Several photos with different exposures are taken with relatively little disturbance
- The photos are passed through the alignment algorithms (we implemented the MTB alignment in this case)
- Retrieve the response curve of the camera
- Recover hdr image and export .hdr file
- Tone map the hdr image (we implemented the photographic global operator) and recover displable image

## Algorithm implementations
1. MTB image alignment
2. HDR image construction: Debevec's Method
3. Tone mapping: Photographic (global operation)

## Implementation details
To accommodate the difference of exposure and ISO speed, we take the product of exposure time and ISO speed as Δt

## Comparisons
Our comparisons are based on the tools implemented in https://viewer.openhdr.org/
| Photographic (ours) | Photographic | 
| -------- | -------- | 
|      |      | 


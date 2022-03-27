# VFX Project 1: HDR-image-construction
Image alignment, High Dynamic Range image construction and tone mapping
---
B08901054 楊學翰 B08901058 陳宏恩

## basic usage
```
python3 main.py --input [input directory] --key [photographic tonemapping key]
# example: 
# python3 code/main.py --input street_lamps --key 0.8
```
## Project overview
- Several photos with different exposures are taken with relatively little disturbance
- The photos are passed through the alignment algorithms (we implemented the MTB alignment in this case)
- Retrieve the response curve of the camera
- Recover hdr image and export .hdr file
- Tone map the hdr image (we implemented the photographic global operator) and recover displable image

## Algorithm implementations
In this project, we implemented the following algorithms
### MTB image alignment
- We slightly modified the MTB image alignment. It matches the *gradient* of a pair of image using the pyrimid method from MTB.
- Alignment may fail for images that are too dark
### HDR image construction: Debevec's Method
- We set the g function's smoothness to 10 by default, and randomly sample 1000 points to compute the camera response function
- Some photos are taken with different ISO speed. To accommodate the difference, we take the product of exposure time and ISO speed as the final Δt
- The HDR radience map is composited using the average camera response function of the three color channels
### Tone mapping: Photographic (global operation)
- We set the key to 0.7 by default
- Lwhite is set to the maximum luminance

## Response curve
We randomly selected 1000 pixels to compute the response curve
![response_curve](https://user-images.githubusercontent.com/62785735/160246864-4e986a67-46cc-47fe-b6ee-7bcb28936ed6.png)

## Intensity heatmap
![intensity](https://user-images.githubusercontent.com/62785735/160284502-84ee7a70-1fdc-4689-9a3b-d892f89dee3b.jpg)

## Final results
(Photographic global operation key = 0.8)
![our_tonemap](https://user-images.githubusercontent.com/62785735/160285011-1f6fed9e-87e9-468a-84fc-5c37c1ca58dd.jpg)

## Requirements
To install dependencies from requirements.txt:
```
pip install -r requirements.txt
```

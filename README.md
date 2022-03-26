# VFX Project 1: HDR-image-construction
Image alignment, High Dynamic Range image construction and tone mapping
---
B08901054 楊學翰 B08901058 陳宏恩

## basic usage
```
python3 main.py --input [input directory] --key [photographic tonemapping key]
# example: 
# python3 code/main.py --input data --key 0.9
```
## Project overview
- Several photos with different exposures are taken with relatively little disturbance
- The photos are passed through the alignment algorithms (we implemented the MTB alignment in this case)
- Retrieve the response curve of the camera
- Recover hdr image and export .hdr file
- Tone map the hdr image (we implemented the photographic global operator) and recover displable image

## Algorithm implementations
We implemented these algorithms
1. MTB image alignment
2. HDR image construction: Debevec's Method
3. Tone mapping: Photographic (global operation)

## Implementation details
To accommodate the difference of exposure and ISO speed, we take the product of exposure time and ISO speed as Δt

## Response curve
![response_curve](https://user-images.githubusercontent.com/62785735/160246864-4e986a67-46cc-47fe-b6ee-7bcb28936ed6.png)

## Requirements
action-tutorials-py==0.9.3
ament-clang-format==0.9.6
ament-clang-tidy==0.9.6
ament-copyright==0.9.6
ament-cppcheck==0.9.6
ament-cpplint==0.9.6
ament-flake8==0.9.6
ament-index-python==1.0.1
ament-lint==0.9.6
ament-lint-cmake==0.9.6
ament-mypy==0.9.6
ament-package==0.9.3
ament-pclint==0.9.6
ament-pep257==0.9.6
ament-pycodestyle==0.9.6
ament-pyflakes==0.9.6
ament-uncrustify==0.9.6
ament-xmllint==0.9.6
cycler==0.11.0
demo-nodes-py==0.9.3
domain-coordinator==0.9.0
examples-rclpy-executors==0.9.4
examples-rclpy-minimal-action-client==0.9.4
examples-rclpy-minimal-action-server==0.9.4
examples-rclpy-minimal-client==0.9.4
examples-rclpy-minimal-publisher==0.9.4
examples-rclpy-minimal-service==0.9.4
examples-rclpy-minimal-subscriber==0.9.4
examples-tf2-py==0.13.9
fonttools==4.31.2
kiwisolver==1.4.0
launch==0.10.4
launch-ros==0.11.1
launch-testing==0.10.4
launch-testing-ros==0.11.1
launch-xml==0.10.4
launch-yaml==0.10.4
matplotlib==3.5.1
numpy==1.22.3
opencv-python==4.5.5.64
osrf-pycommon==0.1.10
packaging==21.3
pandas==1.4.1
Pillow==9.0.1
pyparsing==3.0.7
python-dateutil==2.8.2
pytz==2022.1
quality-of-service-demo-py==0.9.3
ros2action==0.9.8
ros2bag==0.3.5
ros2cli==0.9.8
ros2component==0.9.8
ros2doctor==0.9.8
ros2interface==0.9.8
ros2launch==0.11.1
ros2lifecycle==0.9.8
ros2multicast==0.9.8
ros2node==0.9.8
ros2param==0.9.8
ros2pkg==0.9.8
ros2run==0.9.8
ros2service==0.9.8
ros2test==0.2.1
ros2topic==0.9.8
ros2trace==1.0.4
rosidl-runtime-py==0.9.0
rpyutils==0.2.0
rqt==1.0.6
rqt-action==0.4.9
rqt-graph==1.0.4
rqt-gui==1.0.6
rqt-gui-py==1.0.6
rqt-reconfigure==1.0.5
rqt-topic==0.4.9
scipy==1.8.0
seaborn==0.11.2
six==1.16.0
sros2==0.9.4
test-launch-ros==0.11.1
topic-monitor==0.9.3
tracetools-launch==1.0.4
tracetools-read==1.0.4
tracetools-trace==1.0.4



# driving-assistant

![demo video](./media/demo_output.gif)

## Setup
To set up the environment and install the required dependencies, follow these steps:

**Create Virtual Environment**
```bash
python3.11 -m venv env
source env/bin/activate
```

**Install dependencies**
```bash
pip install -e .
```

## Usage

```bash
driving-assistant --video_path <path_to_video_file.mp4> --weights_path <path_to_yolo_weights.pt>
```
**Arguments**:
- --video_path: (Required) The path to the input video file to be processed.
- --weights_path: (Optional) The path to the YOLO weights file to be used for object detection.


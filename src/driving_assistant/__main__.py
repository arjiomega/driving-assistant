import argparse

from driving_assistant.driving_assistant import VideoLoader, Inference, DrivingAssistant
from driving_assistant.boundary import Boundary


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--video_path", type=str, required=True, help="path to video.")
    parser.add_argument("--weights_path", type=str, help="path to yolo weights")

    args = parser.parse_args()

    video_loader = VideoLoader.from_path(args.video_path)

    weights_path = args.weights_path if args.weights_path else "./models/best.pt"

    inference = Inference.from_path(weights_path)
    boundary = Boundary.from_frame_shape(height=video_loader.height, width=video_loader.width)

    driving_assistant = DrivingAssistant(inference, boundary)

    video_loader.run(driving_assistant, "output")

if __name__ == "__main__":
    main()
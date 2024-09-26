#!/usr/bin/env python
import os.path
import os
import cv2
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner
from tracker import Tracker  # tracker.py library
from math import dist

runner = None
show_camera = True

if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

# Tracker and parameters
tracker = Tracker()
cy1 = 380  # Adjusted line for vehicles moving out
cy2 = 500  # Adjusted line for vehicles moving in
offset = 20  # Adjusted margin for line crossing
vh_down = {}  # Dictionary to store time for vehicles moving down
vh_up = {}  # Dictionary to store time for vehicles moving up
counter_in = []  # Vehicles moving in
counter_out = []  # Vehicles moving out
vehicle_speeds = {}  # Dictionary to store vehicle speeds
speed_display_timestamps = {}  # Time to show speed display

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        print('Error: Invalid arguments')
        sys.exit(2)

    if len(args) == 0:
        print('Usage: python traffic.py <path_to_model.eim> <path_to_video_file>')
        sys.exit(2)

    model = args[0]
    video_path = args[1]  # Path to video file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']

            # Open the video file instead of camera
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                raise Exception("Couldn't open video file: %s" % video_path)

            print(f"Video file {video_path} opened successfully.")
            next_frame = 0

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    print("End of video file reached or unable to read the frame.")
                    break

                if next_frame > now():
                    time.sleep((next_frame - now()) / 1000)

                # Resize the frame to 640x640 before processing
                resized_frame = cv2.resize(frame, (640, 640))

                # Send the resized frame to the Edge Impulse model for classification
                features, img = runner.get_features_from_image(resized_frame)
                res = runner.classify(features)

                if "bounding_boxes" in res["result"].keys():
                    detected_objects = []
                    for bb in res["result"]["bounding_boxes"]:
                        label = bb['label']
                        if label in ['car', 'truck']:
                            x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
                            cx = int(x + w / 2)
                            cy = int(y + h / 2)
                            detected_objects.append([x, y, x + w, y + h])

                            # Color bounding boxes
                            color = (50, 255, 0) if label == 'car' else (0, 50, 200)  # Green for cars, Orange for trucks
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    bbox_id = tracker.update(detected_objects)
                    for bbox in bbox_id:
                        x3, y3, x4, y4, obj_id = bbox
                        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

                        # Track vehicles moving down (in)
                        if cy2 - offset < cy < cy2 + offset:
                            vh_down[obj_id] = time.time()
                        if obj_id in vh_down and cy1 - offset < cy < cy1 + offset and obj_id not in counter_in:
                            elapsed_time = time.time() - vh_down[obj_id]
                            counter_in.append(obj_id)
                            distance = 70  # Meters
                            speed_ms = distance / elapsed_time
                            speed_kh = speed_ms * 3.6
                            vehicle_speeds[obj_id] = speed_kh
                            speed_display_timestamps[obj_id] = time.time()
                            print(f"Vehicle {obj_id} exited, speed: {speed_kh} kph")  # Debugging

                        # Track vehicles moving up (out)
                        if cy1 - offset < cy < cy1 + offset:
                            vh_up[obj_id] = time.time()
                        if obj_id in vh_up and cy2 - offset < cy < cy2 + offset and obj_id not in counter_out:
                            elapsed_time = time.time() - vh_up[obj_id]
                            counter_out.append(obj_id)
                            distance = 70  # Meters
                            speed_ms = distance / elapsed_time
                            speed_kh = speed_ms * 3.6
                            vehicle_speeds[obj_id] = speed_kh
                            speed_display_timestamps[obj_id] = time.time()
                            print(f"Vehicle {obj_id} entered, speed: {speed_kh} kph")  # Debugging

                        # Display speed for 10 seconds
                        if obj_id in vehicle_speeds and time.time() - speed_display_timestamps[obj_id] <= 10:
                            speed_text = f"ID{obj_id}: {int(vehicle_speeds[obj_id])}kph"
                            cv2.putText(img, speed_text, (x3, y3 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            #print(f"Displaying speed for Vehicle: {speed_text}")  # Debugging

                # Display in/out count
                in_count_text = f"OUT: {len(counter_in)}"
                out_count_text = f"IN: {len(counter_out)}"
                cv2.putText(img, in_count_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, out_count_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the in/out lines
                cv2.line(img, (0, cy1), (img.shape[1], cy1), (255, 0, 0), 1)
                cv2.line(img, (0, cy2), (img.shape[1], cy2), (0, 0, 255), 1)

                # Show video feed
                if show_camera:
                    cv2.imshow('edgeimpulse', img)
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100

            video_capture.release()
            cv2.destroyAllWindows()

        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])

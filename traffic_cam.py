#!/usr/bin/env python
import os.path
import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner
from tracker import Tracker  # tracker.py library (put this file in the same directory)
from math import dist

runner = None
show_camera = True

if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams(): # Find connected usb camera
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

# Setup the tracker and parameters for in/out detection and speed tracking
tracker = Tracker()
cy1 = 370  # Line for vehicle moving up (going out)
cy2 = 470  # Line for vehicle moving down (coming in)
offset = 30
vh_down = {}
vh_up = {}
counter_in = []
counter_out = []
vehicle_speeds = {}
speed_display_timestamps = {}

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        print('Error: Invalid arguments')
        sys.exit(2)

    if len(args) == 0:
        print('Usage: python classify.py <path_to_model.eim> <Camera port ID>')
        sys.exit(2)

    model = args[0]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']

            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args)<= 1 and len(port_ids)> 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")
                
            next_frame = 0

            for res, img in runner.classifier(videoCaptureDeviceId):
                if next_frame > now():
                    time.sleep((next_frame - now()) / 1000)

                if "bounding_boxes" in res["result"].keys():
                    detected_objects = []
                    for bb in res["result"]["bounding_boxes"]:
                        label = bb['label']
                        if label in ['car', 'truck']:  # Only track cars and trucks
                            x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
                            cx = int(x + w / 2)
                            cy = int(y + h / 2)
                            detected_objects.append([x, y, x + w, y + h])

                            # Draw the bounding box and label
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    bbox_id = tracker.update(detected_objects)
                    for bbox in bbox_id:
                        x3, y3, x4, y4, obj_id = bbox
                        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

                        # Track vehicles moving down (coming in)
                        if cy2 - offset < cy < cy2 + offset:
                            vh_down[obj_id] = time.time()
                        if obj_id in vh_down:
                            if cy1 - offset < cy < cy1 + offset and obj_id not in counter_in:
                                elapsed_time = time.time() - vh_down[obj_id]
                                counter_in.append(obj_id)
                                distance = 25  # Meters
                                speed_ms = distance / elapsed_time
                                speed_kh = speed_ms * 3.6
                                vehicle_speeds[obj_id] = speed_kh
                                speed_display_timestamps[obj_id] = time.time()
                                print(f"Vehicle {obj_id} exited, speed: {speed_kh} kph")  # Print vehicle speed 
                        # Track vehicles moving up (going out)
                        if cy1 - offset < cy < cy1 + offset:
                            vh_up[obj_id] = time.time()
                        if obj_id in vh_up:
                            if cy2 - offset < cy < cy2 + offset and obj_id not in counter_out:
                                elapsed_time = time.time() - vh_up[obj_id]
                                counter_out.append(obj_id)
                                distance = 25  # Meters
                                speed_ms = distance / elapsed_time
                                speed_kh = speed_ms * 3.6
                                vehicle_speeds[obj_id] = speed_kh
                                speed_display_timestamps[obj_id] = time.time()
                                print(f"Vehicle {obj_id} entered, speed: {speed_kh} kph")  # Print vehicle speed 
                        # Display speed for 5 seconds
                        if obj_id in vehicle_speeds and time.time() - speed_display_timestamps[obj_id] <= 5:
                            speed_text = f"ID{obj_id}: {int(vehicle_speeds[obj_id])}kph"
                            cv2.putText(img, speed_text, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Display in/out count
                in_count_text = f"OUT: {len(counter_in)}"
                out_count_text = f"IN: {len(counter_out)}"
                cv2.putText(img, in_count_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, out_count_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the in/out lines on the frame
                cv2.line(img, (0, cy1), (img.shape[1], cy1), (255, 0, 0), 2)  # Line for moving up
                cv2.line(img, (0, cy2), (img.shape[1], cy2), (0, 0, 255), 2)  # Line for moving down

                # Show camera feed
                if show_camera:
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])

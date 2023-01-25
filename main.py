from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2

(major, minor) = cv2.__version__.split(".")[:2]

if int(major) == 3 and int(minor) < 3:
    tracker  = cv2.Tracker_create("kcf".upper())
else:
    tracker = cv2.TrackerKCF_create()

initBB = None
print("Starting video stream")

vs = cv2.VideoCapture("2.mp4")
time.sleep(1.0)

fps = None

while True:
    
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fps.update()
        fps.stop()

        info = [
            {"Tracker", "kcf"},
            {"Successs", "Yes" if success else "No"},
            {"FPS", "{:.2f}".format(fps.fps())}
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, frameCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
        fps = fps.start()
    elif key == ord("q"):
        break

vs.stop()

cv2.destroyAllWindows()
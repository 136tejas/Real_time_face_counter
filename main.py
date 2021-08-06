import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    check, frame = video.read()
    flip = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(frameRGB)
    print(results)

    ID = 0

    if results.detections:
        for id, detection in enumerate(results.detections):

            # __________Draw rectangle by mediapipe function ___________

            mpDraw.draw_detection(flip, detection)

            #print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # __________Draw Rectangle by cordinates ______________

            # bboxC = detection.location_data.relative_bounding_box
            # fh, fw, fc = frame.shape
            # bbox = int(bboxC.xmin * fw), int(bboxC.ymin * fh),\
            #     int(bboxC.xmin * fw), int(bboxC.ymin * fh)
            # #cv2.rectangle(frame, bbox, (255, 0, 255), 2)

            ID = id + 1

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(flip, f'Fps :{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.putText(flip, f'Faces : {int(ID)}', (250, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow("Face_detection", flip)

    if cv2.waitKey(1) == ord('q'):
        break

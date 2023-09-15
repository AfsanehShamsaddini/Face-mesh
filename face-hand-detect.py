import cv2
import  mediapipe as mp

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True,  min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=0, color=(255,0,0))
mp_drawing_styles = mp.solutions.drawing_styles

mpHand = mp.solutions.hands
handMesh = mpHand.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)
while True:
    _, img = cap.read()
    # Convert the BGR image to RGB before processing.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    # Draw the face mesh annotations on the image.
    if result.multi_face_landmarks:
       for faceLand in result.multi_face_landmarks:
           mpDraw.draw_landmarks(img, faceLand, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
           mpDraw.draw_landmarks(img, faceLand, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, mp_drawing_styles
                        .get_default_face_mesh_contours_style())

    results = handMesh.process(imgRGB)
    # Initially set finger count to 0 for each cap
    fingerCount = 0
    namehand = ' '
    if results.multi_hand_landmarks:
        for handLand in results.multi_hand_landmarks:
            handIndex = results.multi_hand_landmarks.index(handLand)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            handLandmarks = []
            for lands in handLand.landmark:
                handLandmarks.append([lands.x, lands.y])
            # Thumb: TIP x position must be greater or lower than IP x position,
            #   deppeding on hand label.
            if handLabel == 'Left' and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount += 1
                namehand = 'Right Hand'
                if len(results.multi_handedness) == 2:
                    namehand = ''
                    cv2.putText(img, 'Both Hands', (20, 450),
                                cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)
            elif handLabel == 'Right' and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount += 1
                namehand = 'Left Hand'
                if len(results.multi_handedness) == 2:
                    namehand = ''
                    cv2.putText(img, 'Both Hands', (20, 450),
                                cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)
            # Other fingers: TIP y position must be lower than PIP y position
            if handLandmarks[8][1] < handLandmarks[6][1]:          #Index finger
                fingerCount += 1
            if handLandmarks[12][1] < handLandmarks[10][1]:       #Middle finger
                fingerCount += 1
            if handLandmarks[16][1] < handLandmarks[14][1]:         #Ring finger
                fingerCount += 1
            if handLandmarks[20][1] < handLandmarks[18][1]:        #Pinky
                fingerCount += 1

            mpDraw.draw_landmarks(img, handLand, mpHand.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
            cv2.putText(img, namehand, (20, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (0, 255, 0), 2)

    cv2.putText(img, str(fingerCount), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)


    cv2.imshow('Face Mesh', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
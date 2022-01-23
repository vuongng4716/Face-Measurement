import cv2
import time
import FaceMeshModule

cap = cv2.VideoCapture(0)
detector = FaceMeshModule.FaceMeshDetector(maxFaces=1)

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)

    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    return img, [x1, y2, x2, y1]

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) / w

        putTextRect(img, f'Depth: {int(d)}cm',
                    (face[10][0] - 100, face[10][1] - 50),
                    scale=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

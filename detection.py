import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO('Models/best.pt') 


cap = cv2.VideoCapture(0)  

plt.ion()
fig, ax = plt.subplots()

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            confidence = bbox.conf[0]
            class_id = int(bbox.cls[0])
            label = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(frame_rgb)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

    if plt.waitforbuttonpress(timeout=0.001) and plt.get_current_fig_manager().canvas.keypress == 'q':
        break

plt.ioff()  
cap.release()
cv2.destroyAllWindows()

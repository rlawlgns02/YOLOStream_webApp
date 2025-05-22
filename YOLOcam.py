import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
# cap = cv2.VideoCapture(0)
url = 'rtsp://admin:Admin001@gold33.iptime.org:557/2'  # 선택한 URL
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error: 웹캠 연결에 실패했습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 웹캠에서 영상을 읽어올 수 없습니다.")
        break

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get detections

    # Draw bounding boxes and labels
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv5 객체 감지", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

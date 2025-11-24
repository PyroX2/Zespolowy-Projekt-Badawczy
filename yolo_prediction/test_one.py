from ultralytics import YOLO
import cv2
import numpy as np


def detect_with_boxes(
    image_path: str,
    weights_path: str,
    show: bool = False,
    save: bool = False,
    save_path: str | None = None,
    imgsz: int = 1024,
    conf: float = 0.15,
    iou: float = 0.7,
):


    #Zaladuj model
    model = YOLO(weights_path)

    #Wczytaj obraz 
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Nie mogę wczytać obrazu: {image_path}")

    #Detekcja
    result = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    boxes_out = []

    #konwersja bo csvopean
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            boxes_out.append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(score), "cls": int(cls)}
            )

            #zielony prostoka
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #show save
    if save:
        if save_path is None:
            save_path = image_path.rsplit(".", 1)[0] + "_boxes.jpg"
        cv2.imwrite(save_path, img_bgr)

    if show:
        cv2.imshow("detekcja", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes_out


if __name__ == "__main__":
    boxes = detect_with_boxes(
        image_path=r"C:\Users\Acer\zdj.png",
        weights_path=r"C:\Users\Acer\best.pt",
        show=True,
        save=True,
        save_path=r"C:\Users\Acer\zdj_out.png",
    )
    print("Boksy:", boxes)

# Parse ROI coordinates from string to list of bounding boxes
def parse_roi(roi_coords: str):
    roi_coords = roi_coords.translate({ord(c): None for c in "][)(,"})
    flat = list(map(float, roi_coords.split()))
    return [flat[i:i+4] for i in range(0, len(flat), 4)]

def roi_rescale(roi: list, image_height: int, image_width) -> tuple:
    ymin, xmin, ymax, xmax = roi

    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height
    width = abs(xmax - xmin) / image_width
    height = abs(ymax - ymin) / image_height

    return x_center, y_center, width, height
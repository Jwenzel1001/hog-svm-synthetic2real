import os
import cv2
import numpy as np
from skimage.feature import hog
from imutils.object_detection import non_max_suppression
from sklearn.pipeline import make_pipeline
import joblib

# Model paths Linear SVM models
#models = {
#    "Original_Synthetic": "hog_svm_original_synthetic_model.pkl",
#    "Original_Real": "hog_svm_original_real_model.pkl",
#    "Original_Mixed": "hog_svm_original_mixed_model.pkl"
#}

# Model paths kernel SVM models
models = {
    "Original_Synthetic": "hog_svm_kernel_synthetic.pkl",
    "Original_Real": "hog_svm_kernel_real.pkl",
    "Original_Mixed": "hog_svm_kernel_mixed.pkl"
}

input_dir = "../HOGSVM_Test/images"
label_dir = "../HOGSVM_Test/labels"
output_dir = "../HOGSVM_Test/Full_Outputs_Final"
os.makedirs(output_dir, exist_ok=True)

hog_params = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys'
}

window_size = (128, 128)
scale_factor = 1.25
iou_threshold = 0.5
step_sizes = [8, 16, 32] # Smaller step size take longer to run
score_thresholds = [0.5, 1.0, 1.5, 2.0]

# Max image input
MAX_WIDTH = 640
MAX_HEIGHT = 480

# Extract multiscale hog features for models trained using multiscale hog feature extraction
def multiscale_hog(image, cell_sizes=[(18, 18), (6, 6), (3, 3)]):  # use training config
    features = []
    for cell in cell_sizes:
        h = hog(image,
                orientations=9,
                pixels_per_cell=cell,
                cells_per_block=(1, 1),
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True)
        features.append(h)
    return np.concatenate(features)

# Creates image pyramid for different size wolf detection
def pyramid(image, scale=1.25, min_size=(128, 128)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

# Apply sliding window over input image returning the window coordinates on the image
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Convert labels from Roboflow datasets from YOLO ground truth annotation to box coordinates (XC, YC, W, H) to (X1, Y1, X2, Y2)
def yolo_to_bbox(yolo_line, img_w, img_h):
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    x_center, y_center, width, height = map(float, parts[1:5])
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    return [x1, y1, x2, y2]

# Computes intersection over union
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Evaluate each model
for model_name, model_path in models.items():
    print(f"Evaluating {model_name}...")
    model = joblib.load(model_path)
    
    result_file = os.path.join(output_dir, f"{model_name}_results.txt")
    with open(result_file, 'w') as f_out:
        
        # Iterate over different step sizes
        for step_size in step_sizes:
            for score_threshold in score_thresholds:
                
                total_TP, total_FP, total_FN = 0, 0, 0
                image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                for img_name in image_files:
                    img_path = os.path.join(input_dir, img_name)
                    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    # Resize images to ensure they are 640x480
                    h, w = image.shape[:2]
                    if w > MAX_WIDTH or h > MAX_HEIGHT:
                        scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
                        image = cv2.resize(image, (int(w * scale), int(h * scale)))

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    detections = []

                    # Perform detections on the image pyramid with the sliding window
                    for resized in pyramid(gray, scale=scale_factor, min_size=window_size):
                        scale = gray.shape[1] / float(resized.shape[1])
                        for (x, y, window) in sliding_window(resized, step_size, window_size):
                            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                                continue
                            #features = hog(window, **hog_params) # For linear SVM trained models, does not use multiscale_hog
                            features = multiscale_hog(window) # For kernel SVM trained models, trained on multiscale_hog descriptors
                            score = model.decision_function([features])[0]
                            if score > score_threshold:
                                x1 = int(x * scale)
                                y1 = int(y * scale)
                                x2 = int((x + window_size[0]) * scale)
                                y2 = int((y + window_size[1]) * scale)
                                detections.append([x1, y1, x2, y2, score])

                    # Apply non max suppresion for overlapping or redundant predictions
                    pred_boxes = non_max_suppression(np.array([d[:4] for d in detections]), probs=[d[4] for d in detections])

                    # Read in ground truth bounding boxes
                    gt_boxes = []
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            for line in f:
                                box = yolo_to_bbox(line, image.shape[1], image.shape[0])
                                if box:
                                    gt_boxes.append(box)

                    # Match predictions with ground truth boxes
                    matched = set()
                    TP, FP = 0, 0
                    for pred in pred_boxes:
                        match = False
                        for i, gt in enumerate(gt_boxes):
                            if i in matched:
                                continue
                            if compute_iou(pred, gt) >= iou_threshold:
                                TP += 1
                                matched.add(i)
                                match = True
                                break
                        if not match:
                            FP += 1
                    FN = len(gt_boxes) - len(matched)
                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                # Compute evaluation metrics
                precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
                recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

                # Log results
                log = f"[Step={step_size} | Thresh={score_threshold}] Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, TP={total_TP}, FP={total_FP}, FN={total_FN}\n"
                print(log.strip())
                f_out.write(log)

print("\n ALL DONE!")

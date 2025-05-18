# Synthetic-to-Real Object Detection Using HOG+SVM

This project investigates the domain gap between synthetic and real-world data in object detection. It applies traditional machine learning techniques—Histogram of Oriented Gradients (HOG) and Support Vector Machines (SVM)—to detect wolves in images collected from both simulated and real environments.

Developed as part of EE 5290 at Iowa State University.

---

## Objectives

- Explore the limitations of classical object detectors under domain shift
- Use Microsoft AirSim to generate synthetic image datasets
- Apply HOG+SVM in both linear and kernelized forms (chi-squared kernel)
- Evaluate performance on real-world test data using standard metrics:
  - Precision
  - Recall
  - F1 Score

---

## System Overview

| Module               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Data Generator**    | Uses Microsoft AirSim to simulate labeled images of wolves in varying conditions |
| **Feature Extractor** | Implements HOG using scikit-image with Dalal & Triggs configuration         |
| **Classifier**        | SVM models trained with linear and kernel approaches using scikit-learn     |
| **Evaluator**         | Precision, recall, F1 computed against ground truth annotations             |
| **Detection Pipeline**| Sliding window with multi-scale image pyramid and non-max suppression       |

---

## Dataset Structure

- **Synthetic Dataset**: Generated in AirSim with weather and perspective variation
- **Real Dataset**: Collected from Roboflow
- **Combined Dataset**: 80% real, 20% synthetic for balanced training
- **Test Set**: Withheld real-world images containing annotated wolf instances

---

## Algorithm Summary

| Model Type     | Training Source   | Feature Type         | Kernel Method        |
|----------------|-------------------|----------------------|----------------------|
| Linear SVM     | Synthetic / Real / Mixed | Single-scale HOG | None (Linear)       |
| Kernel SVM     | Synthetic / Real / Mixed | Multi-scale HOG  | Chi-squared (AdditiveChi2) |

---

## Tools & Libraries

- Python 3.9
- `scikit-image`, `scikit-learn`
- `OpenCV`, `joblib`, `tqdm`
- `imutils` for non-max suppression
- Microsoft AirSim (simulation)

---

## Project Limitations

- Classical models struggle with complex shapes and perspectives (e.g., wolves)
- Detection used a fixed step size (32px) for performance constraints
- Models suffer from generalization issues under domain shift
- Evaluation limited to a small real-world test set (68 images)

---

## License

MIT License — see `LICENSE` file.

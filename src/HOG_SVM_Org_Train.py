import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from tqdm import tqdm
import joblib

# Set fixed seed for reproducibility
random.seed(42)

# Dataset directories
pos_dir = r"../HOGSVM_Mixed_Dataset/pos"
neg_dir = r"../HOGSVM_Mixed_Dataset/neg"

# Target number of samples
n_samples = 1824
image_size = (128, 128)
cell_sizes = [(18, 18), (6, 6), (3, 3)]
orientations = 9
block_size = (1, 1)

X, y = [], []

def extract_maji_style_hog(img, levels):
    features = []
    for cell in levels:
        h = hog(img,
                orientations=orientations,
                pixels_per_cell=cell,
                cells_per_block=block_size,
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True)
        features.append(h)
    return np.concatenate(features)

def load_fixed_samples(folder, label, limit, shuffle=True):
    files = os.listdir(folder)
    if shuffle:
        files = random.sample(files, len(files))
    files = files[:limit]
    for file in tqdm(files, desc=f"Label {label}"):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        feat = extract_maji_style_hog(img, cell_sizes)
        X.append(feat)
        y.append(label)

# Load a fixed number of samples from each class
load_fixed_samples(pos_dir, label=1, limit=n_samples)
load_fixed_samples(neg_dir, label=0, limit=n_samples)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Build and train model
clf = make_pipeline(
    AdditiveChi2Sampler(sample_steps=2),
    StandardScaler(),
    LinearSVC(C=0.01, max_iter=10000)
)

print("Training balanced model with Maji et al. settings...")
clf.fit(X, y)
joblib.dump(clf, "hog_svm_kernel_mixed_.pkl")
print("Model saved as hog_svm_maji_balanced.pkl")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import cv2
from keras.models import model_from_json
import numpy as np
import os
import json
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_data_gen = ImageDataGenerator(rescale=1./255)

zip_path = "/content/archive (1).zip"
fixed_zip = "/content/dataset_project.zip"
extract_dir = "/content/dataset"
train_dir = "/content/train"
test_dir = "/content/test"
if os.path.exists(extract_dir):
    print(f"Cleaning up existing extraction directory: {extract_dir}")
    shutil.rmtree(extract_dir)
if os.path.exists(zip_path):
    if os.path.exists(fixed_zip):
        os.remove(fixed_zip)
    shutil.copy(zip_path, fixed_zip)
    print(f"✅ Copied '{zip_path}' to '{fixed_zip}' for consistent processing.")
else:
    if not os.path.exists(fixed_zip):
        raise ValueError(f"❌ Neither '{zip_path}' nor '{fixed_zip}' found. Please upload a dataset zip file.")
    print(f"⚠️ '{zip_path}' not found. Proceeding with '{fixed_zip}'.")
if not os.path.exists(extract_dir):
    if zipfile.is_zipfile(fixed_zip):
        with zipfile.ZipFile(fixed_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Extracted successfully to:", extract_dir)
    else:
        raise ValueError("❌ Not a valid ZIP file! Please re-upload a proper zip.")
else:
    print("⚠️ Extraction folder already exists. Skipping extraction.")

all_images = []
all_labels = []

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.lower().endswith(tuple([".jpg", ".jpeg", ".png"])):
            all_images.append(os.path.join(root, file))

            label = os.path.basename(root)
            if label == os.path.basename(extract_dir):
                print(f"⚠️ Image '{file}' found directly in extraction root. Skipping or assign a default label if intended.")

                continue
            all_labels.append(label)

if len(all_images) == 0:
    raise ValueError("❌ No images found! Please check your zip structure.")

print(f"📸 Total images found: {len(all_images)}")
print("✅ Unique emotion classes:", set(all_labels))
print("🗂 Image count per class:", Counter(all_labels))

label_counts = Counter(all_labels)

if not label_counts:
    raise ValueError("No labels found after processing images.")

min_class_count = min(label_counts.values())


stratify_labels = all_labels if min_class_count > 1 else None
if stratify_labels is None:
    print("⚠️ Some classes have only 1 image or fewer. Splitting without stratification may lead to uneven distribution.")

try:
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=stratify_labels
    )
except ValueError as e:
    print(f"❌ Error during train_test_split: {e}")
    print("This often happens if a class has only one sample and stratify is enabled.")
    print("Attempting split without stratification.")
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )

print(f"🟢 Training images: {len(train_images)}, Testing images: {len(test_images)}")

def save_images(image_list, label_list, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for img_path, label in zip(image_list, label_list):
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(class_dir, os.path.basename(img_path)))
        else:
            print("⚠️ Missing image, skipped:", img_path)

save_images(train_images, train_labels, train_dir)
save_images(test_images, test_labels, test_dir)

print("✅ Train/Test split completed.")
print("📂 Training folder:", train_dir)
print("📂 Testing folder:", test_dir)

train_generator = train_data_gen.flow_from_directory(
    '/content/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        '/content/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


emotion_model = Sequential()


emotion_model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                         input_shape=(48, 48, 1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))


emotion_model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))


emotion_model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))


emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.5))

emotion_model.add(Dense(train_generator.num_classes, activation='softmax'))

emotion_model.summary()

cv2.ocl.setUseOpenCL(False)


optimizer = Adam(learning_rate=0.0001)


loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

emotion_model.compile(loss=loss_fn, optimizer=optimizer,
                      metrics=['accuracy'])


class_labels = list(train_generator.class_indices.values())
class_weights_arr = compute_class_weight(
    'balanced', classes=np.unique(class_labels),
    y=train_generator.classes
)
class_weight_dict = dict(zip(np.unique(class_labels), class_weights_arr))
print("⚖️ Class weights:", class_weight_dict)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_emotion_model.weights.h5', monitor='val_accuracy',
                    save_best_only=True, save_weights_only=True, verbose=1),
]

#
steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks)

# ── Evaluation ──
eval_results = emotion_model.evaluate(validation_generator)
print(f"\n📊 Final Validation Loss: {eval_results[0]:.4f}")
print(f"📊 Final Validation Accuracy: {eval_results[1]*100:.2f}%")

# ── Training History ──
accuracy = emotion_model_info.history['accuracy']
val_accuracy = emotion_model_info.history['val_accuracy']
loss = emotion_model_info.history['loss']
val_loss = emotion_model_info.history['val_loss']

# Save training history as JSON (for dashboard)
history_data = {
    'accuracy': [float(x) for x in accuracy],
    'val_accuracy': [float(x) for x in val_accuracy],
    'loss': [float(x) for x in loss],
    'val_loss': [float(x) for x in val_loss],
}
with open('training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)
print("✅ Training history saved to training_history.json")

# ── Accuracy & Loss Plots ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(accuracy, label='Train Accuracy', linewidth=2)
ax1.plot(val_accuracy, label='Val Accuracy', linewidth=2, linestyle='--')
ax1.set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(loss, label='Train Loss', linewidth=2)
ax2.plot(val_loss, label='Val Loss', linewidth=2, linestyle='--')
ax2.set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training curves saved to training_curves.png")


validation_generator.reset()
y_pred = emotion_model.predict(validation_generator,
                                steps=validation_steps, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes[:len(y_pred_classes)]
class_names = list(validation_generator.class_indices.keys())

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved to confusion_matrix.png")

# ── Save Model ──
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)
emotion_model.save_weights('emotion_model.weights.h5')
print("✅ Model saved: emotion_model.json + emotion_model.weights.h5")

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised",
                7: "Class_7", 8: "Class_8", 9: "Class_9", 10: "Class_10",
                11: "Class_11", 12: "Class_12", 13: "Class_13", 14: "Class_14",
                15: "Class_15", 16: "Class_16", 17: "Class_17", 18: "Class_18"}


with open('emotion_dict.json', 'w') as f:
    json.dump(emotion_dict, f, indent=2)


with open('emotion_model.json', 'r') as f:
    loaded_model_json = f.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('emotion_model.weights.h5')  # BUG FIX: was missing!
print("✅ Model reloaded for inference test")
frame = cv2.imread('/content/Screenshot 2025-09-02 224553.png')

if frame is None:
    print("❌ Error: Could not load image. Check the file path.")
else:

    frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_frame, cmap='gray')
    plt.axis('off')
    plt.title("Grayscale Debug View")
    plt.show()

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    num_faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,   
        minSize=(30, 30)
    )

    print(f"Faces detected: {len(num_faces)}")
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_text = emotion_dict[maxindex]

        cv2.putText(frame, emotion_text, (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Emotion Detection Result")
    plt.show()


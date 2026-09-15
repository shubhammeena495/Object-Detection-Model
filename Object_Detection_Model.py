import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

# --- 1. Visualization Utilities ---
def draw_bounding_boxes_on_image_array(image, boxes, color=None, thickness=1, display_str_list=()):
    image_pil = PIL.Image.fromarray(np.uint8(image * 255)).convert("RGB")
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness, display_str_list)
    return np.array(image_pil)

def draw_bounding_boxes_on_image(image, boxes, color=None, thickness=1, display_str_list=()):
    if boxes is None or len(boxes) == 0:
        return
    for i in range(len(boxes)):
        box = boxes[i]
        draw_bounding_box_on_image(image, box[1], box[0], box[3], box[2], color=color[i] if color else 'red', thickness=thickness)

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=1):
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    # Get a batch for visualization
    for training_digits, (training_labels, training_bboxes) in training_dataset.take(1):
        training_digits = training_digits.numpy()
        training_labels = training_labels.numpy()
        training_bboxes = training_bboxes.numpy()
        
    for validation_digits, (validation_labels, validation_bboxes) in validation_dataset.take(1):
        validation_digits = validation_digits.numpy()
        validation_labels = validation_labels.numpy()
        validation_bboxes = validation_bboxes.numpy()
        
    training_labels = np.argmax(training_labels, axis=1)
    validation_labels = np.argmax(validation_labels, axis=1)
    return (training_digits, training_labels, training_bboxes, validation_digits, validation_labels, validation_bboxes)

def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 10
    indexes = np.random.choice(len(digits), size=n)
    fig = plt.figure(figsize=(20, 4))
    plt.suptitle(title)
    
    for i in range(n):
        idx = indexes[i]
        ax = fig.add_subplot(1, n, i + 1)
        
        # Combine true and predicted boxes for plotting
        boxes_to_plot = []
        if len(pred_bboxes) > idx: boxes_to_plot.append(pred_bboxes[idx])
        if len(bboxes) > idx: boxes_to_plot.append(bboxes[idx])
            
        img_to_draw = draw_bounding_boxes_on_image_array(digits[idx], np.array(boxes_to_plot), color=['red', 'green'])
        ax.imshow(img_to_draw)
        ax.set_xlabel(f"Pred: {predictions[idx]}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_metrics(history, metric_name, title):
    plt.figure()
    plt.plot(history.history[metric_name], label=metric_name)
    plt.plot(history.history['val_' + metric_name], label='val_' + metric_name)
    plt.title(title)
    plt.legend()
    plt.show()

# --- 2. Data Loading & Preprocessing ---
strategy = tf.distribute.get_strategy()
BATCH_SIZE = 64 * strategy.num_replicas_in_sync

def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (28, 28, 1))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0
    
    xmin_norm = tf.cast(xmin, tf.float32) / 75.0
    ymin_norm = tf.cast(ymin, tf.float32) / 75.0
    xmax_norm = (tf.cast(xmin, tf.float32) + 28) / 75.0
    ymax_norm = (tf.cast(ymin, tf.float32) + 28) / 75.0
    
    return tf.squeeze(image), (tf.one_hot(label, 10), [xmin_norm, ymin_norm, xmax_norm, ymax_norm])

training_dataset = tfds.load("mnist", split="train", as_supervised=True).map(read_image_tfds).shuffle(5000).batch(BATCH_SIZE).repeat()
validation_dataset = tfds.load("mnist", split="test", as_supervised=True).map(read_image_tfds).batch(10000)

(train_d, train_l, train_b, val_d, val_l, val_b) = dataset_to_numpy_util(training_dataset, validation_dataset, 1000)

# --- 3. Model Definition ---
def build_model():
    inputs = tf.keras.Input(shape=(75, 75, 1))
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    class_out = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    bbox_out = tf.keras.layers.Dense(4, name="bounding_box")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[class_out, bbox_out])
    model.compile(optimizer='adam', 
                  loss={'classification': 'categorical_crossentropy', 'bounding_box': 'mse'},
                  metrics={'classification': 'accuracy', 'bounding_box': 'mse'})
    return model

model = build_model()

# --- 4. Training ---
history = model.fit(training_dataset, steps_per_epoch=200, validation_data=validation_dataset, epochs=5)

# --- 5. Prediction & Visualization ---
preds = model.predict(val_d)
predicted_labels = np.argmax(preds[0], axis=1)
display_digits_with_boxes(val_d, predicted_labels, val_l, preds[1], val_b, [], "Test Results")
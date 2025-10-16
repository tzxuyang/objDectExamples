import transformers
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_bbox(image_path, boxes, labels, scores, threshold=0.5):
    """
    Draws bounding boxes on an image with associated labels and scores.

    Args:
        image_path (str): Path to the input image file.
        boxes (list or np.ndarray): List of bounding boxes, each defined as [x_min, y_min, x_max, y_max].
        labels (list or np.ndarray): List of label indices corresponding to each bounding box.
        scores (list or np.ndarray): List of confidence scores for each bounding box.
        threshold (float, optional): Minimum score required to display a bounding box. Defaults to 0.5.

    Displays:
        The image with bounding boxes, labels, and scores overlaid.
    """

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min, f"{model.config.id2label[label]}: {score:.2f}", color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model_name = "hustvl/yolos-tiny"
    image_path = "/home/yang/MyRepos/object_detection/images/dog2.jpg"  # Replace with your image path

    # Load model and processor
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Print results
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score >= 0.5:  # Confidence threshold
            box = [round(i, 2) for i in box.tolist()]
            print(f"Label: {model.config.id2label[label.item()]}, Score: {round(score.item(), 3)}, Box: {box}")
    
    draw_bbox(
        image_path, 
        results["boxes"].tolist(), 
        results["labels"].tolist(), 
        results["scores"].tolist(), 
        threshold=0.5
    )
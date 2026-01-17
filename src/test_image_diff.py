from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def read_image_rgb(image_path_1, image_path_2):
    image_1 = Image.open(image_path_1).convert("RGB")
    image_2 = Image.open(image_path_2).convert("RGB")
    diff_rgb = np.array(image_1) - np.array(image_2)
    image_diff = Image.fromarray(diff_rgb, 'RGB')
    image_diff.show()
    diff = np.abs(np.array(image_1) - np.array(image_2))
    # return diff.mean()
    return np.mean(diff)

def cal_image_diff(image_path_1, image_path_2):
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM between the two images
    # score is the SSIM metric, diff is the difference image
    (score, diff) = ssim(gray1, gray2, full=True)
    print(f"Image Similarity Score (SSIM): {score:.4f}")

    # The diff image is a float array, convert to 8-bit unsigned integer
    diff = (diff * 255).astype("uint8")

    return score

if __name__ == "__main__":
    img_path_1 = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4_test/1766396067.006229.jpg"
    img_path_2 = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4_test/1766396067.039565.jpg"
    img_path_1 = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4_test/1766396067.506269.jpg"
    img_path_2 = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4_test/1766396067.539605.jpg"
    diff_value = cal_image_diff(img_path_1, img_path_2)
    print(f"Mean RGB difference between images: {diff_value}")
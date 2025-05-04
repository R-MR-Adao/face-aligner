import os
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm
import cv2
import termcolor
from mtcnn import MTCNN
import math
import numpy as np
import sys
import shutil

# Initialize MediaPipe Face Detection
detector = MTCNN()

# Function to get the date from EXIF metadata
def get_exif_date(image_path):
    """
    Extracts the original date from the EXIF metadata of an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str or None: The original date in 'YYYY-MM-DD' format if available, otherwise None.
    """

    try:
        # Open the image and get the EXIF data
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data:
            for tag, value in exif_data.items():
                # Look for the DateTimeOriginal tag (0x9003)
                if TAGS.get(tag) == 'DateTimeOriginal':
                    return value.split(' ')[0].replace(':','-')

    except (AttributeError, KeyError, IndexError) as e:
        # If no EXIF data or DateTimeOriginal tag is found, return None
        termcolor.cprint(f"Runtime Exception: {e}", "red")
        return None

# Function to add watermark
def add_watermark(image_path, output_path, watermark_text, fontsize=200):
    """
    Adds a text watermark (typically a date) to the top-right corner of an image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the watermarked image.
        watermark_text (str): Text to use as the watermark.
        fontsize (int, optional): Font size of the watermark text. Default is 200.
    """

    # Open the image
    img = Image.open(image_path)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    # Get image dimensions
    img_width, img_height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Calculate text size
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position the text in the top-right corner
    position = (img_width - text_width - 20, 10)

    # Add the text as a watermark
    draw.text(position, watermark_text, font=font, fill=(0,0,0))

    # Save the new image
    img.save(output_path)

def crop_and_offset(y, h_, H):
    """
    Calculates top and bottom Y-coordinates for cropping with boundary checks and offset correction.

    Args:
        y (int): Y-coordinate of the crop center.
        h_ (float): Height of the crop box.
        H (int): Total image height.

    Returns:
        tuple: (y0, y1) bounds for cropping within the image height.
    """

    y0 = int(y - h_/2)

    if y0 < 0:
        off = abs(y0)
        y0 = 0
    
    else:
        off = 0

    y1 = int(y + h_/2 + off)

    if y1 > H:
        off = y1 - H
        y1 = H
    
    else:
        off = 0
    
    y0 -= off
    return y0, y1

def crop_face(image_path, output_path, margin=0.35, eye_scale=0.11):
    """
    Detects a face in an image, aligns it based on key landmarks, and saves the aligned image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the aligned image.
        margin (float, optional): Unused in this version; reserved for cropping margin.
        eye_scale (float, optional): Unused in this version; reserved for scaling based on eye distance.

    Returns:
        numpy.ndarray or None: Affine transformation matrix used to align the face,
                            or None if no face was detected.
    """

    def _raise(msg):
        termcolor.cprint(f" {msg}", "red")
        cv2.imwrite(output_path, img)
    
    # Read the image
    img = cv2.imread(image_path)
    H, W, *_ = img.shape

    # Convert the image to RGB (Mediapipe works with RGB images)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        return _raise(f"No face detected in the image {image_path}")
    
    # Assume the first face is the main one
    areas = [f["box"][2] * f["box"][3] for f in faces]
    face = faces[areas.index(max(areas))]
    
    left_eye = face['keypoints']['left_eye']
    right_eye = face['keypoints']['right_eye']
    nose = face['keypoints']['nose']
    mouth_left = face['keypoints']['mouth_left']
    mouth_right = face['keypoints']['mouth_right']

    # Calculate the rotation angle
    target_left_eye =    (W * (0.5 - 0.103) , H * (0.5 + 0.0547))
    target_right_eye=    (W * (0.5 + 0.103) , H * 0.5 + 0.0547)
    target_nose =        (W * 0.5           , H * (0.5 + 0.128))
    target_mouth_left =  (W * (0.5 - 0.0569), H * (0.5 + 0.264))
    target_mouth_right = (W * (0.5 + 0.0569), H * (0.5 + 0.264))

    left_eye = face['keypoints']['left_eye']
    right_eye = face['keypoints']['right_eye']
    nose = face['keypoints']['nose']
    mouth_left = face['keypoints']['mouth_left']
    mouth_right = face['keypoints']['mouth_right']

    # Calculate the distance between the eyes in both the original and target images
    original_eye_distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
    target_eye_distance = math.sqrt((target_right_eye[0] - target_left_eye[0])**2 + (target_right_eye[1] - target_left_eye[1])**2)

    # Scaling factor based on eye distance
    scale_factor = target_eye_distance / original_eye_distance

    # Calculate the angle of rotation using the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Calculate the center of the landmarks (eyes, nose, and mouth corners)
    center_x = (left_eye[0] + right_eye[0] + nose[0] + mouth_left[0] + mouth_right[0]) / 5
    center_y = (left_eye[1] + right_eye[1] + nose[1] + mouth_left[1] + mouth_right[1]) / 5

    # Calculate the center of the target landmarks
    target_center_x = (target_left_eye[0] + target_right_eye[0] + target_nose[0] + target_mouth_left[0] + target_mouth_right[0]) / 5
    target_center_y = (target_left_eye[1] + target_right_eye[1] + target_nose[1] + target_mouth_left[1] + target_mouth_right[1]) / 5

    # Calculate the translation needed to move the center of the landmarks to the target center
    translation_x = target_center_x - center_x
    translation_y = target_center_y - center_y

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale_factor)

    # Add translation to the rotation matrix
    M[0, 2] += translation_x
    M[1, 2] += translation_y

    # Apply the affine transformation to align the face
    aligned_img = cv2.warpAffine(img, M, (W, H), borderValue=(0, 255, 0))

    termcolor.cprint(f" Successfully processed {image_path}", "green")    
    cv2.imwrite(output_path, aligned_img)

    return M

def replace_background(image_path, output_path, rotation_matrix, scale_factor=1.5, blur_size=(501,501)):
    """
    Replaces the background of an image with a blurred and magnified version,
    based on a green-colored mask area created during face alignment.

    Args:
        image_path (str): Path to the aligned face image.
        output_path (str): Path to save the image with replaced background.
        rotation_matrix (numpy.ndarray): The affine matrix from face alignment.
        scale_factor (float, optional): Magnification factor for background. Default is 1.5.
        blur_size (tuple, optional): Kernel size for background blurring. Default is (501, 501).
    """

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Create a mask for non-black regions of the rotated image
    mask = (img[:,:,0] - 0 <= 40) & (img[:,:,1] >= 215) & (img[:,:,2] <= 40)

    if np.sum(mask) == 0:
        return

    # Magnify the rotated image
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    magnified_size = (int(new_w * scale_factor), int(new_h * scale_factor))
    magnified_image = cv2.resize(img, magnified_size, interpolation=cv2.INTER_LINEAR)

    # Blur the magnified image
    blurred_background = cv2.blur(magnified_image, blur_size)

    # Center crop the blurred image to match rotated image size
    hb, wb, *_ = blurred_background.shape
    x0, y0 = int((wb - w) / 2), int((hb - h) / 2)
    cropped_background = blurred_background[y0:y0 + h, x0: x0 + w]

    # Blend the rotated image with the background using the mask
    img_back = np.where(mask[..., None], cropped_background, img)
    cv2.imwrite(output_path, img_back)

def process_images(folder_path, apply_crop:bool=True, font_size:int=200):
    """
    Processes all images in a folder: applies face alignment, background replacement,
    and adds date watermarks.

    Args:
        folder_path (str): Path to the folder containing images.
        apply_crop (bool, optional): Whether to perform face alignment. Default is True.
        font_size (int, optional): Font size for watermark text. Default is 200.
    """

    # Ensure the output folder exists
    output_folder = os.path.join(folder_path, "processed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Processing images in {folder_path}")

    # Iterate over all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        
        # Only process images (filter out non-image files)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        image_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        # crop image around face
        if apply_crop:
            rotation_matrix = crop_face(image_path, output_path)

            # add blurred background
            replace_background(output_path, output_path, rotation_matrix)
        
        else:
            shutil.copyfile(image_path, output_path)
        
        # add date watermark
        # Get the date from EXIF metadata
        exif_date = get_exif_date(image_path)
        watermark_text = exif_date if exif_date else "Unknown Date"  # Fallback text if no date found
        add_watermark(output_path, output_path, watermark_text, font_size)

# Function to open a folder dialog to select folder path
def select_folder():
    """
    Opens a dialog to select a folder using Tkinter.

    Returns:
        str: Path to the selected folder, or empty string if cancelled.
    """

    root = Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    return folder_path

# Main script
if __name__ == "__main__":
    # get execution arguments
    apply_crop = eval(sys.argv[1]) if len(sys.argv) > 1 else True
    font_size = eval(sys.argv[2]) if len(sys.argv) > 2 else 200

    folder_path = select_folder()  # Ask user to select folder
    if folder_path:
        process_images(folder_path, apply_crop, font_size)
    else:
        print("No folder selected, exiting.")

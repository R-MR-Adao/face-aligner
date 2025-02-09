import os
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm
import cv2
import termcolor

# Function to get the date from EXIF metadata
def get_exif_date(image_path):
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


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_and_offset(y, h_, H):
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

def crop_face(image_path, output_path, margin=0.3):
    def _raise(msg):
        termcolor.cprint(f" {msg}", "red")
        cv2.imwrite(output_path, img)
    
    # Read the image
    img = cv2.imread(image_path)
    H, W, *_ = img.shape
    
    # Convert the image to grayscale (face detection works on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(1000, 1000))
    
    if len(faces) == 0:
        return _raise(f"No face detected in the image {image_path}")
    
    # Assume the first face is the main one
    areas = [f[2] * f[3] for f in faces]
    x, y, w, h = faces[areas.index(max(areas))]

    h = max(H * 0.75 / (1 + 2 * margin), h )
    w = max(W * 0.75 / (1 + 2 * margin), h )

    x, y, = x + w/2,  y + h/2
    
    y0 = max(0, y - h * (1/2 + margin))
    y1 = min(H, y + h * (1/2 + margin))

    h_ = y1 - y0                # new height
    w_ = min(W, W / H * h_)     # new width
    
    if h_ < 0 or w_ < 0:
        return _raise(f"Failed to crop image {image_path}")
    
    h_ = H / W * w_             # restrict height
    
    # Add margin around the detected face, but stay within the image bounds
    y0, y1 = crop_and_offset(y, h_, H)
    x0, x1 = crop_and_offset(x, w_, W)

    if x1 <= x0 or y1 <= y0:
        return _raise(f"Crop resulted in empty image for {image_path}")
       
    # Crop the image around the face
    cropped_face = img[y0:y1, x0:x1]

    termcolor.cprint(f" Successfully processed {image_path}", "green")    
    cv2.imwrite(output_path, cropped_face)

def process_images(folder_path):
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
        crop_face(image_path, output_path)
        
        # add date watermark
        # Get the date from EXIF metadata
        exif_date = get_exif_date(image_path)
        watermark_text = exif_date if exif_date else "Unknown Date"  # Fallback text if no date found
        add_watermark(output_path, output_path, watermark_text)

# Function to open a folder dialog to select folder path
def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    return folder_path

# Main script
if __name__ == "__main__":
    folder_path = select_folder()  # Ask user to select folder
    if folder_path:
        process_images(folder_path)
    else:
        print("No folder selected, exiting.")

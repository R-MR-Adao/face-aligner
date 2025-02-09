import os
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS

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
        print(f"Runtime Exception: {e}")
        return None

# Function to add watermark
def add_watermark(image_path, output_path, fontsize=200):
    # Get the date from EXIF metadata
    exif_date = get_exif_date(image_path)
    watermark_text = exif_date if exif_date else "Unknown Date"  # Fallback text if no date found

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

# Main function to process all images in a folder
def add_watermarks_to_images(folder_path):
    # Ensure the output folder exists
    output_folder = os.path.join(folder_path, "watermarked")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Only process images (filter out non-image files)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            add_watermark(image_path, output_path)
            print(f"Watermark added to {filename}")

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
        add_watermarks_to_images(folder_path)
    else:
        print("No folder selected, exiting.")

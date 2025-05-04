# Face Aligner

**Face Aligner** is a Python-based tool that automates facial alignment and enhancement in batches of images. It detects faces using MTCNN, aligns them with geometric transformations, adds watermarks based on EXIF dates, and optionally replaces the background with a blurred version. Perfect for organizing photo archives, creating polished portraits, or preprocessing images for machine learning datasets.

---

## ✨ Features

- ✅ Face detection and alignment using MTCNN  
- ✅ Optional blurred background enhancement  
- ✅ Watermark images with original photo dates (from EXIF metadata)  
- ✅ Batch processing of entire folders  
- ✅ Simple GUI folder selector (via Tkinter)  

# 📷 Sample Use Cases

- Clean and align portrait photos
- Preprocess image datasets for training facial recognition models
- Timestamp archival photos
- Quickly batch-process event or family photos

# 🚀 Usage
You can run the script directly from the command line:

```bash
python face-aligner.py [apply_crop] [font_size]
```

## Arguments:
 - apply_crop (optional): Set to True to detect and align faces (default is True)
 - font_size (optional): Font size for the watermark text (default is 200)

### Example:

```bash
python face-aligner.py True 180
```

When run, a folder selection dialog will appear. Choose the folder containing the images you want to process.

# 🖼 Output
Processed images are saved in a subfolder named processed/ inside the original image folder. For each image:

- If cropping is enabled:
  - Face is aligned and centered.
  - Background may be enhanced with a blurred zoom effect.
- A watermark is added to the top-right corner with the image's original date (from EXIF metadata).

# 🧠 How It Works
- Face Detection: Uses MTCNN to locate key facial landmarks.
- Alignment: Applies geometric transformations to align facial features based on predefined target positions.
- Watermarking: Extracts DateTimeOriginal from EXIF metadata and overlays it on the image.
- Background Enhancement (optional): Creates a blurred, magnified version of the image to fill black or green-screened areas.

# 📝 License
MIT License. See [License](License.txt) for details.

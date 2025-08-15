# Step-by-Step Guide: Preparing Labeled Data for Object Detection

## Overview: What is Object Detection Labeling?

Object detection requires you to:
- **Draw bounding boxes** around objects in images
- **Assign class labels** to each box (e.g., "car", "person", "dog")
- **Create annotation files** that store this information

## Step 1: Define Your Classes

### 1.1 List All Object Types
Write down every type of object you want to detect:
```
Examples:
- Person
- Car
- Bicycle
- Traffic light
- Stop sign
```

### 1.2 Create Class Guidelines
For each class, define:
- **What to include**: "Car includes sedans, SUVs, trucks"
- **What to exclude**: "Don't label toy cars or car parts"
- **Edge cases**: "Label partially visible cars if >30% visible"

### 1.3 Number Your Classes
Assign each class a number (starting from 0):
```
0: Person
1: Car
2: Bicycle
3: Traffic_light
4: Stop_sign
```

## Step 2: Choose Your Annotation Format

### 2.1 Common Formats

**YOLO Format** (most popular):
```
class_id center_x center_y width height
0 0.5 0.3 0.2 0.4
```

**COCO Format** (JSON):
```json
{
  "bbox": [x, y, width, height],
  "category_id": 1,
  "id": 1
}
```

**Pascal VOC Format** (XML):
```xml
<object>
  <name>car</name>
  <bndbox>
    <xmin>100</xmin>
    <ymin>150</ymin>
    <xmax>300</xmax>
    <ymax>400</ymax>
  </bndbox>
</object>
```

### 2.2 Recommendation
Start with **YOLO format** - it's simple and widely supported.

## Step 3: Set Up Your Folder Structure

### 3.1 Organize Your Data
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt
└── dataset.yaml
```

### 3.2 Create Essential Files

**classes.txt**:
```
Person
Car
Bicycle
Traffic_light
Stop_sign
```

**dataset.yaml** (for YOLO):
```yaml
train: images/train
val: images/val
test: images/test

nc: 5  # number of classes
names: ['Person', 'Car', 'Bicycle', 'Traffic_light', 'Stop_sign']
```

## Step 4: Choose an Annotation Tool

### 4.1 Free Tools

**LabelImg** (Recommended for beginners):
- Simple interface
- Supports YOLO and Pascal VOC formats
- Download: [LabelImg GitHub](https://github.com/tzutalin/labelImg)

**CVAT** (Computer Vision Annotation Tool):
- Web-based
- Team collaboration
- Advanced features
- Website: [CVAT.org](https://www.cvat.org/)

**Roboflow** (Free tier available):
- Web-based
- Auto-suggestions
- Format conversion
- Website: [Roboflow.com](https://roboflow.com/)

### 4.2 Installation Example (LabelImg)
```bash
pip install labelImg
labelImg
```

## Step 5: Start Annotating

### 5.1 Basic Annotation Process

1. **Load your image** in the annotation tool
2. **Select the rectangle tool**
3. **Draw a bounding box** around each object
4. **Assign the correct class label**
5. **Save the annotation**
6. **Move to the next image**

### 5.2 Bounding Box Guidelines

**Good Bounding Box**:
- Tight around the object
- Includes all visible parts
- Doesn't include unnecessary background

**Example**:
```
✅ GOOD: Box tightly around the car
❌ BAD: Box includes half the road
❌ BAD: Box cuts off part of the car
```

### 5.3 What to Label

**Include**:
- All instances of your target classes
- Partially visible objects (if >30% visible)
- Objects in the background

**Skip**:
- Objects smaller than 10x10 pixels
- Heavily occluded objects (<30% visible)
- Unclear/ambiguous objects

## Step 6: Quality Control

### 6.1 Common Mistakes to Avoid

1. **Inconsistent labeling**: Same object labeled differently
2. **Missing objects**: Forgetting small or background objects
3. **Wrong class assignments**: Labeling a truck as a car
4. **Loose bounding boxes**: Too much empty space around objects
5. **Duplicate labels**: Multiple boxes for the same object

### 6.2 Review Process

1. **Self-review**: Check your own work after each session
2. **Cross-validation**: Have someone else review random samples
3. **Use visualization tools** to spot issues:

```python
# Simple visualization script
import cv2
import matplotlib.pyplot as plt

def visualize_annotations(image_path, label_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
```

## Step 7: Data Split and Organization

### 7.1 Split Your Data

**Typical splits**:
- **Training**: 70-80% of your data
- **Validation**: 10-15% of your data
- **Testing**: 10-15% of your data

### 7.2 Ensure Balance

Each split should have:
- Similar distribution of classes
- Variety of scenarios (lighting, angles, backgrounds)
- Range of object sizes

### 7.3 Move Files to Correct Folders

```python
import os
import shutil
from sklearn.model_selection import train_test_split

# Example script to split data
def split_dataset(image_dir, label_dir, train_ratio=0.8, val_ratio=0.1):
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    # Split into train, val, test
    train_imgs, temp_imgs = train_test_split(images, test_size=1-train_ratio)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5)
    
    # Move files to appropriate folders
    for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        for img in img_list:
            # Move image
            shutil.move(os.path.join(image_dir, img), 
                       f'dataset/images/{split_name}/{img}')
            
            # Move corresponding label
            label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(label_dir, label_file)):
                shutil.move(os.path.join(label_dir, label_file),
                           f'dataset/labels/{split_name}/{label_file}')
```

## Step 8: Validation and Testing

### 8.1 Check Data Integrity

Run these checks before training:

```python
def validate_dataset(dataset_path):
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_dir = f"{dataset_path}/images/{split}"
        lbl_dir = f"{dataset_path}/labels/{split}"
        
        images = set([f.split('.')[0] for f in os.listdir(img_dir)])
        labels = set([f.split('.')[0] for f in os.listdir(lbl_dir)])
        
        # Check for missing labels
        missing_labels = images - labels
        if missing_labels:
            issues.append(f"Missing labels in {split}: {missing_labels}")
        
        # Check for orphaned labels
        orphaned_labels = labels - images
        if orphaned_labels:
            issues.append(f"Orphaned labels in {split}: {orphaned_labels}")
    
    return issues
```

### 8.2 Statistics Review

Check your dataset statistics:
- **Number of images per split**
- **Number of objects per class**
- **Average objects per image**
- **Bounding box size distribution**

## Step 9: Best Practices

### 9.1 Annotation Guidelines

1. **Be consistent**: Use the same criteria throughout
2. **Label everything**: Don't skip difficult cases
3. **Use precise boxes**: Tight but complete coverage
4. **Document decisions**: Keep notes on edge cases
5. **Take breaks**: Avoid fatigue-induced errors

### 9.2 Efficiency Tips

1. **Use keyboard shortcuts** in your annotation tool
2. **Batch similar images** together
3. **Start with easy, clear examples**
4. **Use pre-trained models** for initial suggestions (if available)
5. **Set daily targets** (e.g., 50 images per day)

### 9.3 Quality Metrics to Track

- **Annotation speed**: Images per hour
- **Inter-annotator agreement**: Consistency between different people
- **Review rate**: Percentage of annotations that need correction

## Step 10: Final Checklist

Before using your dataset:

- [ ] All images have corresponding label files
- [ ] No empty label files (unless image has no objects)
- [ ] Class IDs are consistent and start from 0
- [ ] Bounding box coordinates are normalized (0-1 for YOLO)
- [ ] Train/val/test splits are properly separated
- [ ] No data leakage between splits
- [ ] Annotation guidelines are documented
- [ ] Sample visualizations look correct

## Common Troubleshooting

### Issue: "My model isn't learning"
**Solutions**:
- Check if labels match images correctly
- Verify class IDs are correct
- Ensure bounding boxes aren't too small
- Review annotation consistency

### Issue: "Annotations take too long"
**Solutions**:
- Use better annotation tools with shortcuts
- Start with pre-trained model suggestions
- Focus on quality over quantity initially
- Consider hiring professional annotators for large datasets

### Issue: "Inconsistent results"
**Solutions**:
- Create detailed annotation guidelines
- Use multiple annotators and measure agreement
- Regular quality reviews
- Standardize lighting/viewing conditions while annotating

## Next Steps

Once your data is ready:
1. **Train a baseline model** to test data quality
2. **Analyze model performance** to identify labeling issues
3. **Iterate and improve** annotations based on results
4. **Scale up** annotation for better performance

Remember: **Good data beats good algorithms!** Spend time getting your annotations right - it's the foundation of a successful object detection system.

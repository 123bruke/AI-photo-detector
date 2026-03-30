def load_images_from_folder(folder_path, label, img_size=(224, 224), max_images=None):
    images = []
    labels = []
    if not os.path.exists(folder_path):
        print(f"Path does not exist: {folder_path}")
        return np.array([]), np.array([])
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images from {folder_path}...")
    
    for img_file in tqdm(image_files):
        try:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img = cv2.resize(img, img_size)
            # Normalize to [0, 1]
            img = img / 255.0
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    return np.array(images), np.array(labels)

# Load datasets (limit images for faster training - adjust as needed)
IMG_SIZE = (224, 224)

# Load AI-generated people images (label = 1 for AI)
print("\n" + "="*50)
print("Loading AI-generated images...")
print("="*50)
ai_images, ai_labels = load_images_from_folder(ai_people_path, label=1, img_size=IMG_SIZE)

# Load Real people images (label = 0 for Real)
print("\n" + "="*50)
print("Loading Real images...")
print("="*50)
real_images, real_labels = load_images_from_folder(real_people_path, label=0, img_size=IMG_SIZE)

print(f"\nLoaded {len(ai_images)} AI-generated images")
print(f"Loaded {len(real_images)} Real images")

X = np.concatenate([ai_images, real_images], axis=0)
y = np.concatenate([ai_labels, real_labels], axis=0)

print(f"\nTotal dataset shape: {X.shape}")
print(f"Class distribution:")
print(f"  AI (1): {np.sum(y == 1)}")
print(f"  Real (0): {np.sum(y == 0)}") 

def plot_sample_images(images, labels, num_samples=10):
    """
    Display sample images from the dataset
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Get indices for AI and Real images
    ai_indices = np.where(labels == 1)[0]
    real_indices = np.where(labels == 0)[0]
    
    # Display 5 AI and 5 Real images
    for i in range(5):
        if i < len(ai_indices):
            idx = ai_indices[i]
            axes[i].imshow(images[idx])
            axes[i].set_title(f"AI-generated", fontsize=12, color='red')
            axes[i].axis('off')
    
    for i in range(5):
        if i < len(real_indices):
            idx = real_indices[i]
            axes[i+5].imshow(images[idx]) # the color aslo have been grey 
            axes[i+5].axis('off')
    
    plt.tight_layout()
    plt.show()

# Plot sample images
plot_sample_images(X, y)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# formal data creating 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
print(f"\nTraining set distribution:")
print(f"  AI: {np.sum(y_train == 1)}")
print(f"  Real: {np.sum(y_train == 0)}")


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test, only rescaling
val_test_datagen = ImageDataGenerator()

# Create data generators
batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_test_datagen.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)
test =  ImageDataGenerator().flow
test_generator = val_test_datagen.flow(
    X_test, y_test,
    batch_size=batch_size,
    shuffle=False
)

print("Data generators created successfully!")
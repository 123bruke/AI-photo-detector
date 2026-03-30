
model.save('ai_vs_real_classifier.h5')
print("Model saved as 'ai_vs_real_classifier.h5'")

# Save in TensorFlow SavedModel format
model.save('saved_model', save_format='tf')
print("Model saved in SavedModel format")

# Convert to TensorFlow Lite for mobile deployment (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('ai_vs_real_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to TensorFlow Lite")

def predict_image(image_path, model, img_size=(224, 224)):
 
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    proba = model.predict(img, verbose=0)[0][0]
    
    if proba > 0.5:
        prediction = 'AI-generated'
        confidence = proba
    else:
        prediction = 'Real'
        confidence = 1 - proba
    
    return prediction, confidence

# Test the prediction function with sample images
print("\n" + "="*50)
print("Testing Prediction Function")
print("="*50)

# Test on a few test images
for i in range(min(5, len(X_test))):
    pred, conf = predict_image(None, model)  # Note: This needs actual paths
    # For demonstration, we'll use test set images directly
    img = X_test[i]
    proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    pred_class = 'AI' if proba > 0.5 else 'Real'
    true_class = 'AI' if y_test[i] == 1 else 'Real'
    
    print(f"Sample {i+1}: True: {true_class}, Predicted: {pred_class}, Confidence: {proba:.4f}")

def visualize_predictions(model, X_test, y_test, num_samples=10):

    # Randomly select test images
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = y_test[idx]
        
        # Predict
        proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = 1 if proba > 0.5 else 0

        color = 'green' if pred_label == true_label else 'red'
        axes[i].imshow(img)
        title = f"True: {'AI' if true_label==1 else 'Real'}\nPred: {'AI' if pred_label==1 else 'Real'}\nConf: {proba:.2f}"
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize predictions
visualize_predictions(model, X_test, y_test)
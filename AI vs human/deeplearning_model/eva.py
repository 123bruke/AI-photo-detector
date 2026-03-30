def predict_new_image(image_path, model, show_image=True):
    prediction, confidence = predict_image(image_path, model)
    
    if prediction is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    if show_image:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
        plt.show()
    
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    
    return prediction, confidence

print(" Model training and evaluation complete!")
print("To use the model on new images, use the predict_new_image() function")
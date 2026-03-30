
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict(X_test, verbose=0)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.4f}")
print(f"TPR at optimal threshold: {tpr[optimal_idx]:.4f}")
print(f"FPR at optimal threshold: {fpr[optimal_idx]:.4f}")


print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Architecture: MobileNetV2 + Custom Head")
print(f"Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}x3")
print(f"Total Parameters: {model.count_params():,}")
print(f"Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
print(f"  AUC-ROC: {test_auc:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if test_accuracy > 0.9:
    print("✅ Excellent performance - Model can reliably distinguish AI from real images")
elif test_accuracy > 0.8:
    print("👍 Good performance - Model shows strong capability in detection")
elif test_accuracy > 0.7:
    print("⚠️ Moderate performance - Further improvement needed")
else:
    print("❌ Poor performance - Consider more data or different architecture")

print("\nPotential Improvements:")
print("1. Use more advanced architectures (EfficientNet, ConvNeXt)")
print("2. Increase dataset size with data augmentation")
print("3. Implement ensemble methods")
print("4. Use more fine-tuning epochs")
print("5. Try different preprocessing techniques")
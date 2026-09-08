# Calculate class centroids from the training data
classes = np.unique(y_train)
centroids = {}

for k in classes:
    centroids[k] = np.mean(X_train_scaled[y_train == k], axis=0)

print("Class centroids:")
for k in classes:
    print(f"{k} ({class_names[k]}):")
    print(np.round(centroids[k], 6))



# Prediction from scratch using Euclidean distance
def centroid_predict(X_samples, centroids):
    predictions = []

    for x in X_samples:
        distances = []

        for k in classes:
            distance = np.sqrt(np.sum((x - centroids[k]) ** 2))
            distances.append(distance)

        predictions.append(classes[np.argmin(distances)])

    return np.array(predictions)

y_pred_centroid = centroid_predict(X_test_scaled, centroids)

centroid_accuracy = accuracy_score(y_test, y_pred_centroid)
centroid_cm = confusion_matrix(y_test, y_pred_centroid)

print("Test predictions:")
print(y_pred_centroid)

print(f"\nTest accuracy: {centroid_accuracy * 100:.2f}%")

print("\nConfusion matrix:")
print(centroid_cm)

# Perceptron hyperparameters
epochs = 500
learning_rate = 0.1

def train_binary_perceptron(X, y_binary, epochs, learning_rate):
    # Initialize weights and bias
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(epochs):
        for x_i, y_i in zip(X, y_binary):
            # Linear output
            z = np.dot(w, x_i) + b

            # Step activation
            y_hat = 1 if z >= 0 else 0

            # Perceptron update
            error = y_i - y_hat
            w = w + learning_rate * error * x_i
            b = b + learning_rate * error

    return w, b


# Train one binary Perceptron for each class (OvR)
weights = {}
biases = {}

for k in classes:
    y_binary = (y_train == k).astype(int)

    weights[k], biases[k] = train_binary_perceptron(
        X_train_scaled,
        y_binary,
        epochs,
        learning_rate
    )

print("Number of epochs:", epochs)
print("Learning rate:", learning_rate)

print("\nLearned parameters:")
for k in classes:
    print(f"Class {k} ({class_names[k]}):")
    print("Weights:", np.round(weights[k], 6))
    print("Bias   :", round(float(biases[k]), 6))



def perceptron_scores(X_samples):
    # Calculate z = w^T x + b for every OvR classifier
    score_matrix = np.zeros((X_samples.shape[0], len(classes)))

    for j, k in enumerate(classes):
        score_matrix[:, j] = X_samples @ weights[k] + biases[k]

    return score_matrix


def perceptron_predict(X_samples):
    score_matrix = perceptron_scores(X_samples)

    predictions = []

    for scores in score_matrix:
        # Step activation for each OvR classifier
        step_outputs = (scores >= 0).astype(int)

        positive_classes = np.where(step_outputs == 1)[0]

        if len(positive_classes) == 0:
            # If no classifier fires, choose the largest decision score
            predicted_index = np.argmax(scores)
        else:
            # If one or more classifiers fire, choose the highest score
            predicted_index = positive_classes[np.argmax(scores[positive_classes])]

        predictions.append(classes[predicted_index])

    return np.array(predictions)


# Training and testing predictions
y_train_pred_perceptron = perceptron_predict(X_train_scaled)
y_test_pred_perceptron = perceptron_predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred_perceptron)
test_accuracy = accuracy_score(y_test, y_test_pred_perceptron)
perceptron_cm = confusion_matrix(y_test, y_test_pred_perceptron)

print("Number of epochs:", epochs)
print("Learning rate:", learning_rate)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing accuracy : {test_accuracy * 100:.2f}%")

print("\nConfusion matrix:")
print(perceptron_cm)


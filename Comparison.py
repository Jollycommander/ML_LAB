comparison = pd.DataFrame({
    "Algorithm": [
        "Centroid-Based Mean",
        "Multiclass Perceptron (OvR)"
    ],
    "Test Accuracy": [
        centroid_accuracy,
        test_accuracy
    ]
})

comparison["Test Accuracy"] = comparison["Test Accuracy"].round(4)

print(comparison.to_string(index=False))

if test_accuracy > centroid_accuracy:
    print("\nConclusion: Perceptron performed better on the test dataset.")
elif test_accuracy < centroid_accuracy:
    print("\nConclusion: Centroid-Based Mean performed better on the test dataset.")
else:
    print("\nConclusion: Both algorithms achieved the same test accuracy.")

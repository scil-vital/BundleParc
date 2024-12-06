import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import defaultdict


def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def extract_dice_voxels(data):
    """
    Extract dice_voxels for each bundle under the 'Proposed' method.

    Parameters:
        data (dict): The JSON data loaded from the file.

    Returns:
        sorted_bundles (list): List of bundles sorted by mean dice score.
        sorted_dice_voxels (list): Corresponding dice_voxels for the sorted bundles.
    """
    dice_voxels_dict = defaultdict(list)

    for bundle, values in data.items():
        dice_voxels = values.get("ssd", {})
        for label, dice in dice_voxels.items():
            dice_for_label = dice
            if dice_voxels:  # Only include bundles with dice_voxels data
                dice_voxels_dict[bundle] += dice_for_label

    # Sort bundles by mean dice scores in descending order
    sorted_bundles = sorted(dice_voxels_dict.keys(), key=lambda b: np.mean(
        dice_voxels_dict[b]), reverse=False)
    sorted_dice_voxels = [dice_voxels_dict[bundle]
                          for bundle in sorted_bundles]

    # Print overall mean dice score
    mean_dice = np.mean([dice for bundle_dice in sorted_dice_voxels
                         for dice in bundle_dice])
    print(f"Overall Mean Dice: {mean_dice:.2f}")

    return sorted_bundles, sorted_dice_voxels


def plot_sorted_boxplots(sorted_bundles, sorted_dice_voxels):
    """
    Plot boxplots for sorted dice_voxels per bundle.

    Parameters:
        sorted_bundles (list): Bundles sorted by mean dice score.
        sorted_dice_voxels (list): Corresponding dice_voxels values.
    """
    plt.figure(figsize=(10, 15))
    plt.boxplot(sorted_dice_voxels, vert=False, patch_artist=True)

    # Customize plot
    plt.title("Proposed Method: Dice Voxels Distribution by Bundle", fontsize=20)
    plt.xlabel("Dice Score", fontsize=15)
    plt.ylabel("Bundles", fontsize=15)
    plt.yticks(range(1, len(sorted_bundles) + 1), sorted_bundles, fontsize=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add x ticks every 0.1
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Add a vertical line at the mean of all dice scores
    mean_dice = np.mean([dice for bundle_dice in sorted_dice_voxels
                         for dice in bundle_dice])
    plt.axvline(mean_dice, color='r', linestyle='--', label=f"Mean Dice: {mean_dice:.2f}")

    # Add a horizontal grid line for each bundle
    for i in range(1, len(sorted_bundles) + 1):
        plt.axhline(i, color='gray', linestyle='--', alpha=0.5)

    # Make text bigger
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Make title bigger


    # Tighten layout and show plot
    plt.tight_layout()
    plt.show()

    # Print mean of all dice scores
    print(f"Mean Dice: {mean_dice:.2f}")


def main():
    """
    Main function to load JSON data, process it, and plot boxplots.
    """
    # Input JSON file
    input_file = sys.argv[1]

    # Load data
    data = load_json(input_file)

    # Extract and sort dice_voxels
    sorted_bundles, sorted_dice_voxels = extract_dice_voxels(data)

    # Plot boxplots
    plot_sorted_boxplots(sorted_bundles, sorted_dice_voxels)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt 

def draw_results(img, signature_box, grid_info, output_path="result_detection.png"):
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img, cmap="gray")

    if signature_box is not None:
        x1, y1, x2, y2 = signature_box

        ax.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            linewidth=3
        )

        ax.text(x1, y1 - 15, "Bloc signature", fontsize=12)

    if grid_info is not None:
        x_centers = grid_info["x_centers"]
        y_centers = grid_info["y_centers"]

        for row, y in enumerate(y_centers):
            for col, x in enumerate(x_centers):
                ax.plot(x, y, marker="o", markersize=4)

            ax.text(x_centers[0] - 40, y, str(row), fontsize=10)

        ax.text(x_centers[0], y_centers[0] - 30, "Grille Student ID", fontsize=12)

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print("Image sauvegardée :", output_path)

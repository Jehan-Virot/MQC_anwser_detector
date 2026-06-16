from skimage import draw, exposure, io
import numpy as np

def draw_rect_rgb(rgb, bbox, color_value):
    x1, y1, x2, y2 = bbox
    rr, cc = draw.rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=rgb.shape)
    rgb[rr, cc, 0] = color_value[0]
    rgb[rr, cc, 1] = color_value[1]
    rgb[rr, cc, 2] = color_value[2]


def draw_results(img_gray, signature_box, grid_info, out_path):
    """
    Image debug :
    - bleu : boîte signature
    - rouge : cases ID reconstruites
    - vert : cases choisies pour former l'ID
    """
    if img_gray.dtype == bool:
        base = img_gray.astype(float)
    else:
        base = img_gray.astype(float)

    base = exposure.rescale_intensity(base, out_range=(0, 1))
    rgb = np.dstack([base, base, base])

    if signature_box is not None:
        draw_rect_rgb(rgb, signature_box, (0.0, 0.3, 1.0))

    if grid_info is not None and grid_info.get("grid") is not None:
        grid = grid_info["grid"]
        ratios = grid_info.get("ratios")

        chosen_rows = []
        if ratios is not None:
            for col in range(5):
                chosen_rows.append(int(np.argmax(ratios[:, col])))

        for row in range(10):
            for col in range(5):
                bbox = grid[row][col]
                if col < len(chosen_rows) and row == chosen_rows[col]:
                    draw_rect_rgb(rgb, bbox, (0.0, 1.0, 0.0))
                else:
                    draw_rect_rgb(rgb, bbox, (1.0, 0.0, 0.0))

    io.imsave(out_path, (rgb * 255).astype(np.uint8))


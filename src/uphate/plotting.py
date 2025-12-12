import numpy as np
import matplotlib.pyplot as plt


def create_gradient_sprite(size, cmap_name):
    """
    Creates a square RGBA image with a radial gradient circle in the middle.
    Pixels outside the circle are transparent.
    """
    # Create a grid of coordinates from -1 to 1
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Calculate radius
    R = np.sqrt(X**2 + Y**2)

    # Normalize radius for the colormap (0 at center, 1 at edge)
    # You can flip this (1 - R) if you want the "hottest" color in the center
    norm_R = 1 - R

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Apply colormap to the radius values
    # This gives us an (N, N, 4) array (RGBA)
    image = cmap(norm_R)

    # Set Alpha channel:
    # 1. Make pixels outside the circle (R > 1) completely transparent
    image[R > 1, 3] = 0.0

    # Fade the alpha towards the edge for a "glowing" effect
    image[:, :, 3] = np.maximum((1 - R).clip(0, 1) ** 0.5 * (R <= 1), 0.1)

    return image


def plot_ellipses_with_sprites(positions, axis_lengths, figsize):
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Generate the generic gradient image once
    sprite = create_gradient_sprite(size=128, cmap_name="Blues")

    # 2. Plot each ellipse as a stretched image
    for pos, axes in zip(positions, axis_lengths):
        cx, cy = pos
        width, height = axes

        # Calculate the bounding box (extent) for imshow
        # extent = [left, right, bottom, top]
        left = cx - width / 2
        right = cx + width / 2
        bottom = cy - height / 2
        top = cy + height / 2

        # Plot the image stretched to the ellipse bounds
        ax.imshow(
            sprite,
            extent=(left, right, bottom, top),
            aspect="auto",  # Allows stretching
            origin="lower",
            interpolation="bilinear",
        )  # Smooths the gradient

    # 3. Plot the black dots on top
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=1, zorder=10)
    return fig, ax

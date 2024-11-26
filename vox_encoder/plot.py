import matplotlib.pyplot as plt


def plot_voxel_data(original, reconstructed, epoch):
    fig = plt.figure(figsize=(12, 6))

    # Original Data Plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title(f"Original Voxel Data - Epoch {epoch}")

    original_occupied = original.view(-1, 4)
    orig_coords = original_occupied[original_occupied[:, 3] > 0].numpy()
    ax1.scatter(orig_coords[:, 0], orig_coords[:, 1], orig_coords[:, 2], c="r", marker="o")

    # Reconstructed Data Plot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title(f"Reconstructed Voxel Data - Epoch {epoch} (Confidence Level)")

    reconstructed_occupied = reconstructed.view(-1, 4)
    recon_coords = reconstructed_occupied[:, :3].detach().numpy()
    confidence_levels = reconstructed_occupied[:, 3].detach().numpy()

    # Use the confidence_levels as the color value for plotting
    scatter = ax2.scatter(
        recon_coords[:, 0],
        recon_coords[:, 1],
        recon_coords[:, 2],
        c=confidence_levels,
        cmap="viridis",
        marker="o",
    )

    # Adding color bar to visualize intensity levels
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.1)
    cbar.set_label("Occupancy Confidence")

    plt.show()


import numpy as np
from scipy.stats import gaussian_kde

def model(df, x_col='UMAP1', y_col='UMAP2', resolution=100,
                        x_min=None, x_max=None, y_min=None, y_max=None):
    """Create density estimates for heatmap with error handling and fixed bounds"""
    if len(df) < 10:
        return None
    try:
        x = df[x_col]
        y = df[y_col]

        # Add noise to prevent singular matrix
        x = x + np.random.normal(0, 0.0001, size=len(x))
        y = y + np.random.normal(0, 0.0001, size=len(y))

        # Use provided bounds instead of calculating from the data
        if x_min is None:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            # Add padding
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding

        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi, yi = np.meshgrid(xi, yi)

        # Calculate density
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        density = kernel(positions).reshape(resolution, resolution)

        return {
            'x': xi,
            'y': yi,
            'density': density,
            'x_flat': xi.ravel(),
            'y_flat': yi.ravel(),
            'density_flat': density.ravel()
        }
    except Exception as e:
        print(f"Error in density calculation: {e}")

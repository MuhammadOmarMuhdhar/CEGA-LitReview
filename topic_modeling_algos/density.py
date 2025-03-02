import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


def create_density_data(df, x_col='umap_0', y_col='umap_1', resolution=100,
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
    except:
        return None


def get_point_density(x_points, y_points, density_result, density_threshold=0.2):
    """
    Calculate density values at specific points and create a mask for 
    points in areas with density above the threshold
    """
    # Create points array
    points = np.vstack((density_result['x_flat'], density_result['y_flat'])).T
    
    # Get density values at topic points through interpolation
    point_density = griddata(
        points, 
        density_result['density_flat'], 
        (x_points, y_points), 
        method='linear',
        fill_value=0
    )
    
    # Normalize density to 0-1 range for thresholding
    if point_density.max() > 0:
        normalized_density = point_density / point_density.max()
    else:
        normalized_density = point_density
    
    # Create mask for points with density above threshold
    mask = normalized_density > density_threshold
    
    return mask
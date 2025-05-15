import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from featureEngineering import density

def create_cumulative_visualization(plot_df, topic_clusters):
    """
    Create an animated cumulative visualization of document density with optimized performance.
    Uses percentile-based boundaries to focus on the core data distribution.
    
    Args:
        plot_df (pd.DataFrame): DataFrame with document data
        topics_summary (pd.DataFrame): DataFrame with topic cluster information
    
    Returns:
        plotly.graph_objs._figure.Figure: Animated visualization
    """
    import numpy as np
    import plotly.graph_objs as go
    
    # Focus on the core distribution by limiting the range to data within percentiles
    def get_percentile_bounds(series, lower_percentile=1, upper_percentile=99):
        """Get the values at the specified percentiles of the data"""
        lower_bound = np.percentile(series, lower_percentile)
        upper_bound = np.percentile(series, upper_percentile)
        return lower_bound, upper_bound
    
    # Get percentile-based bounds for the core data
    x_min_core, x_max_core = get_percentile_bounds(plot_df['UMAP1'])
    y_min_core, y_max_core = get_percentile_bounds(plot_df['UMAP2'])
    
    # Add some padding to the core boundaries
    padding_factor = 0.1
    x_range = x_max_core - x_min_core
    y_range = y_max_core - y_min_core
    
    x_min_global = x_min_core - x_range * padding_factor
    x_max_global = x_max_core + x_range * padding_factor
    y_min_global = y_min_core - y_range * padding_factor
    y_max_global = y_max_core + y_range * padding_factor
    
    # Prepare and process data
    plot_df = plot_df.sort_values('date')
    unique_dates = sorted(plot_df['date'].unique())

    # Create animated figure
    fig = go.Figure()

    # Generate cumulative plots for each date
    valid_dates, traces = [], []
    for date in unique_dates:
        # Filter data for dates up to and including the current date
        cumulative_data = plot_df[plot_df['date'] <= date]
        
        if len(cumulative_data) < 10:
            continue

        # Focus only on data within our core boundaries for density calculation
        core_data = cumulative_data[
            (cumulative_data['UMAP1'] >= x_min_global) & 
            (cumulative_data['UMAP1'] <= x_max_global) &
            (cumulative_data['UMAP2'] >= y_min_global) & 
            (cumulative_data['UMAP2'] <= y_max_global)
        ]
        
        # Ensure we have enough data to calculate density
        if len(core_data) < 10:
            print(f"Not enough core data for {date}")
            continue

        # Compute density data using only core data
        try:
            density_result = density.model(
                core_data, 
                x_min=x_min_global, 
                x_max=x_max_global, 
                y_min=y_min_global, 
                y_max=y_max_global
            )
        except Exception as e:
            print(f"Density calculation failed for {date}: {e}")
            continue
        
        if density_result is not None:
            # Contour levels
            contour_levels = max(10, min(15, int(len(core_data) / 100)))
            contour_size = (density_result['density'].max() - density_result['density'].min()) / contour_levels
            
            # Contour trace
            contour = go.Contour(
                x=np.unique(density_result['x']),
                y=np.unique(density_result['y']),
                z=density_result['density'],
                contours=dict(
                    start=density_result['density'].min(),
                    end=density_result['density'].max(),
                    size=contour_size,
                    showlabels=False
                ),
                line=dict(width=1, color='black'),
                colorscale="Viridis",
                showscale=True,
                hoverinfo='skip',
                colorbar=dict(
                    tickvals=[
                        density_result['density'].min(),
                        (density_result['density'].min() + density_result['density'].max())/2,
                        density_result['density'].max() * .9
                    ],
                    ticktext=["Low", "Medium", "High"],
                                
                    title=dict(
                        text='Research<br>Intensity',  
                        font=dict(
                            size=16,
                            color='black'
                        )
                    ),
                    tickfont=dict(
                        size=14,
                        family='Arial',
                        color='black'
                    ),
                    len=1,
                ),
                visible=False
            )
            
            # Use core data for scatter points to match density visualization
            data_points = go.Scatter(
                x=core_data['UMAP1'],
                y=core_data['UMAP2'],
                mode='markers',
                hovertext=core_data['title'],
                hoverinfo='text',
                marker=dict(
                    size=3,
                    color='rgba(255, 255, 255, 0.3)'
                ),
                visible=False
            )

            # Filter topic clusters to only display those within our core area
            visible_clusters = topic_clusters[
                (topic_clusters['umap_1_mean'] >= x_min_global) &
                (topic_clusters['umap_1_mean'] <= x_max_global) &
                (topic_clusters['umap_2_mean'] >= y_min_global) &
                (topic_clusters['umap_2_mean'] <= y_max_global)
            ]

            # Topic cluster labels
            topic_labels = go.Scatter(
                x=visible_clusters['umap_1_mean'],
                y=visible_clusters['umap_2_mean'],
                mode='markers+text',
                text=visible_clusters['label'],
                textposition='middle center',
                marker=dict(
                    size=4,
                    color='rgba(255, 255, 255, 0.3)', 
                    opacity=1
                ),
                textfont=dict(color='white'),
                visible=False
            )
            
            # Add traces and store date
            fig.add_traces([contour, data_points, topic_labels])
            valid_dates.append(date)

    # Create animation steps
    steps = [
        dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Cumulative Papers Up to {valid_dates[i]}"}
            ],
            label=str(valid_dates[i])
        ) for i in range(len(valid_dates))
    ]

    # Update visibility for each step
    for i, step in enumerate(steps):
        traces_per_frame = 3
        start_idx = i * traces_per_frame
        end_idx = start_idx + traces_per_frame
        
        for j in range(start_idx, min(end_idx, len(fig.data))):
            step["args"][0]["visible"][j] = True

    # Configure layout
    fig.update_layout(
        sliders=[dict(
            active=max(0 ,len(valid_dates) - 1),  # Set active to the last/most recent date
            currentvalue={"prefix": "Year: "},
            pad={"t": 50},
            steps=steps
        )],
        title="Use the Slider to Explore How Research Topics Evolve Over Time",
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[x_min_global, x_max_global]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[y_min_global, y_max_global]
        ),
        width=1000,
        height=750,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        modebar=dict(
            add=['toggleSpikelines']
        ),
        annotations=[
            dict(
                text="Note: Each White Dot Represents a Paper",
                xref="paper", yref="paper",
                x=0.5, y=0, 
                showarrow=False,
                font=dict(size=10, color="white")
            )
        ]
    )

    # Modify config to remove trace numbers
    fig.update_layout(
        showlegend=False,
    )

    # Show last frame
    if len(fig.data) >= 3:
        for i in range(-3, 0):
            fig.data[i].visible = True

    return fig
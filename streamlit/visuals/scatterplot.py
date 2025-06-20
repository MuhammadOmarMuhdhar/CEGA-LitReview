import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_simple_scatter(plot_df, topic_clusters=None, 
                         x_col='UMAP1', y_col='UMAP2',
                         color_col=None, title_col='title',
                         width=1000, height=750):
    """
    Create a simple scatter plot to diagnose UMAP embedding issues.
    
    Args:
        plot_df: DataFrame with UMAP coordinates
        topic_clusters: Optional DataFrame with topic cluster information
        x_col, y_col: Column names for UMAP coordinates
        color_col: Optional column to color points by (e.g., 'date', 'cluster', etc.)
        title_col: Column name for hover text
        width, height: Plot dimensions
    
    Returns:
        plotly.graph_objs._figure.Figure: Simple scatter plot
    """
    
    fig = go.Figure()
    
    # Create basic scatter plot
    if color_col and color_col in plot_df.columns:
        # Colored by specified column
        scatter = go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=3,
                color=plot_df[color_col],
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title=color_col)
            ),
            text=plot_df[title_col] if title_col in plot_df.columns else None,
            hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>',
            name='Papers'
        )
    else:
        # Single color
        scatter = go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(100, 150, 255, 0.6)',
                opacity=0.7
            ),
            text=plot_df[title_col] if title_col in plot_df.columns else None,
            hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>',
            name='Papers'
        )
    
    fig.add_trace(scatter)
    
    # Add topic labels if provided
    if topic_clusters is not None and len(topic_clusters) > 0:
        # Check if the required columns exist
        if 'umap_1_mean' in topic_clusters.columns and 'umap_2_mean' in topic_clusters.columns:
            topic_labels = go.Scatter(
                x=topic_clusters['umap_1_mean'],
                y=topic_clusters['umap_2_mean'],
                mode='markers+text',
                text=topic_clusters['label'] if 'label' in topic_clusters.columns else topic_clusters.index,
                textposition='middle center',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                textfont=dict(color='white', size=10),
                name='Topic Centers',
                hovertemplate='Topic: %{text}<extra></extra>'
            )
            fig.add_trace(topic_labels)
    
    # Update layout
    fig.update_layout(
        title=f"UMAP Scatter Plot - {len(plot_df)} Papers",
        xaxis=dict(
            title=f"{x_col}",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=f"{y_col}",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=width,
        height=height,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig

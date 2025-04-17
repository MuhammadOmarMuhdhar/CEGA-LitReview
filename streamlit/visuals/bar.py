import streamlit as st
import plotly.express as px
import pandas as pd

def create(
    data: pd.DataFrame, 
    x_column: str = 'country', 
    y_column: str = 'count', 
    title: str = 'Institutions by Country',
    width: int = 800,
    height: int = 500,
    margin: dict = None,
    show_data: bool = True,
    coord_flip: bool = False
) -> px.bar:
    """
    Create a Plotly bar chart with Streamlit integration and optional coordinate flip.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data to be plotted
    x_column : str, optional
        Name of the column to use for x-axis (default: 'country')
    y_column : str, optional
        Name of the column to use for y-axis (default: 'count')
    title : str, optional
        Title of the chart (default: 'Institutions by Country')
    width : int, optional
        Width of the chart in pixels (default: 800)
    height : int, optional
        Height of the chart in pixels (default: 500)
    margin : dict, optional
        Custom margins for the chart. If None, uses default margins
    show_data : bool, optional
        Whether to display the underlying DataFrame (default: True)
    coord_flip : bool, optional
        Whether to flip x and y coordinates (default: False)
    
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        The created Plotly bar chart
    """
    # Set default margins if not provided
    # if margin is None:
    #     margin = dict(l=50, r=50, t=50, b=50)
    
    # Create the bar chart
    if coord_flip:
        # If coordinate flip is True, swap x and y
        fig = px.bar(
            data, 
            x=y_column, 
            y=x_column, 
            title=title,
            orientation='h'  # Horizontal orientation for flipped coordinates
        )
    else:
        # Default vertical bar chart
        fig = px.bar(
            data, 
            x=x_column, 
            y=y_column, 
            title=title
        )
    
    # Customize layout
    fig.update_layout(
        width=width,
        height=height,
        margin=margin,
        showlegend=False
    )
    
    # Display chart in Streamli    

    return fig

# Example usage:
# Uncomment and modify the following lines as needed
# country = pd.DataFrame({
#     'country': ['USA', 'UK', 'Canada', 'Germany'],
#     'count': [50, 30, 20, 15]
# })
# 
# # Vertical bar chart (default)
# create_country_bar_chart(country)
# 
# # Horizontal bar chart (coordinate flipped)
# create_country_bar_chart(country, coord_flip=True)




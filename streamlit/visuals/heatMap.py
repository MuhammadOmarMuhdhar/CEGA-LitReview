import numpy as np
from scipy import ndimage
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

class heatmap:
    """
    Memory-safe cumulative visualization class with Streamlit progress tracking.
    Uses intelligent sampling to prevent crashes while maintaining visual quality.
    """
    
    def __init__(self, plot_df, topic_clusters=None, resolution=50, sigma=2.0):
        """
        Initialize the visualization class.
        
        Args:
            plot_df (pd.DataFrame): DataFrame with document data including 'UMAP1', 'UMAP2', 'date', 'title'
            topic_clusters (pd.DataFrame, optional): DataFrame with topic cluster information
            resolution (int): Grid resolution for density calculation (default 50)
            sigma (float): Gaussian blur amount for density smoothing (default 2.0)
        """
        self.plot_df = plot_df.copy()
        self.topic_clusters = topic_clusters
        self.resolution = resolution
        self.sigma = sigma
        
        # Validation
        required_cols = ['UMAP1', 'UMAP2', 'date', 'title']
        missing_cols = [col for col in required_cols if col not in self.plot_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in plot_df: {missing_cols}")
        
        if self.topic_clusters is not None:
            topic_required_cols = ['umap_1_mean', 'umap_2_mean', 'label']
            missing_topic_cols = [col for col in topic_required_cols if col not in self.topic_clusters.columns]
            if missing_topic_cols:
                raise ValueError(f"Missing required columns in topic_clusters: {missing_topic_cols}")
    
    def _get_percentile_bounds(self, series, lower_percentile=1, upper_percentile=99):
        """Get the values at the specified percentiles of the data"""
        lower_bound = np.percentile(series, lower_percentile)
        upper_bound = np.percentile(series, upper_percentile)
        return lower_bound, upper_bound
    
    def _calculate_global_bounds(self, progress_bar):
        """Calculate global bounds for the visualization based on percentiles"""
        
        # Get percentile-based bounds for the core data
        x_min_core, x_max_core = self._get_percentile_bounds(self.plot_df['UMAP1'])
        progress_bar.progress(15)
        
        y_min_core, y_max_core = self._get_percentile_bounds(self.plot_df['UMAP2'])
        progress_bar.progress(20)
        
        # Add some padding to the core boundaries
        padding_factor = 0.1
        x_range = x_max_core - x_min_core
        y_range = y_max_core - y_min_core
        
        self.x_min_global = x_min_core - x_range * padding_factor
        self.x_max_global = x_max_core + x_range * padding_factor
        self.y_min_global = y_min_core - y_range * padding_factor
        self.y_max_global = y_max_core + y_range * padding_factor
    
    def _calculate_dynamic_resolution(self, data_size):
        """
        Calculate optimal resolution based on data size.
        Less data = higher resolution for better detail
        More data = lower resolution for performance
        """
        if data_size < 100:
            return min(100, self.resolution * 2)  # High detail for small datasets
        elif data_size < 500:
            return min(80, int(self.resolution * 1.5))
        elif data_size < 2000:
            return self.resolution  # Use default
        elif data_size < 10000:
            return max(30, int(self.resolution * 0.8))
        elif data_size < 50000:
            return max(25, int(self.resolution * 0.6))
        else:
            return max(20, int(self.resolution * 0.4))  # Low resolution for huge datasets

    def _smart_sampling_density_model(self, df, x_col='UMAP1', y_col='UMAP2'):
        """
        Memory-safe density estimation with intelligent sampling AND dynamic resolution.
        """
        if len(df) < 10:
            return None
            
        try:
            # Define sampling thresholds
            data_size = len(df)
            
            # Calculate dynamic resolution based on data size
            dynamic_resolution = self._calculate_dynamic_resolution(data_size)
            
            if data_size < 1000:
                sample_pct = 1.0  # Use all data
                max_sample = data_size
            elif data_size < 5000:
                sample_pct = 0.8  # 80%
                max_sample = int(data_size * sample_pct)
            elif data_size < 20000:
                sample_pct = 0.5  # 50%
                max_sample = int(data_size * sample_pct)
            elif data_size < 50000:
                sample_pct = 0.3  # 30%
                max_sample = int(data_size * sample_pct)
            elif data_size < 100000:
                sample_pct = 0.2  # 20%
                max_sample = int(data_size * sample_pct)
            else:
                sample_pct = 0.1  # 10%
                max_sample = int(data_size * sample_pct)
            
            # Smart sampling that preserves spatial distribution
            if sample_pct < 1.0:
                sampled_df = self._stratified_spatial_sample(df, max_sample, x_col, y_col)
            else:
                sampled_df = df
            
            x = sampled_df[x_col].values
            y = sampled_df[y_col].values
            
            # Create 2D histogram with DYNAMIC resolution
            hist, x_edges, y_edges = np.histogram2d(
                x, y, 
                bins=dynamic_resolution,
                range=[[self.x_min_global, self.x_max_global], 
                       [self.y_min_global, self.y_max_global]]
            )
            
            # Apply Gaussian smoothing using FFT
            density = ndimage.gaussian_filter(hist.T, sigma=self.sigma, mode='constant')
            
            # Scale density to account for sampling
            if sample_pct < 1.0:
                density = density / sample_pct
            
            # Create coordinate meshes
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            xi, yi = np.meshgrid(x_centers, y_centers)
            
            return {
                'x': xi,
                'y': yi,
                'density': density,
                'x_flat': xi.ravel(),
                'y_flat': yi.ravel(),
                'density_flat': density.ravel()
            }
            
        except Exception as e:
            st.error(f"Error in density calculation: {e}")
            return None
    
    def _stratified_spatial_sample(self, df, n_sample, x_col, y_col):
        """
        Stratified spatial sampling that preserves the spatial distribution of points.
        """
        try:
            # Create spatial grid for stratification
            n_bins = max(10, min(50, int(np.sqrt(n_sample / 10))))  # Adaptive grid size
            
            x_bins = pd.cut(df[x_col], bins=n_bins, labels=False)
            y_bins = pd.cut(df[y_col], bins=n_bins, labels=False)
            
            # Create stratification groups
            df_temp = df.copy()
            df_temp['spatial_bin'] = x_bins * n_bins + y_bins
            
            # Sample proportionally from each spatial bin
            sampled_dfs = []
            bin_counts = df_temp['spatial_bin'].value_counts()
            
            for bin_id, count in bin_counts.items():
                bin_data = df_temp[df_temp['spatial_bin'] == bin_id]
                # Calculate proportional sample size for this bin
                bin_sample_size = max(1, int((count / len(df)) * n_sample))
                
                if len(bin_data) <= bin_sample_size:
                    sampled_dfs.append(bin_data)
                else:
                    sampled_bin = bin_data.sample(n=bin_sample_size, random_state=42)
                    sampled_dfs.append(sampled_bin)
            
            # Combine all sampled bins
            sampled_df = pd.concat(sampled_dfs, ignore_index=True)
            
            # If we're still over target, randomly sample down
            if len(sampled_df) > n_sample:
                sampled_df = sampled_df.sample(n=n_sample, random_state=42)
            
            # Remove temporary column
            sampled_df = sampled_df.drop('spatial_bin', axis=1)
            
            return sampled_df
            
        except Exception as e:
            # Fallback to simple random sampling
            return df.sample(n=min(n_sample, len(df)), random_state=42)
    
    def _prepare_data(self, progress_bar):
        """Prepare and sort data by date"""

        # Sort data by date
        self.plot_df = self.plot_df.sort_values('date')
        progress_bar.progress(25)
        
        # Get unique dates
        self.unique_dates = sorted(self.plot_df['date'].unique())
        progress_bar.progress(30)
    
    def _generate_traces(self, progress_bar):
        """Generate traces for each date with progress tracking"""
        
        self.valid_dates = []
        fig = go.Figure()
        
        # Progress tracking
        total_dates = len(self.unique_dates)
        base_progress = 30  # Starting from 30% after data prep
        trace_progress_range = 50  # Use 50% of progress bar for this step
        
        for i, date in enumerate(self.unique_dates):
            # Update progress
            current_progress = base_progress + int((i / total_dates) * trace_progress_range)
            progress_bar.progress(current_progress)
            
            # Filter data for dates up to and including the current date
            cumulative_data = self.plot_df[self.plot_df['date'] == date]
            
            if len(cumulative_data) < 10:
                continue

            # Focus only on data within our core boundaries
            core_data = cumulative_data[
                (cumulative_data['UMAP1'] >= self.x_min_global) & 
                (cumulative_data['UMAP1'] <= self.x_max_global) &
                (cumulative_data['UMAP2'] >= self.y_min_global) & 
                (cumulative_data['UMAP2'] <= self.y_max_global)
            ]
            
            # Ensure we have enough data to calculate density
            if len(core_data) < 10:
                continue

            # Compute density using memory-safe sampling
            try:
                density_result = self._smart_sampling_density_model(core_data)
            except Exception as e:
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
                            font=dict(size=16, color='black')
                        ),
                        tickfont=dict(size=14, family='Arial', color='black'),
                        len=1,
                    ),
                    visible=False
                )
                
                max_display_points = 10000
                if len(core_data) > max_display_points:
                    display_data = core_data.sample(n=max_display_points, random_state=42)
                else:
                    display_data = core_data
                
                data_points = go.Scatter(
                    x=display_data['UMAP1'],
                    y=display_data['UMAP2'],
                    mode='markers',
                    hovertext=display_data['title'],
                    hoverinfo='text',
                    marker=dict(
                        size=3,
                        color='rgba(255, 255, 255, 0.3)'
                    ),
                    visible=False
                )

                # Add topic cluster labels if provided
                if self.topic_clusters is not None:
                    # Filter topic clusters to only display those within our core area
                    visible_clusters = self.topic_clusters[
                        (self.topic_clusters['umap_1_mean'] >= self.x_min_global) &
                        (self.topic_clusters['umap_1_mean'] <= self.x_max_global) &
                        (self.topic_clusters['umap_2_mean'] >= self.y_min_global) &
                        (self.topic_clusters['umap_2_mean'] <= self.y_max_global)
                    ]

                    # Topic cluster labels
                    topic_labels = go.Scatter(
                        x=visible_clusters['umap_1_mean'],
                        y=visible_clusters['umap_2_mean'],
                        mode='markers+text',
                        text=visible_clusters['label'],
                        textposition='middle center',
                        marker=dict(
                            size=10,
                            color='rgba(255, 255, 255, 0.3)', 
                            opacity=1
                        ),
                        textfont=dict(color='white'),
                        visible=False
                    )
                    
                    # Add traces with topic labels
                    fig.add_traces([contour, data_points, topic_labels])
                else:
                    # Add traces without topic labels
                    fig.add_traces([contour, data_points])
                
                self.valid_dates.append(date)
        
        return fig
    
    def _create_animation_steps(self, fig, progress_bar):
        """Create animation steps for the slider"""
        
        # Create animation steps with paper counts
        steps = []
        for i in range(len(self.valid_dates)):
            # Calculate paper count for this year
            year_data = self.plot_df[self.plot_df['date'] == self.valid_dates[i]]
            paper_count = len(year_data)
            
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {
                        "title": f"Research Papers in {self.valid_dates[i]}",
                        "annotations": [
                            dict(
                                text="Note: Each White Dot Represents a Paper",
                                xref="paper", yref="paper",
                                x=0.5, y=0, 
                                showarrow=False,
                                font=dict(size=10, color="white")
                            ),
                            dict(
                                text=f"Papers visualized: {paper_count:,}",
                                xref="paper", yref="paper",
                                x=0.5, y=-0.08, 
                                showarrow=False,
                                font=dict(size=12, color="white", family="Arial Bold")
                            )
                        ]
                    }
                ],
                label=str(self.valid_dates[i])
            )
            steps.append(step)
        progress_bar.progress(85)

        # Update visibility for each step
        for i, step in enumerate(steps):
            # Determine number of traces per frame based on whether topic clusters are included
            traces_per_frame = 3 if self.topic_clusters is not None else 2
            start_idx = i * traces_per_frame
            end_idx = start_idx + traces_per_frame
            
            for j in range(start_idx, min(end_idx, len(fig.data))):
                step["args"][0]["visible"][j] = True
        
        progress_bar.progress(90)
        return steps
    
    def _configure_layout(self, fig, steps, progress_bar):
        """Configure the final layout of the figure"""
        
        # Configure layout
        fig.update_layout(
            sliders=[dict(
                active=max(0, len(self.valid_dates) - 1),
                currentvalue={"prefix": "Year: "},
                pad={"t": 50},
                steps=steps
            )],
            title="Research Papers by Year - Use Slider to Explore Different Years",
            xaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[self.x_min_global, self.x_max_global]
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[self.y_min_global, self.y_max_global]
            ),
            width=1000,
            height=750,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            modebar=dict(add=['toggleSpikelines']),
            annotations=[
                dict(
                    text="Note: Each White Dot Represents a Paper",
                    xref="paper", yref="paper",
                    x=0.5, y=0, 
                    showarrow=False,
                    font=dict(size=10, color="white")
                )
            ],
            showlegend=False,
        )
        progress_bar.progress(95)

        # Show last frame
        traces_per_frame = 3 if self.topic_clusters is not None else 2
        if len(fig.data) >= traces_per_frame:
            for i in range(-traces_per_frame, 0):
                fig.data[i].visible = True
        
        progress_bar.progress(100)
        return fig
    
    def draw(self):
        """
        Main method to create the complete cumulative visualization.
        
        Returns:
            plotly.graph_objs._figure.Figure: The complete animated visualization
        """
        # Create progress bar
        progress_bar = st.progress(0)
        
        try:
            # Validation
            progress_bar.progress(5)
            
            # Step 1: Calculate global bounds
            self._calculate_global_bounds(progress_bar)
            
            # Step 2: Prepare data
            self._prepare_data(progress_bar)
            
            # Step 3: Generate traces
            fig = self._generate_traces(progress_bar)
            
            # Step 4: Create animation steps
            steps = self._create_animation_steps(fig, progress_bar)
            
            # Step 5: Configure layout
            fig = self._configure_layout(fig, steps, progress_bar)
            
            return fig
            
        finally:
            # Clean up progress indicators after a short delay
            import time
            time.sleep(0.5)
            progress_bar.empty()
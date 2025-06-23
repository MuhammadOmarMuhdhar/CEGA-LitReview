import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from functools import lru_cache
import hashlib
from typing import Dict, List, Optional, Any
import numpy as np


class Sankey:
    """
    High-performance Sankey class with optimized caching and filtering.
    """
    
    def __init__(self, filters_json=None, default_colors=None):
        self.filters_json = filters_json
        self.default_colors = default_colors or px.colors.qualitative.Vivid
        self.mappers = None
        
        # Enhanced caching system
        self._preprocessed_data = None
        self._preprocessed_hash = None
        self._filter_cache = {}
        self._aggregation_cache = {}
        self._node_cache = {}
        
        # Pre-computed data for faster filtering
        self._value_counts_cache = {}
        self._unique_values_cache = {}
        
        self.available_columns = {
            'poverty_context': {
                'hierarchy_key': 'poverty_contexts',
                'display_name': 'Poverty Context',
                'color_offset': 0
            },
            'study_type': {
                'hierarchy_key': 'study_types', 
                'display_name': 'Study Type',
                'color_offset': 5
            },
            'mechanism': {
                'hierarchy_key': 'mechanisms',
                'display_name': 'Mechanism',
                'color_offset': 10
            },
            'behavior': {
                'hierarchy_key': 'Behaviors',
                'display_name': 'Behavior',
                'color_offset': 15
            }
        }
        
        if filters_json:
            self.mappers = self._create_hierarchy_mappers()
    
    def _create_stable_hash(self, *args) -> str:
        """Create stable hash from multiple arguments."""
        content = str(args)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @lru_cache(maxsize=128)
    def _get_hierarchy_mapping(self, col_name: str, detail_level: int):
        """Cached hierarchy mapping for better performance."""
        if not self.mappers or col_name not in self.mappers:
            return None
        
        # Pre-compute all possible mappings for this column and level
        mapper = self.mappers[col_name]
        return lambda x: mapper(x, detail_level)
    
    def _preprocess_dataframe(self, df: pd.DataFrame, columns_to_show: List[str]) -> pd.DataFrame:
        """One-time preprocessing with aggressive caching - FIXED for categoricals"""
        df_info = (df.shape, tuple(df.columns), tuple(columns_to_show))
        current_hash = self._create_stable_hash(df_info)
        
        if self._preprocessed_hash == current_hash and self._preprocessed_data is not None:
            return self._preprocessed_data
        
        # Reset caches when base data changes
        self._filter_cache.clear()
        self._aggregation_cache.clear()
        self._node_cache.clear()
        self._value_counts_cache.clear()
        self._unique_values_cache.clear()
        
        # Efficient preprocessing
        existing_columns = [col for col in columns_to_show if col in df.columns]
        result_df = df[existing_columns].copy()
        
        # Vectorized operations for better performance
        exclude_values = {'Insufficient info'}
        
        # Single pass for all exclusions
        mask = pd.Series(True, index=result_df.index)
        
        for col in existing_columns:
            col_mask = (
                result_df[col].notna() & 
                ~result_df[col].isin(exclude_values)
            )
            mask &= col_mask
            
            # FIXED: Handle categorical columns for value counting
            if col in ['poverty_context', 'mechanism', 'study_type']:
                # Convert categorical to string temporarily for value_counts
                if result_df[col].dtype.name == 'category':
                    col_series = result_df[col].astype(str)
                else:
                    col_series = result_df[col]
                    
                value_counts = col_series.value_counts()
                self._value_counts_cache[col] = value_counts
                
                # Apply the > 2 filter using string comparison
                mask &= col_series.map(value_counts) > 2
            
            # Cache unique values - handle categoricals
            if result_df[col].dtype.name == 'category':
                self._unique_values_cache[col] = result_df[col].cat.categories.tolist()
            else:
                self._unique_values_cache[col] = result_df[col].unique()
        
        result_df = result_df[mask]
        
        # Keep categorical columns as categorical (don't convert back to object)
        # This preserves the memory benefits
        
        self._preprocessed_data = result_df
        self._preprocessed_hash = current_hash
        
        return result_df
    
    def _apply_filters_optimized(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Optimized filtering with better caching."""
        if not filters:
            return df
        
        # Create cache key
        filter_key = self._create_stable_hash(df.shape, tuple(sorted(filters.items())))
        
        if filter_key in self._filter_cache:
            return self._filter_cache[filter_key]
        
        # Apply filters using vectorized operations
        mask = pd.Series(True, index=df.index)
        
        filter_mapping = {
            'contexts': 'poverty_context',
            'study_types': 'study_type',
            'mechanisms': 'mechanism',
            'behaviors': 'behavior'
        }
        
        for filter_name, filter_values in filters.items():
            if filter_values and filter_name in filter_mapping:
                col_name = filter_mapping[filter_name]
                if col_name in df.columns:
                    # Use efficient isin operation
                    mask &= df[col_name].isin(filter_values)
        
        result_df = df[mask]
        
        # Cache result (limit cache size)
        if len(self._filter_cache) > 50:
            # Remove oldest entries
            old_keys = list(self._filter_cache.keys())[:25]
            for key in old_keys:
                del self._filter_cache[key]
        
        self._filter_cache[filter_key] = result_df
        return result_df
    
    def _create_aggregated_data(self, df: pd.DataFrame, detail_levels: Dict[str, int], 
                               columns_to_show: List[str]) -> pd.DataFrame:
        """Optimized aggregation with caching."""
        cache_key = self._create_stable_hash(
            df.shape, tuple(sorted(detail_levels.items())), tuple(columns_to_show)
        )
        
        if cache_key in self._aggregation_cache:
            return self._aggregation_cache[cache_key]
        
        if not self.mappers:
            self._aggregation_cache[cache_key] = df
            return df
        
        # Efficient column mapping
        df_display = df.copy()
        
        for col_name in columns_to_show:
            if col_name in self.mappers and col_name in detail_levels:
                detail_level = detail_levels[col_name]
                mapping_func = self._get_hierarchy_mapping(col_name, detail_level)
                
                if mapping_func:
                    display_col = f'display_{col_name}'
                    # Vectorized application
                    df_display[display_col] = df_display[col_name].apply(mapping_func)
        
        # Cache with size limit
        if len(self._aggregation_cache) > 20:
            old_keys = list(self._aggregation_cache.keys())[:10]
            for key in old_keys:
                del self._aggregation_cache[key]
        
        self._aggregation_cache[cache_key] = df_display
        return df_display
    
    def _create_links_optimized(self, df_display: pd.DataFrame, node_indices: Dict, 
                               node_colors: List, column_mappings: Dict, 
                               columns_to_show: List[str]) -> Dict:
        """Optimized link creation using vectorized operations."""
        links = {'source': [], 'target': [], 'value': [], 'color': []}
        
        # Process all column pairs efficiently
        for i in range(len(columns_to_show) - 1):
            source_col = columns_to_show[i]
            target_col = columns_to_show[i + 1]
            
            source_col_name = column_mappings[source_col]
            target_col_name = column_mappings[target_col]
            
            if source_col_name not in df_display.columns or target_col_name not in df_display.columns:
                continue
            
            # Efficient groupby with size
            grouped = df_display.groupby([source_col_name, target_col_name], observed=True).size()
            
            # Vectorized processing of groups
            for (source_val, target_val), count in grouped.items():
                if (source_val in node_indices[source_col] and 
                    target_val in node_indices[target_col]):
                    
                    source_idx = node_indices[source_col][source_val]
                    target_idx = node_indices[target_col][target_val]
                    
                    links['source'].append(source_idx)
                    links['target'].append(target_idx)
                    links['value'].append(int(count))  # Ensure int for JSON serialization
                    links['color'].append(self._add_transparency(node_colors[source_idx]))
        
        return links
    
    def _determine_detail_levels(self, active_filters, columns_to_show):
        """Cached detail level determination."""
        cache_key = self._create_stable_hash(active_filters, columns_to_show)
        
        if hasattr(self, '_detail_levels_cache'):
            if cache_key in self._detail_levels_cache:
                return self._detail_levels_cache[cache_key]
        else:
            self._detail_levels_cache = {}
        
        detail_levels = {}
        max_depths = {}
        
        if self.filters_json:
            for col_name in columns_to_show:
                if col_name in self.available_columns:
                    hierarchy_key = self.available_columns[col_name]['hierarchy_key']
                    max_depths[col_name] = self._get_max_depth(self.filters_json.get(hierarchy_key, {}))
        else:
            for col_name in columns_to_show:
                max_depths[col_name] = 2
        
        base_level = 1
        filter_key_to_column = {
            'contexts': 'poverty_context',
            'study_types': 'study_type', 
            'mechanisms': 'mechanism',
            'behaviors': 'behavior'
        }
        
        for col_name in columns_to_show:
            filter_key = None
            for fk, cn in filter_key_to_column.items():
                if cn == col_name:
                    filter_key = fk
                    break
            
            if filter_key and active_filters.get(filter_key):
                detail_levels[col_name] = max_depths.get(col_name, base_level)
            else:
                detail_levels[col_name] = base_level
        
        self._detail_levels_cache[cache_key] = detail_levels
        return detail_levels
    
    # Keep existing helper methods with minimal changes
    def _find_item_path(self, item, data, path=[]):
        """Recursively find the full path to an item in nested data structure."""
        for key, value in data.items():
            current_path = path + [key]
            if isinstance(value, dict):
                result = self._find_item_path(item, value, current_path)
                if result:
                    return result
            elif isinstance(value, list):
                if item in value:
                    return current_path + [item]
        return None
    
    def _create_hierarchy_mappers(self):
        """Create mapping functions to convert specific categories to broader ones."""
        if not self.filters_json:
            return None
        
        mappers = {}
        
        for col_name, col_info in self.available_columns.items():
            hierarchy_key = col_info['hierarchy_key']
            
            def create_mapper(hierarchy_key):
                def mapper(specific_value, target_level=1):
                    if hierarchy_key not in self.filters_json:
                        return specific_value
                    
                    path = self._find_item_path(specific_value, self.filters_json[hierarchy_key])
                    if path and len(path) >= target_level:
                        return path[target_level - 1]
                    elif path:
                        return path[-1]
                    return specific_value
                return mapper
            
            mappers[col_name] = create_mapper(hierarchy_key)
        
        return mappers
    
    def _get_max_depth(self, data, current_depth=1):
        """Helper function to find maximum depth for a category."""
        max_depth = current_depth
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, dict):
                    max_depth = max(max_depth, self._get_max_depth(value, current_depth + 1))
                elif isinstance(value, list):
                    max_depth = max(max_depth, current_depth + 1)
        return max_depth
    
    def _add_transparency(self, color, alpha=0.4):
        """Add transparency to a color."""
        if color.startswith('#'):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            return f'rgba({r},{g},{b},{alpha})'
        elif color.startswith('rgb'):
            return color.replace('rgb', 'rgba').rstrip(')') + f',{alpha})'
        return f'rgba(128,128,128,{alpha})'
    
    def _create_node_structure(self, df_display, columns_to_show, use_display_columns=True):
        """Optimized node structure creation with caching - FIXED for categoricals"""
        cache_key = self._create_stable_hash(
            df_display.shape, tuple(columns_to_show), use_display_columns
        )
        
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]
        
        column_mappings = {}
        for col_name in columns_to_show:
            if use_display_columns and f'display_{col_name}' in df_display.columns:
                column_mappings[col_name] = f'display_{col_name}'
            else:
                column_mappings[col_name] = col_name
        
        # Efficient unique value extraction - FIXED for categoricals
        categories = {}
        for col_name in columns_to_show:
            mapped_col = column_mappings[col_name]
            if mapped_col in df_display.columns:
                # FIXED: Handle categorical columns properly
                if df_display[mapped_col].dtype.name == 'category':
                    # For categorical, get unique values from the actual data, not all categories
                    categories[col_name] = df_display[mapped_col].dropna().unique().tolist()
                else:
                    categories[col_name] = df_display[mapped_col].unique().tolist()
            else:
                categories[col_name] = []
        
        # Create node structure efficiently
        node_labels = []
        node_indices = {}
        current_index = 0
        
        for col_name in columns_to_show:
            node_indices[col_name] = {}
            for label in categories[col_name]:
                node_indices[col_name][label] = current_index
                node_labels.append(label)
                current_index += 1
        
        # Efficient color assignment
        vivid = self.default_colors
        node_colors = []
        
        for col_name in columns_to_show:
            col_info = self.available_columns.get(col_name, {})
            
            if col_info.get('color_type') == 'fixed':
                node_colors.extend([col_info['color']] * len(categories[col_name]))
            else:
                color_offset = col_info.get('color_offset', 0)
                for i in range(len(categories[col_name])):
                    color_idx = (i + color_offset) % len(vivid)
                    node_colors.append(vivid[color_idx])
        
        result = (categories, node_labels, node_indices, node_colors, column_mappings)
        
        # Cache with size limit
        if len(self._node_cache) > 30:
            old_keys = list(self._node_cache.keys())[:15]
            for key in old_keys:
                del self._node_cache[key]
        
        self._node_cache[cache_key] = result
        return result
    
    def _create_figure(self, node_labels, node_colors, links, columns_to_show):
        """Create the Plotly Sankey figure."""
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=2),
                label=node_labels,
                color=node_colors,
                hovertemplate="<b>%{label}</b><br>" +
                             "Connections: %{value}<br>" +
                             "<extra></extra>"
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=links['color'],
                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>" +
                             "Count: <b>%{value}</b><br>" +
                             "<extra></extra>"
            )
        )])
        
        # Create annotations for column headers
        annotations = []
        num_columns = len(columns_to_show)
        if num_columns > 1:
            for i, col_name in enumerate(columns_to_show):
                display_name = self.available_columns.get(col_name, {}).get('display_name', col_name.replace('_', ' ').title())
                x_pos = i / (num_columns - 1) if num_columns > 1 else 0.5
                annotations.append(
                    dict(x=x_pos, y=1.1, text=display_name, showarrow=False, 
                         xref="paper", yref="paper",
                         font=dict(size=16, color="black", weight="bold"))
                )

        annotations.append(
            dict(x=0.5, y=-0.15, 
                 text="Note: Papers with multiple contexts or mechanisms appear separately for each category",
                 showarrow=False, xref="paper", yref="paper",
                 font=dict(size=12, color="gray", style="italic"))
        )
        
        fig.update_layout(
            font_size=12,
            height=600,
            width=1200,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font=dict(color="black", size=14, family="Arial, sans-serif")
            ),
            annotations=annotations,
            plot_bgcolor='rgba(248, 248, 248, 1)',
            paper_bgcolor='white'
        )
        
        fig.update_traces(
            textfont=dict(color='black', size=14, family="Arial Black"),
            selector=dict(type='sankey')
        )
        
        return fig
    
    def draw(self, df: pd.DataFrame, columns_to_show: Optional[List[str]] = None, 
         active_filters: Optional[Dict[str, Any]] = None, manual_detail_level: Optional[int] = None):
        """
        Optimized draw method with improved caching and performance.
        """
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Set defaults
            status_text.text(" ")
            progress_bar.progress(10)
            
            if columns_to_show is None:
                columns_to_show = ['poverty_context', 'study_type', 'mechanism', 'behavior']
            
            active_filters = active_filters or {}
            
            # Validate inputs
            if len(columns_to_show) < 2:
                raise ValueError("At least 2 columns are required to create a Sankey diagram")
            
            missing_columns = [col for col in columns_to_show if col not in df.columns]
            if missing_columns:
                raise ValueError(f"The following columns are missing from the dataframe: {missing_columns}")
            
            # Step 1: Preprocess data
            status_text.text(" ")
            progress_bar.progress(25)
            df_clean = self._preprocess_dataframe(df, columns_to_show)
            
            # Step 2: Apply filters
            status_text.text(" ")
            progress_bar.progress(40)
            df_filtered = self._apply_filters_optimized(df_clean, active_filters)
            
            # Early return for empty data
            if df_filtered.empty:
                status_text.text("No data found matching filters")
                progress_bar.progress(100)
                return go.Figure().add_annotation(
                    text="No data matches the current filters",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="gray")
                )
            
            # Handle case without hierarchy
            if self.filters_json is None:
                status_text.text(" ")
                progress_bar.progress(80)
                result = self._draw_without_hierarchy(df_filtered, columns_to_show)
                progress_bar.progress(100)
                return result
            
            # Step 3: Determine detail levels
            status_text.text(" ")
            progress_bar.progress(55)
            if manual_detail_level:
                detail_levels = {col: manual_detail_level for col in columns_to_show}
            else:
                detail_levels = self._determine_detail_levels(active_filters, columns_to_show)
            
            # Step 4: Create aggregated data
            status_text.text(" ")
            progress_bar.progress(65)
            df_display = self._create_aggregated_data(df_filtered, detail_levels, columns_to_show)
            
            # Step 5: Create visualization components
            status_text.text(" ")
            progress_bar.progress(75)
            categories, node_labels, node_indices, node_colors, column_mappings = self._create_node_structure(
                df_display, columns_to_show, use_display_columns=True)
            
            # Step 6: Create links
            status_text.text(" ")
            progress_bar.progress(85)
            links = self._create_links_optimized(df_display, node_indices, node_colors, column_mappings, columns_to_show)
            
            # Step 7: Create figure
            status_text.text(" ")
            progress_bar.progress(95)
            result = self._create_figure(node_labels, node_colors, links, columns_to_show)
            
            status_text.text(" ")
            progress_bar.progress(100)
            return result
            
        finally:
            # Clean up progress indicators after a short delay
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    def _draw_without_hierarchy(self, df, columns_to_show):
        """Fallback method for drawing without hierarchy information."""
        categories, node_labels, node_indices, node_colors, column_mappings = self._create_node_structure(
            df, columns_to_show, use_display_columns=False)
        
        links = self._create_links_optimized(df, node_indices, node_colors, column_mappings, columns_to_show)
        
        return self._create_figure(node_labels, node_colors, links, columns_to_show)
    
    def clear_cache(self):
        """Clear all caches for memory management."""
        self._filter_cache.clear()
        self._aggregation_cache.clear()
        self._node_cache.clear()
        self._value_counts_cache.clear()
        self._unique_values_cache.clear()
        if hasattr(self, '_detail_levels_cache'):
            self._detail_levels_cache.clear()
        
        # Clear LRU cache
        self._get_hierarchy_mapping.cache_clear()
    
    def get_cache_stats(self):
        """Get cache statistics for debugging."""
        return {
            'filter_cache_size': len(self._filter_cache),
            'aggregation_cache_size': len(self._aggregation_cache),
            'node_cache_size': len(self._node_cache),
            'hierarchy_cache_info': self._get_hierarchy_mapping.cache_info()
        }
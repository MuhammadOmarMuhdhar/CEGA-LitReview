import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class Sankey:
    """
    A class for creating hierarchical Sankey diagrams with dynamic aggregation levels and modular node selection.
    
    This class creates Sankey diagrams that can display data at different levels of detail
    based on active filters and hierarchy information, with full control over which nodes to display.
    """
    
    def __init__(self, filters_json=None, default_colors=None):
        """
        Initialize the SankeyDiagram.
        
        Parameters:
        - filters_json: Dictionary containing hierarchy information for categories
        - default_colors: Custom color palette (defaults to Plotly Vivid)
        """
        self.filters_json = filters_json
        self.default_colors = default_colors or px.colors.qualitative.Vivid
        self.mappers = None
        
        # Define all available columns and their hierarchy keys
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
    
    def _find_item_path(self, item, data, path=[]):
        """Recursively find the full path to an item in nested data structure."""
        for key, value in data.items():
            current_path = path + [key]
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                result = self._find_item_path(item, value, current_path)
                if result:
                    return result
            elif isinstance(value, list):
                # Check if item is in this list
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
                    """Map value to broader category with variable depth handling."""
                    if hierarchy_key not in self.filters_json:
                        return specific_value
                    
                    path = self._find_item_path(specific_value, self.filters_json[hierarchy_key])
                    if path and len(path) >= target_level:
                        return path[target_level - 1]
                    elif path:
                        # If target level is deeper than available, return the deepest available
                        return path[-1]
                    return specific_value  # fallback if not found
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
    
    def _determine_detail_levels(self, active_filters, columns_to_show):
        """Determine appropriate detail level for each category based on active filters."""
        detail_levels = {}
        
        # Get maximum depths for each category that will be shown
        max_depths = {}
        if self.filters_json:
            for col_name in columns_to_show:
                if col_name in self.available_columns:
                    hierarchy_key = self.available_columns[col_name]['hierarchy_key']
                    max_depths[col_name] = self._get_max_depth(self.filters_json.get(hierarchy_key, {}))
        else:
            # Default max depths if no filter info available
            for col_name in columns_to_show:
                max_depths[col_name] = 2
        
        # Default to broadest level
        base_level = 1
        
        # Map active_filters keys to column names
        filter_key_to_column = {
            'contexts': 'poverty_context',
            'study_types': 'study_type', 
            'mechanisms': 'mechanism',
            'behaviors': 'behavior'
        }
        
        # If any filters are active, show the deepest level for ONLY those specific categories
        for col_name in columns_to_show:
            # Find the corresponding filter key
            filter_key = None
            for fk, cn in filter_key_to_column.items():
                if cn == col_name:
                    filter_key = fk
                    break
            
            if filter_key and active_filters.get(filter_key):
                detail_levels[col_name] = max_depths.get(col_name, base_level)
            else:
                detail_levels[col_name] = base_level
        
        return detail_levels
    
    def _aggregate_dataframe(self, df, detail_levels, columns_to_show):
        """Create display columns with appropriate aggregation level."""
        if not self.mappers:
            return df
            
        df_display = df.copy()
        
        # Apply mapping for each column that will be shown
        for col_name in columns_to_show:
            if col_name in self.mappers and col_name in detail_levels:
                display_col = f'display_{col_name}'
                df_display[display_col] = df_display[col_name].apply(
                    lambda x: self.mappers[col_name](x, detail_levels[col_name])
                )
        
        return df_display
    
    def _preprocess_dataframe(self, df, columns_to_show):
        """Clean and filter the dataframe before processing."""
        # Only apply preprocessing to columns that will be shown and exist in the dataframe
        existing_columns = [col for col in columns_to_show if col in df.columns]
        
        # Filter categories with counts > 2 (only for specific columns)
        filter_count_columns = ['poverty_context', 'mechanism', 'study_type']
        for col in filter_count_columns:
            if col in existing_columns:
                df = df[df[col].map(df[col].value_counts()) > 2]

        # Drop all null values and 'Insufficient info' values
        exclude_values = ['Insufficient info']
        
        for col in existing_columns:
            df = df[df[col].notnull()]
            df = df[~df[col].isin(exclude_values)]
        
        return df
    
    def _add_transparency(self, color, alpha=0.4):
        """Add transparency to a color."""
        if color.startswith('#'):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            return f'rgba({r},{g},{b},{alpha})'
        elif color.startswith('rgb'):
            return color.replace('rgb', 'rgba').rstrip(')') + f',{alpha})'
        return f'rgba(128,128,128,{alpha})'
    
    def _create_node_structure(self, df_display, columns_to_show, use_display_columns=True):
        """Create node labels, indices, and colors for specified columns only."""
        # Determine which columns to use
        column_mappings = {}
        for col_name in columns_to_show:
            if use_display_columns and f'display_{col_name}' in df_display.columns:
                column_mappings[col_name] = f'display_{col_name}'
            else:
                column_mappings[col_name] = col_name
        
        # Get unique values for each category that will be shown
        categories = {}
        for col_name in columns_to_show:
            if column_mappings[col_name] in df_display.columns:
                categories[col_name] = df_display[column_mappings[col_name]].unique().tolist()
            else:
                categories[col_name] = []
        
        # Create node labels and indices
        node_labels = []
        node_indices = {}
        current_index = 0
        
        for col_name in columns_to_show:
            node_indices[col_name] = {}
            for label in categories[col_name]:
                node_indices[col_name][label] = current_index
                node_labels.append(label)
                current_index += 1
        
        # Assign colors using the color palette
        vivid = self.default_colors
        node_colors = []
        
        for col_name in columns_to_show:
            col_info = self.available_columns.get(col_name, {})
            
            if col_info.get('color_type') == 'fixed':
                # Fixed color (like study_type)
                node_colors.extend([col_info['color']] * len(categories[col_name]))
            else:
                # Use color palette with offset
                color_offset = col_info.get('color_offset', 0)
                for i in range(len(categories[col_name])):
                    color_idx = (i + color_offset) % len(vivid)
                    node_colors.append(vivid[color_idx])
        
        return categories, node_labels, node_indices, node_colors, column_mappings
    
    def _create_links(self, df_display, node_indices, node_colors, column_mappings, columns_to_show):
        """Create links between consecutive nodes in the specified column order."""
        links = {'source': [], 'target': [], 'value': [], 'color': []}
        
        # Create links between consecutive columns
        for i in range(len(columns_to_show) - 1):
            source_col = columns_to_show[i]
            target_col = columns_to_show[i + 1]
            
            source_col_name = column_mappings[source_col]
            target_col_name = column_mappings[target_col]
            
            # Skip if columns don't exist in the dataframe
            if source_col_name not in df_display.columns or target_col_name not in df_display.columns:
                continue
            
            # Group by source and target columns to get counts
            for (source_val, target_val), count in df_display.groupby([source_col_name, target_col_name]).size().items():
                if (source_val in node_indices[source_col] and 
                    target_val in node_indices[target_col]):
                    
                    source_idx = node_indices[source_col][source_val]
                    target_idx = node_indices[target_col][target_val]
                    
                    links['source'].append(source_idx)
                    links['target'].append(target_idx)
                    links['value'].append(count)
                    
                    # Use source node color for the link
                    links['color'].append(self._add_transparency(node_colors[source_idx]))
        
        return links
    
    
    
    def _create_figure(self, node_labels, node_colors, links, columns_to_show):
        """Create the Plotly Sankey figure."""
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=2),  # Darker, thicker border for better contrast
                label=node_labels,
                color=node_colors,
                # Enhanced hover template for nodes
                hovertemplate="<b>%{label}</b><br>" +
                             "Connections: %{value}<br>" +
                             "<extra></extra>"
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=links['color'],
                # Enhanced hover template for links
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
        
        # Update layout with enhanced styling
        fig.update_layout(
            font_size=12,
            height=600,
            width=1200,
            # Enhanced hover mode and styling
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font=dict(color="black", size=14, family="Arial, sans-serif")
            ),
            annotations=annotations,
            # Enhanced plot background for better contrast
            plot_bgcolor='rgba(248, 248, 248, 1)',
            paper_bgcolor='white'
        )
        
        # Update traces with enhanced text styling
        fig.update_traces(
            textfont=dict(color='black', size=14, family="Arial Black"),
            selector=dict(type='sankey')
        )
        
        return fig
    
    def draw(self, df, columns_to_show=None, active_filters=None, manual_detail_level=None):
        """
        Draw sankey diagram with enhanced hover contrast and modular column selection.
        
        Parameters:
        - df: DataFrame with required columns
        - columns_to_show: List of column names to display in order (e.g., ['study_type', 'mechanism'])
                          If None, defaults to ['poverty_context', 'study_type', 'mechanism', 'behavior']
        - active_filters: Current filter state to determine detail levels
        - manual_detail_level: Optional override for detail level (1-3)
        
        Returns:
        - Plotly figure object
        """
        # Set default columns if not specified
        if columns_to_show is None:
            columns_to_show = ['poverty_context', 'study_type', 'mechanism', 'behavior']
        
        # Validate that we have at least 2 columns for a flow diagram
        if len(columns_to_show) < 2:
            raise ValueError("At least 2 columns are required to create a Sankey diagram")
        
        # Validate that all specified columns exist in the dataframe
        missing_columns = [col for col in columns_to_show if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing from the dataframe: {missing_columns}")
        
        # Preprocess the dataframe
        df_clean = self._preprocess_dataframe(df.copy(), columns_to_show)
        
        # If no hierarchy info provided, use original columns
        if self.filters_json is None or active_filters is None:
            return self._draw_without_hierarchy(df_clean, columns_to_show)
        
        # Determine detail levels
        if manual_detail_level:
            detail_levels = {col: manual_detail_level for col in columns_to_show}
        else:
            detail_levels = self._determine_detail_levels(active_filters, columns_to_show)
        
        # Aggregate dataframe to appropriate display level
        df_display = self._aggregate_dataframe(df_clean, detail_levels, columns_to_show)
        
        # Create node structure
        categories, node_labels, node_indices, node_colors, column_mappings = self._create_node_structure(
            df_display, columns_to_show, use_display_columns=True)
        
        # Create links
        links = self._create_links(df_display, node_indices, node_colors, column_mappings, columns_to_show)
        
        # Create and return figure
        return self._create_figure(node_labels, node_colors, links, columns_to_show)
    
    def _draw_without_hierarchy(self, df):
        """Fallback method for drawing without hierarchy information."""
        # Create node structure using original columns
        categories, node_labels, node_indices, node_colors, columns = self._create_node_structure(df, use_display_columns=False)
        
        # Create links
        links = self._create_links(df, node_indices, node_colors, columns)
        
        # Create and return figure
        return self._create_figure(node_labels, node_colors, links)
    
    def set_filters_json(self, filters_json):
        """Update the filters_json and recreate mappers."""
        self.filters_json = filters_json
        if filters_json:
            self.mappers = self._create_hierarchy_mappers()
        else:
            self.mappers = None
    
    def set_colors(self, colors):
        """Update the color palette."""
        self.default_colors = colors


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class Sankey:
    """
    A class for creating hierarchical Sankey diagrams with dynamic aggregation levels.
    
    This class creates Sankey diagrams that can display data at different levels of detail
    based on active filters and hierarchy information.
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
            
        def map_study_type(specific_value, target_level=1):
            """Map study type to broader category with variable depth handling."""
            path = self._find_item_path(specific_value, self.filters_json['study_types'])
            if path and len(path) >= target_level:
                return path[target_level - 1]
            elif path:
                # If target level is deeper than available, return the deepest available
                return path[-1]
            return specific_value  # fallback if not found
        
        def map_mechanism(specific_value, target_level=1):
            """Map mechanism to broader category with variable depth handling."""
            path = self._find_item_path(specific_value, self.filters_json['mechanisms'])
            if path and len(path) >= target_level:
                return path[target_level - 1]
            elif path:
                # If target level is deeper than available, return the deepest available
                return path[-1]
            return specific_value  # fallback if not found
        
        def map_behavior(specific_value, target_level=1):
            """Map behavior to broader category with variable depth handling."""
            path = self._find_item_path(specific_value, self.filters_json['Behaviors'])
            if path and len(path) >= target_level:
                return path[target_level - 1]
            elif path:
                # If target level is deeper than available, return the deepest available
                return path[-1]
            return specific_value  # fallback if not found
        
        def map_poverty_context(specific_value, target_level=1):
            """Map poverty context with variable depth handling."""
            # Handle both direct lists and nested structures
            if isinstance(self.filters_json['poverty_contexts'], dict):
                path = self._find_item_path(specific_value, self.filters_json['poverty_contexts'])
                if path and len(path) >= target_level:
                    return path[target_level - 1]
                elif path:
                    return path[-1]
            return specific_value  # fallback if not found or if already at top level
        
        return {
            'study_type': map_study_type,
            'mechanism': map_mechanism,
            'behavior': map_behavior,
            'poverty_context': map_poverty_context
        }
    
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
    
    def _determine_detail_levels(self, active_filters):
        """Determine appropriate detail level for each category based on active filters."""
        detail_levels = {}
        
        # Get maximum depths for each category
        max_depths = {}
        if self.filters_json:
            max_depths['poverty_context'] = self._get_max_depth(self.filters_json.get('poverty_contexts', {}))
            max_depths['study_type'] = self._get_max_depth(self.filters_json.get('study_types', {}))
            max_depths['mechanism'] = self._get_max_depth(self.filters_json.get('mechanisms', {}))
            max_depths['behavior'] = self._get_max_depth(self.filters_json.get('Behaviors', {}))
        else:
            # Default max depths if no filter info available
            max_depths = {'poverty_context': 2, 'study_type': 3, 'mechanism': 2, 'behavior': 2}
        
        # Default to broadest level
        base_level = 1
        
        # If any filters are active, show the deepest level for ONLY those specific categories
        detail_levels['poverty_context'] = max_depths['poverty_context'] if active_filters.get('contexts') else base_level
        detail_levels['study_type'] = max_depths['study_type'] if active_filters.get('study_types') else base_level
        detail_levels['mechanism'] = max_depths['mechanism'] if active_filters.get('mechanisms') else base_level
        detail_levels['behavior'] = max_depths['behavior'] if active_filters.get('behaviors') else base_level
        
        return detail_levels
    
    def _aggregate_dataframe(self, df, detail_levels):
        """Create display columns with appropriate aggregation level."""
        if not self.mappers:
            return df
            
        df_display = df.copy()
        
        # Apply mapping for each category
        df_display['display_poverty_context'] = df_display['poverty_context'].apply(
            lambda x: self.mappers['poverty_context'](x, detail_levels['poverty_context'])
        )
        
        df_display['display_study_type'] = df_display['study_type'].apply(
            lambda x: self.mappers['study_type'](x, detail_levels['study_type'])
        )
        
        df_display['display_mechanism'] = df_display['mechanism'].apply(
            lambda x: self.mappers['mechanism'](x, detail_levels['mechanism'])
        )
        
        df_display['display_behavior'] = df_display['behavior'].apply(
            lambda x: self.mappers['behavior'](x, detail_levels['behavior'])
        )
        
        return df_display
    
    def _preprocess_dataframe(self, df):
        """Clean and filter the dataframe before processing."""
        # Filter categories with counts > 2
        for col in ['poverty_context', 'mechanism', 'study_type']:
            df = df[df[col].map(df[col].value_counts()) > 2]

        # Drop all null values and 'Insufficient info' values
        exclude_values = ['Insufficient info']
        columns_to_clean = ['poverty_context', 'mechanism', 'study_type', 'behavior']
        
        for col in columns_to_clean:
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
    
    def _create_node_structure(self, df_display, use_display_columns=True):
        """Create node labels, indices, and colors."""
        # Determine which columns to use
        if use_display_columns and 'display_poverty_context' in df_display.columns:
            context_col = 'display_poverty_context'
            study_col = 'display_study_type'
            mechanism_col = 'display_mechanism'
            behavior_col = 'display_behavior'
        else:
            context_col = 'poverty_context'
            study_col = 'study_type'
            mechanism_col = 'mechanism'
            behavior_col = 'behavior'
        
        # Get unique values for each category
        categories = {
            'context': df_display[context_col].unique().tolist(),
            'study': df_display[study_col].unique().tolist(),
            'mechanism': df_display[mechanism_col].unique().tolist(),
            'behavior': df_display[behavior_col].unique().tolist()
        }
        
        # Create node labels and indices
        node_labels = categories['context'] + categories['study'] + categories['mechanism'] + categories['behavior']
        node_indices = {
            'context': {label: i for i, label in enumerate(categories['context'])},
            'study': {label: i + len(categories['context']) for i, label in enumerate(categories['study'])},
            'mechanism': {label: i + len(categories['context']) + len(categories['study']) 
                        for i, label in enumerate(categories['mechanism'])},
            'behavior': {label: i + len(categories['context']) + len(categories['study']) + len(categories['mechanism'])
                        for i, label in enumerate(categories['behavior'])}
        }
        
        # Assign colors using the color palette
        vivid = self.default_colors
        node_colors = (
            [vivid[i % len(vivid)] for i in range(len(categories['context']))] +  # Context colors
            ['#689F38'] * len(categories['study']) +  # Study type (green)
            [vivid[(i + 5) % len(vivid)] for i in range(len(categories['mechanism']))] +  # Mechanism colors
            [vivid[(i + 10) % len(vivid)] for i in range(len(categories['behavior']))]  # Behavior colors
        )
        
        return categories, node_labels, node_indices, node_colors, (context_col, study_col, mechanism_col, behavior_col)
    
    def _create_links(self, df_display, node_indices, node_colors, columns):
        """Create links between nodes."""
        context_col, study_col, mechanism_col, behavior_col = columns
        links = {'source': [], 'target': [], 'value': [], 'color': []}
        
        # Context to study type
        for (context, study), count in df_display.groupby([context_col, study_col]).size().items():
            if context in node_indices['context'] and study in node_indices['study']:
                links['source'].append(node_indices['context'][context])
                links['target'].append(node_indices['study'][study])
                links['value'].append(count)
                links['color'].append(self._add_transparency(node_colors[node_indices['context'][context]]))
        
        # Study type to mechanism
        for (study, mechanism), count in df_display.groupby([study_col, mechanism_col]).size().items():
            if study in node_indices['study'] and mechanism in node_indices['mechanism']:
                links['source'].append(node_indices['study'][study])
                links['target'].append(node_indices['mechanism'][mechanism])
                links['value'].append(count)
                links['color'].append(self._add_transparency(node_colors[node_indices['mechanism'][mechanism]]))
        
        # Mechanism to behavior
        for (mechanism, behavior), count in df_display.groupby([mechanism_col, behavior_col]).size().items():
            if mechanism in node_indices['mechanism'] and behavior in node_indices['behavior']:
                links['source'].append(node_indices['mechanism'][mechanism])
                links['target'].append(node_indices['behavior'][behavior])
                links['value'].append(count)
                links['color'].append(self._add_transparency(node_colors[node_indices['behavior'][behavior]]))
        
        return links
    
    def _create_figure(self, node_labels, node_colors, links):
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
            annotations=[
                dict(x=0, y=1.1, text="Poverty Context", showarrow=False, xref="paper", yref="paper",
                    font=dict(size=16, color="black", weight="bold")),
                dict(x=0.33, y=1.1, text="Study Type", showarrow=False, xref="paper", yref="paper",
                    font=dict(size=16, color="black", weight="bold")),
                dict(x=0.66, y=1.1, text="Mechanism", showarrow=False, xref="paper", yref="paper",
                    font=dict(size=16, color="black", weight="bold")),
                dict(x=1, y=1.1, text="Behavior", showarrow=False, xref="paper", yref="paper",
                    font=dict(size=16, color="black", weight="bold"))
            ],
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
    
    def draw(self, df, active_filters=None, manual_detail_level=None):
        """
        Draw sankey diagram with enhanced hover contrast.
        
        Parameters:
        - df: DataFrame with required columns (poverty_context, study_type, mechanism, behavior)
        - active_filters: Current filter state to determine detail levels
        - manual_detail_level: Optional override for detail level (1-3)
        
        Returns:
        - Plotly figure object
        """
        # Preprocess the dataframe
        df_clean = self._preprocess_dataframe(df.copy())
        
        # If no hierarchy info provided, use original columns
        if self.filters_json is None or active_filters is None:
            return self._draw_without_hierarchy(df_clean)
        
        # Determine detail levels
        if manual_detail_level:
            detail_levels = {
                'poverty_context': manual_detail_level,
                'study_type': manual_detail_level,
                'mechanism': manual_detail_level,
                'behavior': manual_detail_level
            }
        else:
            detail_levels = self._determine_detail_levels(active_filters)
        
        # Aggregate dataframe to appropriate display level
        df_display = self._aggregate_dataframe(df_clean, detail_levels)
        
        # Create node structure
        categories, node_labels, node_indices, node_colors, columns = self._create_node_structure(df_display, use_display_columns=True)
        
        # Create links
        links = self._create_links(df_display, node_indices, node_colors, columns)
        
        # Create and return figure
        return self._create_figure(node_labels, node_colors, links)
    
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


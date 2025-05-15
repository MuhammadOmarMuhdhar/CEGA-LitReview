import plotly.express as px
import plotly.graph_objects as go
import pandas as pd



def draw(df):
            # Filter categories with counts > 10
            for col in ['poverty_context', 'mechanism', 'study_type']:
                df = df[df[col].map(df[col].value_counts()) > 2]

            # drop all null values and 'Insufficient info' values
            df = df[df['poverty_context'].notnull()]
            df = df[df['mechanism'].notnull()]
            df = df[df['study_type'].notnull()]
            df = df[df['poverty_context'] != 'Insufficient info']
            df = df[df['mechanism'] != 'Insufficient info']
            df = df[df['study_type'] != 'Insufficient info']
            
            # Get unique values for each category
            categories = {
                'context': df['poverty_context'].unique().tolist(),
                'study': df['study_type'].unique().tolist(),
                'mechanism': df['mechanism'].unique().tolist()
            }
            
            # Create node labels and indices
            node_labels = categories['context'] + categories['study'] + categories['mechanism']
            node_indices = {
                'context': {label: i for i, label in enumerate(categories['context'])},
                'study': {label: i + len(categories['context']) for i, label in enumerate(categories['study'])},
                'mechanism': {label: i + len(categories['context']) + len(categories['study']) 
                            for i, label in enumerate(categories['mechanism'])}
            }
            
            # Assign colors using Plotly's Vivid palette
            vivid = px.colors.qualitative.Vivid
            node_colors = (
                [vivid[i % len(vivid)] for i in range(len(categories['context']))] +  # Context colors
                ['#689F38'] * len(categories['study']) +  # Study type (green)
                [vivid[(i + 5) % len(vivid)] for i in range(len(categories['mechanism']))]  # Mechanism colors (offset)
            )
            
            # Create links
            links = {'source': [], 'target': [], 'value': [], 'color': []}
            
            # Helper to add transparency
            def add_transparency(color, alpha=0.4):
                if color.startswith('#'):
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    return f'rgba({r},{g},{b},{alpha})'
                elif color.startswith('rgb'):
                    return color.replace('rgb', 'rgba').rstrip(')') + f',{alpha})'
                return f'rgba(128,128,128,{alpha})'
            
            # Add links from context to study type
            for (context, study), count in df.groupby(['poverty_context', 'study_type']).size().items():
                links['source'].append(node_indices['context'][context])
                links['target'].append(node_indices['study'][study])
                links['value'].append(count)
                links['color'].append(add_transparency(node_colors[node_indices['context'][context]]))
            
            # Add links from study type to mechanism
            for (study, mechanism), count in df.groupby(['study_type', 'mechanism']).size().items():
                links['source'].append(node_indices['study'][study])
                links['target'].append(node_indices['mechanism'][mechanism])
                links['value'].append(count)
                links['color'].append(add_transparency(node_colors[node_indices['mechanism'][mechanism]]))
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="white", width=1),
                    label=node_labels,
                    color=node_colors
                ),
                link=dict(
                    source=links['source'],
                    target=links['target'],
                    value=links['value'],
                    color=links['color']
                )
            )])
            
            # Update layout
            fig.update_layout(
                title_text="",
                font_size=12,
                height=600,
                width=1000,
                annotations=[
                    dict(x=0, y=1.1, text="Poverty Context", showarrow=False, xref="paper", yref="paper",
                        font=dict(size=16, color="black", weight="bold")),
                    dict(x=0.5, y=1.1, text="Study Type", showarrow=False, xref="paper", yref="paper",
                        font=dict(size=16, color="black", weight="bold")),
                    dict(x=1, y=1.1, text="Mechanism", showarrow=False, xref="paper", yref="paper",
                        font=dict(size=16, color="black", weight="bold"))
                ]
            )
            
            fig.update_traces(textfont=dict(color='black', size=17))
            
            return fig
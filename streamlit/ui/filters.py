import streamlit as st
from streamlit_tree_select import tree_select
import json
import hashlib


@st.cache_data(ttl=300, max_entries=1)
def load_filters_json():
    """Load filters JSON"""
    with open('data/trainingData/labels.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_and_process_filters():
    """Cache the filter loading and tree building"""
    filters = load_filters_json()

    study_types_tree = build_tree_optimized(filters['study_types'])
    mechanisms_tree = build_tree_optimized(filters['mechanisms'])
    behaviors_tree = build_tree_optimized(filters['Behaviors'])

    return filters, study_types_tree, mechanisms_tree, behaviors_tree

def build_tree_optimized(data, path=""):
    """Optimized tree builder"""
    if not data:
        return []

    tree = []
    for key, value in data.items():
        node_path = f"{path} > {key}" if path else key

        if isinstance(value, dict):
            children = build_tree_optimized(value, node_path)
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
        elif isinstance(value, list):
            children = [
                {"label": item, "value": f"{node_path} > {item}"}
                for item in value
            ]
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
    return tree

def process_tree_selections(selections, min_depth=2):
    """Process tree selections"""
    if not selections or not selections.get('checked'):
        return []

    result = []
    for value in selections['checked']:
        parts = value.split(' > ')
        if len(parts) >= min_depth:
            result.append(parts[min_depth - 1])
    return result


@st.fragment
def render_filters():
    """Renders the filter UI and processes selections"""
    filters, study_types_tree, mechanisms_tree, behaviors_tree = load_and_process_filters()

    _render_filter_ui(filters, study_types_tree, mechanisms_tree, behaviors_tree)

    processed_selections = _process_current_selections(filters)
    _update_session_state(processed_selections)

def _render_filter_ui(filters, study_types_tree, mechanisms_tree, behaviors_tree):
    """Renders all filter UI components"""
    st.markdown("###### Poverty Contexts")
    st.multiselect(
        "Select",
        list(filters['poverty_contexts'].keys()),
        key="sankey_contexts"
    )

    st.markdown("###### Study Types")
    tree_select(study_types_tree, key="sankey_study_types")

    st.markdown("###### Psychological Mechanisms")
    tree_select(mechanisms_tree, key="sankey_mechanisms")

    st.markdown("###### Behavioral Outcomes")
    tree_select(behaviors_tree, key="sankey_behaviors")

def _process_current_selections(filters):
    """Processes current UI selections into usable format"""
    selected_contexts = st.session_state.get("sankey_contexts", [])
    selected_study_types = st.session_state.get("sankey_study_types", {})
    selected_mechanisms = st.session_state.get("sankey_mechanisms", {})
    selected_behaviors = st.session_state.get("sankey_behaviors", {})

    all_selected_context = []
    for context_key in selected_contexts:
        all_selected_context.extend(filters['poverty_contexts'][context_key])

    all_selected_study_types = process_tree_selections(selected_study_types, min_depth=3)
    all_selected_mechanisms = process_tree_selections(selected_mechanisms, min_depth=2)
    all_selected_behaviors = process_tree_selections(selected_behaviors, min_depth=2)

    return {
        'contexts': all_selected_context,
        'study_types': all_selected_study_types,
        'mechanisms': all_selected_mechanisms,
        'behaviors': all_selected_behaviors
    }

def _update_session_state(processed_selections):
    """Updates session state with new selections"""
    for key, value in processed_selections.items():
        st.session_state.ui_state[key] = value

def _create_sankey_signature():
    data = {
        'contexts': sorted(st.session_state.ui_state.get('contexts', [])),
        'study_types': sorted(st.session_state.ui_state.get('study_types', [])),
        'mechanisms': sorted(st.session_state.ui_state.get('mechanisms', [])),
        'behaviors': sorted(st.session_state.ui_state.get('behaviors', [])),
        'country': st.session_state.ui_state.get('selected_country', 'All'),
        'institution': st.session_state.ui_state.get('selected_institution', 'All')
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

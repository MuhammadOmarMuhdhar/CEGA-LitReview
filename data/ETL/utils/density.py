import pandas as pd
import logging
from featureEngineering import density

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run(google_sheets, spreadsheet_id_json):
    """
    Calculate density metrics for the papers and save results to Google Sheets.
    
    Args:
        google_sheets: GoogleSheets API instance
        spreadsheet_id_json: Dictionary containing spreadsheet IDs
        
    Returns:
        dict: Density dictionary containing calculation results
    """
    try:
        # Load papers from Google Sheets
        papers = google_sheets.read(spreadsheet_id_json['papers'])
        papers = papers.to_dict(orient='records')
        # Calculate density
        density_dict = density.run(papers)
        
        # Save density data to Google Sheets
        _save_density_data(google_sheets, density_dict, spreadsheet_id_json)
        
    except Exception as e:
        logger.error(f"Error calculating density: {e}")
        return {}

def _save_density_data(google_sheets, density_dict, spreadsheet_id_json):
    """
    Save density data to Google Sheets.
    
    Args:
        google_sheets: GoogleSheets API instance
        density_dict: Dictionary containing density data
        spreadsheet_id_json: Dictionary containing spreadsheet IDs
    """
    try:
        # Create dataframes for density data
        df_density = pd.DataFrame({
            'x_flat': density_dict['x_flat'],
            'y_flat': density_dict['y_flat'],
            'density': density_dict['density_flat']
        })
        df_x = pd.DataFrame(density_dict['x'])
        df_y = pd.DataFrame(density_dict['y'])
        
        # Save to Google Sheets
        google_sheets.replace(
            df=df_density,
            spreadsheet_id=spreadsheet_id_json['density']
        )
        google_sheets.replace(
            df=df_x,
            spreadsheet_id=spreadsheet_id_json['density_X']
        )
        google_sheets.replace(
            df=df_y,
            spreadsheet_id=spreadsheet_id_json['density_Y']
        )
    except Exception as e:
        logger.error(f"Error saving density data: {e}")
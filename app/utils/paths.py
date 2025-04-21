import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_app_path(relative_path):
    """
    Convert a relative path to an absolute path based on the application root directory.
    
    Args:
        relative_path (str): Relative path from the application root
        
    Returns:
        str: Absolute path
    """
    return os.path.join(BASE_DIR, relative_path.lstrip('/').replace('/', os.sep))

def get_model_path(model_name):
    """
    Get the absolute path to a model file.
    
    Args:
        model_name (str): Name of the model file
        
    Returns:
        str: Absolute path to the model file
    """
    return get_app_path(f'app/models/{model_name}')

def get_asset_path(asset_name):
    """
    Get the absolute path to an asset file.
    
    Args:
        asset_name (str): Name of the asset file
        
    Returns:
        str: Absolute path to the asset file
    """
    return get_app_path(f'app/ui/assets/{asset_name}')

def get_output_path(filename):
    """
    Get the absolute path for an output file.
    
    Args:
        filename (str): Name of the output file
        
    Returns:
        str: Absolute path for the output file
    """
    output_dir = os.path.join(BASE_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)
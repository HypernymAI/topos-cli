import subprocess
import yaml
import os
from ..utilities.utils import get_python_command, get_root_directory


def download_spacy_model(model_selection):
    if model_selection == 'small':
        model_name = "en_core_web_sm"
    elif model_selection == 'med':
        model_name = "en_core_web_md"
    elif model_selection == 'large':
        model_name = "en_core_web_lg"
    elif model_selection == 'trf':
        model_name = "en_core_web_trf"
    else: #default
        model_name = "en_core_web_sm"
        
    python_command = get_python_command()
    
    # Define the path to the config.yaml file
    config_path = os.path.join(get_root_directory(), 'config.yaml')
    try:
        subprocess.run([python_command, '-m', 'spacy', 'download', model_name], check=True)
        # Write updated settings to YAML file
        with open(config_path, 'w') as file:
            yaml.dump({'active_spacy_model': model_name}, file)
        print(f"Successfully downloaded '{model_name}' spaCy model.")
        print(f"'{model_name}' set as active model.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading '{model_name}' spaCy model: {e}")
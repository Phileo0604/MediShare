import os
import json
import shutil
from datetime import datetime


def backup_global_model(config, suffix=None):
    """Create a backup of the current global model."""
    # Get paths from config
    model_path = config["model"]["parameters_file"]
    backup_dir = config["server"].get("backup_dir", "model_backups")
    
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        backup_file = f"global_model_{timestamp}_{suffix}.json"
    else:
        backup_file = f"global_model_{timestamp}.json"
    
    backup_path = os.path.join(backup_dir, backup_file)
    
    # Copy current model to backup if it exists
    if os.path.exists(model_path):
        shutil.copy2(model_path, backup_path)
        print(f"Created backup at {backup_path}")
        return backup_path
    else:
        print(f"No model file found at {model_path}, no backup created")
        return None


def list_model_backups(config):
    """List all available model backups."""
    backup_dir = config["server"].get("backup_dir", "model_backups")
    
    if not os.path.exists(backup_dir):
        print(f"Backup directory {backup_dir} does not exist")
        return []
    
    # Get all JSON files in the backup directory
    backups = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
    backups.sort(reverse=True)  # Most recent first
    
    return backups


def restore_model_from_backup(config, backup_file=None):
    """Restore the global model from a backup file."""
    backup_dir = config["server"].get("backup_dir", "model_backups")
    model_path = config["model"]["parameters_file"]
    
    # If no specific backup file provided, use the most recent one
    if backup_file is None:
        backups = list_model_backups(config)
        if not backups:
            print("No backups available to restore")
            return False
        backup_file = backups[0]
    
    backup_path = os.path.join(backup_dir, backup_file)
    
    # Check if backup exists
    if not os.path.exists(backup_path):
        print(f"Backup file {backup_path} not found")
        return False
    
    # Create backup of current model before restoring
    backup_global_model(config, suffix="pre_restore")
    
    # Restore model from backup
    shutil.copy2(backup_path, model_path)
    print(f"Restored global model from {backup_path}")
    return True


def get_model_info(config):
    """Get information about the current global model."""
    model_path = config["model"]["parameters_file"]
    
    if not os.path.exists(model_path):
        return {
            "exists": False,
            "message": "Global model file not found"
        }
    
    try:
        # Get file modification time
        mod_time = os.path.getmtime(model_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get file size
        file_size = os.path.getsize(model_path) / 1024  # Size in KB
        
        # Load model to get parameter count
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        param_count = sum(len(param) for param in model_data)
        layer_counts = [len(param) for param in model_data]
        
        return {
            "exists": True,
            "path": model_path,
            "last_modified": mod_time_str,
            "size_kb": f"{file_size:.2f}",
            "parameter_count": param_count,
            "layer_shapes": layer_counts
        }
    
    except Exception as e:
        return {
            "exists": True,
            "path": model_path,
            "error": str(e)
        }
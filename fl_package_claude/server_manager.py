#!/usr/bin/env python
import argparse
import json
import os
import sys
from utils.server_utils import (
    backup_global_model,
    list_model_backups,
    restore_model_from_backup,
    get_model_info
)


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    """Server management utility."""
    parser = argparse.ArgumentParser(description="Federated Learning Server Manager")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup of the current global model")
    backup_parser.add_argument("--suffix", type=str, help="Optional suffix for the backup filename")
    
    # List backups command
    subparsers.add_parser("list-backups", help="List all available model backups")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore the global model from a backup")
    restore_parser.add_argument("--file", type=str, help="Specific backup file to restore (default: most recent)")
    
    # Info command
    subparsers.add_parser("info", help="Get information about the current global model")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file {args.config} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file {args.config} is not valid JSON")
        sys.exit(1)
    
    # Execute the specified command
    if args.command == "backup":
        backup_global_model(config, args.suffix)
    
    elif args.command == "list-backups":
        backups = list_model_backups(config)
        if backups:
            print("Available model backups:")
            for i, backup in enumerate(backups):
                print(f"{i+1}. {backup}")
        else:
            print("No model backups found")
    
    elif args.command == "restore":
        restore_model_from_backup(config, args.file)
    
    elif args.command == "info":
        info = get_model_info(config)
        if info["exists"]:
            print("\nGlobal Model Information:")
            print(f"Path: {info['path']}")
            print(f"Last Modified: {info['last_modified']}")
            print(f"Size: {info['size_kb']} KB")
            if "error" in info:
                print(f"Error reading model: {info['error']}")
            else:
                print(f"Parameter Count: {info['parameter_count']}")
                print(f"Layer Shapes: {info['layer_shapes']}")
        else:
            print(info["message"])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
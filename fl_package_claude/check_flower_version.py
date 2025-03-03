#!/usr/bin/env python
"""
Diagnostic script to check Flower version and available APIs.
This will help us determine the correct way to use the Flower library.
"""

import sys
import importlib.metadata
import inspect


def main():
    """Print information about installed Flower version and APIs."""
    print("\n=== Flower Version Diagnostic ===\n")

    # Check if flwr is installed
    try:
        flower_version = importlib.metadata.version('flwr')
        print(f"Flower version: {flower_version}")
    except importlib.metadata.PackageNotFoundError:
        print("Flower (flwr) package is not installed.")
        return

    # Import Flower
    import flwr as fl
    
    # Print available submodules
    print("\nAvailable submodules in flwr:")
    for module_name in dir(fl):
        if not module_name.startswith('_'):
            module = getattr(fl, module_name)
            if isinstance(module, type(fl)):
                print(f"  - {module_name}")
    
    # Check client module
    print("\nExamining flwr.client module:")
    client_module = fl.client
    client_items = dir(client_module)
    
    print("\nAvailable items in flwr.client:")
    for item_name in client_items:
        if not item_name.startswith('_'):
            item = getattr(client_module, item_name)
            if inspect.isclass(item):
                print(f"  - {item_name} (class)")
            elif inspect.isfunction(item):
                print(f"  - {item_name} (function)")
                # If it's a function, show its signature
                try:
                    signature = inspect.signature(item)
                    print(f"      Signature: {item_name}{signature}")
                except Exception:
                    print(f"      Signature: unable to get signature")
            else:
                print(f"  - {item_name}")
    
    # Check if NumPyClient is available
    if 'NumPyClient' in client_items:
        print("\nNumPyClient is available.")
        
        # Check how to start a client
        if 'start_numpy_client' in client_items:
            start_numpy_client = getattr(client_module, 'start_numpy_client')
            print("\nstart_numpy_client is available:")
            try:
                signature = inspect.signature(start_numpy_client)
                print(f"Signature: start_numpy_client{signature}")
                print("\nExample usage:")
                print("fl.client.start_numpy_client(server_address=\"127.0.0.1:8080\", client=numpy_client)")
            except Exception as e:
                print(f"Error getting signature: {e}")
        else:
            print("\nstart_numpy_client is NOT available")
        
        if 'start_client' in client_items:
            start_client = getattr(client_module, 'start_client')
            print("\nstart_client is available:")
            try:
                signature = inspect.signature(start_client)
                print(f"Signature: start_client{signature}")
                print("\nExample usage:")
                print("fl.client.start_client(server_address=\"127.0.0.1:8080\", client=client)")
            except Exception as e:
                print(f"Error getting signature: {e}")
        else:
            print("\nstart_client is NOT available")
    
    # Check server module
    print("\nExamining flwr.server module:")
    server_module = fl.server
    server_items = dir(server_module)
    
    print("\nKey items in flwr.server:")
    for item_name in ['start_server', 'Server', 'ServerConfig']:
        if item_name in server_items:
            item = getattr(server_module, item_name)
            if inspect.isclass(item):
                print(f"  - {item_name} (class is available)")
            elif inspect.isfunction(item):
                print(f"  - {item_name} (function is available)")
                try:
                    signature = inspect.signature(item)
                    print(f"      Signature: {item_name}{signature}")
                except Exception:
                    print(f"      Signature: unable to get signature")
            else:
                print(f"  - {item_name} (available)")
        else:
            print(f"  - {item_name} (NOT available)")

    print("\n=== End of Diagnostic ===\n")
    print("Please share this output to help diagnose the correct Flower API to use.")


if __name__ == "__main__":
    main()
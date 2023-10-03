import pkg_resources
import requests
import os
import sys
import subprocess

REQUIRED_DEPENDENCIES = {  # Dictionary of required dependencies and their versions
    'captum': '0.6.0',
    'contextily': '1.3.0',
    'geopandas': '0.12.2', 
    'geopy': '2.3.0',
    'imbalanced-learn': '0.10.1',
    'joblib': '1.2.0',
    'matplotlib': '3.7.2',
    'notebook': '6.5.4',
    'numpy': '1.24.3',
    'pandas': '2.0.0',
    'plotly': '5.14.1',
    'pyproj': '3.5.0', 
    'pytopk': '0.1.0',
    'pytorch_tabnet': '4.1.0',
    'requests': '2.29.0',
    'rioxarray': '0.14.1', 
    'rtdl': '0.0.13',
    'rtree': '1.0.1',
    'scipy': '1.10.1',
    'shapely': '2.0.1', 
    'scikit-learn': '1.2.2',
    'torch':  '1.13.1',
    'torchmetrics': '0.11.4',
    'tqdm': '4.65.0',
    'xgboost': '1.7.5'
}

class Check:
    """
    Pipeline class for installation check.
    """
    def run(self, args):
        """
        Run the 'Check' pipeline of the framework.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        if args.check_files:  # Check if file checking is enabled
            github_api = 'https://api.github.com/repos/cesar-leblanc/hdm-framework/contents'  # GitHub API URL for repository contents
            stack = [(github_api, '')]  # Initialize stack for traversing repository contents
            missing_files = []  # List to store missing files
            while stack:  # Continue traversing while stack is not empty
                current_api, current_path = stack.pop()  # Get current API and path from stack
                try:
                    response = requests.get(current_api)  # Send GET request to GitHub API
                    if response.status_code != 200:  # Check if response was successful
                        print(f"\nFailed to retrieve GitHub repository contents. Error code: {response.status_code}.")
                    github_contents = response.json()  # Parse response as JSON
                    for file_info in github_contents:  # Iterate over repository contents
                        file_name = file_info["name"]  # Get file name
                        file_path = os.path.join(os.getcwd(), current_path, file_name)  # Get absolute file path
                        if file_info["type"] == "file":  # Check if it's a file
                            if not os.path.exists(file_path):  # Check if file doesn't exist locally
                                missing_files.append(file_path)  # Add missing file to list
                        elif file_info["type"] == "dir":  # Check if it's a directory
                            stack.append((file_info["url"], os.path.join(current_path, file_name)))  # Add directory to stack for further traversal
                except requests.exceptions.RequestException as e:
                    print("\nAn error occurred while accessing the GitHub API.")  # Handle API request error
                except OSError as e:
                    print("\nAn error occurred while accessing the local file system.")  # Handle local file system error
            if missing_files:  # Check if there are missing files
                print("\nMissing files:")
                for file_name in missing_files:  # Print each missing file
                    print(file_name)
            else:
                print("\nNo missing files.")  # Indicate that no files are missing
        else:
            print("\nNot checking if files are missing.")  # File checking is disabled

        if args.check_dependencies:  # Check if dependency checking is enabled
            missing_dependencies = {}  # Dictionary to store missing dependencies
            for dependency, required_version in REQUIRED_DEPENDENCIES.items():  # Iterate over required dependencies
                try:
                    pkg_resources.require(dependency)  # Check if the package is installed
                    installed_version = pkg_resources.get_distribution(dependency).version  # Get the installed version
                    if installed_version != required_version:  # Compare installed version with required version
                        missing_dependencies[dependency] = required_version  # Add missing dependency to the dictionary
                except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):  # Handle package not found or version conflict
                    missing_dependencies[dependency] = required_version  # Add missing dependency to the dictionary
            if missing_dependencies:  # Check if there are missing dependencies
                print("\nMissing dependencies:")
                for dependency, version in missing_dependencies.items():  # Print each missing dependency and its required version
                    print(f"{dependency} - {version}")
            else:
                print("\nNo missing dependencies.")  # Indicate that no dependencies are missing
        else:
            print("\nNot checking if dependencies are missing.")  # Dependency checking is disabled
                
        if args.check_environment:  # Check if environment checking is enabled
            required_python_version = (3, 7)  # Define the minimum required Python version
            python_version = sys.version_info[:2]  # Get the current Python version
            if python_version < required_python_version:  # Compare Python version with required version
                print(f"\nPython {required_python_version[0]}.{required_python_version[1]} or later is required.")
            cuda_version = subprocess.getoutput("nvcc --version")  # Execute shell command to retrieve CUDA version
            if "command not found" in cuda_version:  # Check if CUDA is not installed
                print("\nCUDA is not installed.")
            else:
                if python_version >= required_python_version:  # Check if Python version meets the requirement
                    print("\nEnvironment is properly configured.")  # Indicate that the environment is properly configured
        else:
            print("\nNot checking if environment is properly configured.")  # Environment checking is disabled

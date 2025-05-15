import os
import shutil
import glob

def cleanup_project():
    # Directories to clean
    cache_dirs = [
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        '.coverage',
        'htmlcov',
        '*.egg-info',
        'build',
        'dist',
        '.ipynb_checkpoints'
    ]
    
    # File patterns to remove
    cache_files = [
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.coverage',
        '*.log',
        '*.tmp',
        '*.bak',
        '*.swp',
        '.DS_Store'
    ]
    
    # Walk through the project directory
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment directories
        if 'venv' in root or 'env' in root or '.git' in root:
            continue
            
        # Remove cache directories
        for cache_dir in cache_dirs:
            dir_pattern = os.path.join(root, cache_dir)
            for dir_path in glob.glob(dir_pattern):
                if os.path.isdir(dir_path):
                    print(f"Removing directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
        
        # Remove cache files
        for cache_file in cache_files:
            file_pattern = os.path.join(root, cache_file)
            for file_path in glob.glob(file_pattern):
                if os.path.isfile(file_path):
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)
    
    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup_project() 
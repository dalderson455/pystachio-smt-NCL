import os
import re

def update_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace .tsv with .csv
    updated_content = content.replace('.tsv', '.csv')
    
    # Only write to the file if changes were made
    if updated_content != content:
        with open(file_path, 'w') as file:
            file.write(updated_content)
        print(f"Updated: {file_path}")

def main():
    for root, dirs, files in os.walk('pystachio_smt'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_file(file_path)

if __name__ == "__main__":
    main()
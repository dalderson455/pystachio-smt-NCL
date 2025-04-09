import os
import re

def update_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace old imports with new ones
    content = content.replace('import dash_html_components as html', 'from dash import html')
    content = content.replace('import dash_core_components as dcc', 'from dash import dcc')
    content = content.replace('import dash_table', 'from dash import dash_table')
    
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Updated: {file_path}")

def main():
    for root, dirs, files in os.walk('pystachio_smt'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if 'import dash_html_components as html' in content or 'import dash_core_components as dcc' in content or 'import dash_table' in content:
                    update_file(file_path)

if __name__ == "__main__":
    main()
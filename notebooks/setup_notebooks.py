"""
Setup script for Jupyter notebooks in the SelfOrganizingAI-ML project.
Installs required dependencies and configures the environment.
"""

import sys
import subprocess
import pkg_resources

def install_dependencies():
    print("Installing notebook dependencies...")
    required = {
        'plotly': '5.18.0',
        'ipywidgets': '8.1.1',
        'jupyterlab': '4.0.9',
        'nbformat': '5.9.2',
        'ipykernel': '6.27.1'
    }

    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package, version in required.items():
        if package not in installed:
            print(f"Installing {package}=={version}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        elif installed[package] != version:
            print(f"Upgrading {package} from {installed[package]} to {version}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}", "--upgrade"])

def setup_jupyter():
    print("Setting up Jupyter environment...")
    try:
        # Enable widgets extension
        subprocess.check_call(["jupyter", "labextension", "install", "@jupyter-widgets/jupyterlab-manager"])
        # Enable plotly extension
        subprocess.check_call(["jupyter", "labextension", "install", "jupyterlab-plotly"])
    except Exception as e:
        print(f"Warning: Could not setup Jupyter extensions: {e}")
        print("You may need to run these commands manually:")
        print("  jupyter labextension install @jupyter-widgets/jupyterlab-manager")
        print("  jupyter labextension install jupyterlab-plotly")

if __name__ == "__main__":
    print("Starting notebook environment setup...")
    install_dependencies()
    setup_jupyter()
    print("\nSetup complete! You can now run: jupyter lab")
    print("And open visualization_demo.ipynb to get started.")
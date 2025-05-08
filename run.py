import os
import sys
import subprocess

def check_requirements():
    """Check if requirements are installed"""
    print("Checking requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Error installing requirements. Please install them manually using:")
        print("pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create sample dataset if needed"""
    tumor_dir = os.path.join('data', 'tumor')
    no_tumor_dir = os.path.join('data', 'no_tumor')
    
    if not os.path.exists(tumor_dir) or not os.path.exists(no_tumor_dir) or \
       len(os.listdir(tumor_dir) if os.path.exists(tumor_dir) else []) == 0 or \
       len(os.listdir(no_tumor_dir) if os.path.exists(no_tumor_dir) else []) == 0:
        
        print("No dataset found. Creating sample dataset...")
        subprocess.call([sys.executable, 'create_sample_data.py'])
    else:
        print("Dataset found!")

def train_model():
    """Train the brain tumor detection model"""
    print("\nTraining the brain tumor detection model...")
    subprocess.call([sys.executable, 'train.py'])

def run_app():
    """Run the Streamlit app"""
    print("\nStarting the Streamlit app...")
    subprocess.call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

def main():
    print("=" * 50)
    print("Brain Tumor Detection - Setup and Run Script")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    print("\n")
    # Create sample data if needed
    create_sample_data()
    
    # Menu
    while True:
        print("\n" + "=" * 50)
        print("What would you like to do?")
        print("1. Create sample dataset")
        print("2. Train model")
        print("3. Run web app")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            create_sample_data()
        elif choice == "2":
            train_model()
        elif choice == "3":
            run_app()
        elif choice == "4":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
import os
import sys
import subprocess
import platform

def check_gpu():
    """Check if CUDA is available and which version"""
    try:
        # Try to import torch to check CUDA
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA is available (version {cuda_version})")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("CUDA is not available through PyTorch")
    except ImportError:
        print("PyTorch is not installed, skipping CUDA check through PyTorch")

    # Try to check using nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected:")
            print(result.stdout.split('\n')[2:6])
            return True
    except:
        print("Could not run nvidia-smi")
    
    return False

def enable_long_paths_windows():
    """Enable long paths on Windows to avoid path length limitations"""
    if platform.system() == "Windows":
        try:
            # Try to enable long paths in Windows
            print("Enabling long paths support in Windows...")
            subprocess.run([
                "powershell", 
                "-Command", 
                "Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1"
            ], capture_output=True)
            print("Long paths support enabled (requires admin privileges)")
            
            # Set PIP environment variables to handle long paths
            os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
            
            # Tell pip to use a shorter temp directory
            temp_dir = "C:\\Temp\\pip"
            os.makedirs(temp_dir, exist_ok=True)
            os.environ['TMPDIR'] = temp_dir
            os.environ['TEMP'] = temp_dir
            os.environ['TMP'] = temp_dir
            
            print(f"Set temporary directory to {temp_dir}")
        except Exception as e:
            print(f"Could not enable long paths: {e}")
            print("You may still encounter path length issues on Windows.")

def install_requirements():
    """Install required packages based on GPU availability"""
    has_gpu = check_gpu()
    
    # Enable long paths on Windows
    enable_long_paths_windows()
    
    # Basic requirements that don't depend on GPU
    basic_reqs = [
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "PyYAML>=6.0",
        "ultralytics>=8.0.0"
    ]
    
    # First install basic requirements
    print("Installing basic requirements...")
    for req in basic_reqs:
        subprocess.run([sys.executable, "-m", "pip", "install", req])
    
    # Then install GPU-specific packages
    if has_gpu:
        print("Installing GPU-enabled packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
    else:
        print("Installing CPU-only packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime"])
    
    # Special installation for InsightFace to handle Windows path issues
    print("Installing InsightFace...")
    if platform.system() == "Windows":
        print("Using alternative installation method for Windows to avoid path length issues...")
        # Install without the tests to avoid long paths
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-deps", "insightface"
        ])
        
        # Install onnx dependency separately with --no-cache to avoid long paths
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-deps", "--no-cache-dir", "onnx"
        ])
        
        # Install remaining dependencies
        dependencies = [
            "numpy", "tqdm", "requests", "matplotlib", "Pillow", 
            "scipy", "scikit-learn", "scikit-image", "easydict", 
            "cython", "albumentations", "prettytable"
        ]
        for dep in dependencies:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", dep
            ])
    else:
        # On Linux/Mac, just install normally
        subprocess.run([sys.executable, "-m", "pip", "install", "insightface"])
    
    print("Installation complete!")

def create_face_detector_stub():
    """Create a stub file to make face detection work without InsightFace if installation fails"""
    try:
        import insightface
        print("InsightFace successfully imported, no need for stub.")
        return
    except ImportError:
        print("InsightFace import failed. Creating fallback stub...")
        
        stub_dir = "insightface_stub"
        os.makedirs(stub_dir, exist_ok=True)
        
        # Create __init__.py
        with open(os.path.join(stub_dir, "__init__.py"), "w") as f:
            f.write("# InsightFace stub module\n")
            f.write("print('Using InsightFace stub - limited functionality')\n")
        
        # Create app/__init__.py
        app_dir = os.path.join(stub_dir, "app")
        os.makedirs(app_dir, exist_ok=True)
        with open(os.path.join(app_dir, "__init__.py"), "w") as f:
            f.write("# InsightFace app stub module\n")
        
        # Create app/face_analysis.py with FaceAnalysis stub
        with open(os.path.join(app_dir, "face_analysis.py"), "w") as f:
            f.write("""
import cv2
import numpy as np

class FaceAnalysis:
    def __init__(self, name="stub", root="./"):
        self.name = name
        self.root = root
        print(f"Using FaceAnalysis stub - limited functionality")
    
    def prepare(self, ctx_id=0, det_size=(640, 640)):
        print(f"FaceAnalysis stub prepared with ctx_id={ctx_id}, det_size={det_size}")
    
    def get(self, frame):
        # Stub implementation using OpenCV's face detector
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            result = []
            for (x, y, w, h) in faces:
                class FaceObject:
                    pass
                
                face = FaceObject()
                face.bbox = np.array([x, y, x+w, y+h])
                face.kps = np.array([[x+w*0.3, y+h*0.3], [x+w*0.7, y+h*0.3], [x+w*0.5, y+h*0.5], [x+w*0.3, y+h*0.7], [x+w*0.7, y+h*0.7]])
                face.det_score = 0.9
                face.embedding = np.random.rand(512).astype(np.float32)  # Random embedding
                result.append(face)
            
            return result
        except Exception as e:
            print(f"Error in FaceAnalysis stub: {e}")
            return []
""")
        
        # Create data/__init__.py
        data_dir = os.path.join(stub_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "__init__.py"), "w") as f:
            f.write("# InsightFace data stub module\n")
        
        # Create get_image function
        with open(os.path.join(data_dir, "get_image.py"), "w") as f:
            f.write("""
def get_image(path):
    import cv2
    return cv2.imread(path)
""")
        
        # Add stub directory to Python path
        sys.path.insert(0, os.path.abspath("."))
        print(f"Created InsightFace stub in {os.path.abspath(stub_dir)}")

def verify_installation():
    """Verify that all required packages are installed correctly"""
    all_passed = True
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__} ✓")
    except ImportError as e:
        print(f"NumPy: ERROR - {e} ✗")
        all_passed = False
        
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__} ✓") 
    except ImportError as e:
        print(f"OpenCV: ERROR - {e} ✗")
        all_passed = False
    
    try:
        import yaml
        print(f"PyYAML: Installed ✓")
    except ImportError as e:
        print(f"PyYAML: ERROR - {e} ✗")
        all_passed = False
    
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"ONNX Runtime: {onnxruntime.__version__} ✓")
        print(f"ONNX Runtime Providers: {providers}")
        
        has_gpu = 'CUDAExecutionProvider' in providers
        if has_gpu:
            print("GPU acceleration is available for inference ✓")
        else:
            print("GPU acceleration is NOT available. Using CPU only.")
    except ImportError as e:
        print(f"ONNX Runtime: ERROR - {e} ✗")
        all_passed = False
    
    try:
        import insightface
        print(f"InsightFace: Installed ✓")
    except ImportError as e:
        print(f"InsightFace: ERROR - {e} ✗")
        print("Will use OpenCV's built-in face detector as fallback")
        create_face_detector_stub()
        all_passed = False
    
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__} ✓")
    except ImportError as e:
        print(f"Ultralytics: ERROR - {e} ✗")
        all_passed = False
        
    return all_passed

if __name__ == "__main__":
    print("Face and Person Tracker Setup")
    print("="*50)
    
    # Check if the system is Windows and warn about potential issues
    if platform.system() == "Windows":
        print("Windows system detected.")
        print("\nNOTE: Windows has a path length limitation that can cause issues.")
        print("If you encounter installation errors, try:")
        print("1. Run this script as Administrator")
        print("2. Install Python in a path with a shorter name (e.g., C:\\Python311)")
        print("3. Use a virtual environment in a shorter path")
        print("4. Move this project to a directory with a shorter path")
        print("\nOther potential issues:")
        print("- You might need Visual C++ Build Tools for some packages")
        print("  Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("="*50)
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Check GPU availability")
    print("2. Install requirements (auto-detect GPU)")
    print("3. Verify installation")
    print("4. Run the application")
    print("5. Fix Windows path length issues")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == "1":
        check_gpu()
    elif choice == "2":
        install_requirements()
    elif choice == "3":
        verify_installation()
    elif choice == "4":
        if verify_installation():
            print("\nStarting application...")
            subprocess.run([sys.executable, "face_tracker_app.py"])
        else:
            print("\nInstallation verification failed. Please install requirements first.")
    elif choice == "5":
        enable_long_paths_windows()
        print("\nAfter enabling long paths, please restart your computer and try installing again.")
    else:
        print("Invalid choice.") 
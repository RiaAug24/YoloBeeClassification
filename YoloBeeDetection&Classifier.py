import os
import torch
import yaml
import logging
import shutil
import subprocess
import sys
from pathlib import Path

class EnhancedBeeTrackingPipeline:
    def __init__(self, base_path: str, model_type: str = "yolov8"):
        self.base_path = Path(base_path)
        self.model_type = model_type  # 'yolov5', 'yolov7' or 'yolov8'
        self.logger = self._setup_logger()
        self.model = None
        self.dirs = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def setup_environment(self):
        """Setup complete training environment"""
        self.logger.info("Setting up training environment...")

        # Setup directory structure
        self._setup_directories()

        # Setup model environment based on model type
        if self.model_type == "yolov5":
            self._setup_yolov5()
        elif self.model_type == "yolov7":
            self._setup_yolov7()
        elif self.model_type == "yolov8":
            self._setup_yolov8()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Setup CUDA environment
        self._setup_cuda()

        self.logger.info("Environment setup complete")

    def _setup_directories(self):
        """Create necessary directories"""
        required_dirs = {
            'weights': self.base_path / 'weights',
            'results': self.base_path / 'results',
            'output': self.base_path / 'output',
            'runs': self.base_path / 'runs'
        }

        self.dirs = {}
        for key, dir_path in required_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            self.dirs[key] = dir_path
            self.logger.info(f"Directory ready: {dir_path}")

    def _run_command(self, command, cwd=None):
        """Safely run shell commands with proper error handling"""
        try:
            self.logger.info(f"Running command: {command}")
            if cwd:
                self.logger.info(f"Working directory: {cwd}")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.stdout:
                self.logger.info(f"Command output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Command stderr: {result.stderr}")
                
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("Command timed out after 1 hour")
            return False, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            return False, "", str(e)

    def _setup_yolov5(self):
        """Setup YOLOv5 repository and dependencies"""
        yolov5_dir = self.base_path / 'yolov5'
        
        if not yolov5_dir.exists():
            self.logger.info("Cloning YOLOv5 repository...")
            success, _, _ = self._run_command(
                f'git clone https://github.com/ultralytics/yolov5.git "{yolov5_dir}"'
            )
            if not success:
                raise RuntimeError("Failed to clone YOLOv5 repository")
            
        # Install requirements
        req_file = yolov5_dir / 'requirements.txt'
        if req_file.exists():
            self.logger.info("Installing YOLOv5 requirements...")
            success, _, _ = self._run_command(f'pip install -r "{req_file}"')
            if not success:
                self.logger.warning("Some requirements may have failed to install")
            
        # Download pre-trained weights
        weights_file = yolov5_dir / 'yolov5s.pt'
        if not weights_file.exists():
            self.logger.info("Downloading YOLOv5s weights...")
            success, _, _ = self._run_command(
                'python -c "import torch; torch.hub.load(\'ultralytics/yolov5\', \'yolov5s\', pretrained=True)"',
                cwd=yolov5_dir
            )
            if not success:
                self.logger.warning("Failed to download weights automatically")

    def _setup_yolov7(self):
        """Setup YOLOv7 repository and weights"""
        yolov7_dir = self.base_path / 'yolov7'
        
        if not yolov7_dir.exists():
            self.logger.info("Cloning YOLOv7 repository...")
            success, _, _ = self._run_command(
                f'git clone https://github.com/WongKinYiu/yolov7.git "{yolov7_dir}"'
            )
            if not success:
                raise RuntimeError("Failed to clone YOLOv7 repository")
            
        # Install requirements
        req_file = yolov7_dir / 'requirements.txt'
        if req_file.exists():
            self.logger.info("Installing YOLOv7 requirements...")
            success, _, _ = self._run_command(f'pip install -r "{req_file}"')
            if not success:
                self.logger.warning("Some requirements may have failed to install")

        # Download pre-trained weights using Python instead of wget
        weights_file = yolov7_dir / 'yolov7.pt'
        if not weights_file.exists():
            self.logger.info("Downloading YOLOv7 weights...")
            try:
                import urllib.request
                url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'
                urllib.request.urlretrieve(url, str(weights_file))
                self.logger.info("YOLOv7 weights downloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to download YOLOv7 weights: {e}")

    def _setup_yolov8(self):
        """Setup YOLOv8 environment"""
        try:
            from ultralytics import YOLO
            import ultralytics
            self.logger.info(f"Ultralytics version: {ultralytics.__version__}")
        except ImportError:
            self.logger.info("Installing Ultralytics...")
            success, _, _ = self._run_command('pip install ultralytics')
            if not success:
                raise RuntimeError("Failed to install ultralytics")

    def _setup_cuda(self):
        """Setup CUDA environment"""
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.logger.warning("CUDA not available. Using CPU.")

    def check_dataset(self):
        """Check if dataset is present in the working directory"""
        self.logger.info("Checking for dataset...")
        
        # Check for common dataset directory structures
        possible_paths = [
            self.base_path / 'BeeDataset',
            self.base_path / 'dataset',
            self.base_path / 'data'
        ]
        
        dataset_found = False
        dataset_path = None
        
        for path in possible_paths:
            if path.exists():
                # Check for YOLO format (images and labels directories)
                images_dir = path / 'images'
                labels_dir = path / 'labels'
                
                if images_dir.exists() and labels_dir.exists():
                    train_img = images_dir / 'train'
                    val_img = images_dir / 'val'
                    train_label = labels_dir / 'train'
                    val_label = labels_dir / 'val'
                    
                    if all(p.exists() for p in [train_img, val_img, train_label, val_label]):
                        train_imgs = list(train_img.glob('*.jpg')) + list(train_img.glob('*.png'))
                        val_imgs = list(val_img.glob('*.jpg')) + list(val_img.glob('*.png'))
                        train_labels = list(train_label.glob('*.txt'))
                        val_labels = list(val_label.glob('*.txt'))
                        
                        if train_imgs and val_imgs and train_labels and val_labels:
                            self.logger.info(f"Dataset found at: {path}")
                            self.logger.info(f"Train images: {len(train_imgs)}")
                            self.logger.info(f"Train labels: {len(train_labels)}")
                            self.logger.info(f"Val images: {len(val_imgs)}")
                            self.logger.info(f"Val labels: {len(val_labels)}")
                            
                            dataset_path = path
                            dataset_found = True
                            break
        
        if not dataset_found:
            self.logger.error("No valid dataset found! Please ensure your dataset is in YOLO format with:")
            self.logger.error("- images/train/ and images/val/ directories with image files")
            self.logger.error("- labels/train/ and labels/val/ directories with label files")
            raise FileNotFoundError("Dataset not found")
        
        # Create data.yaml if it doesn't exist
        self._create_dataset_config(dataset_path)
        return dataset_found

    def _create_dataset_config(self, dataset_path: Path):
        """Create YAML configuration file for the dataset"""
        config = {
            'path': str(dataset_path.resolve()),
            'train': str((dataset_path / 'images' / 'train').resolve()),
            'val': str((dataset_path / 'images' / 'val').resolve()),
            'nc': 4,  # number of classes
            'names': ['foraging', 'defense', 'fanning', 'washboarding']
        }

        yaml_path = self.base_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
            
        self.logger.info(f"Created dataset config at {yaml_path}")

    def find_trained_models(self):
        """Find trained models in runs directory (not pre-trained models)"""
        self.logger.info("Searching for trained models...")
        
        trained_models = {}
        runs_dir = self.base_path / 'runs'
        
        if not runs_dir.exists():
            self.logger.info("No runs directory found")
            return trained_models
        
        # Look in train subdirectories
        train_dir = runs_dir / 'train'
        if train_dir.exists():
            for exp_dir in train_dir.iterdir():
                if exp_dir.is_dir():
                    # Check for weights directory
                    weights_dir = exp_dir / 'weights'
                    if weights_dir.exists():
                        # Look for best.pt or last.pt
                        for weight_file in ['best.pt', 'last.pt']:
                            weight_path = weights_dir / weight_file
                            if weight_path.exists():
                                # Determine model type from directory name
                                model_type = self._infer_model_type_from_path(exp_dir.name)
                                model_name = f"{exp_dir.name}_{weight_file.split('.')[0]}"
                                
                                trained_models[model_name] = {
                                    'path': str(weight_path),
                                    'type': model_type,
                                    'size': weight_path.stat().st_size / (1024 * 1024),
                                    'last_modified': weight_path.stat().st_mtime,
                                    'experiment': exp_dir.name
                                }
        
        # Sort by last modified time
        sorted_models = dict(sorted(
            trained_models.items(), 
            key=lambda item: item[1]['last_modified'], 
            reverse=True
        ))
        
        if sorted_models:
            self.logger.info(f"Found {len(sorted_models)} trained models:")
            for name, info in sorted_models.items():
                self.logger.info(f"  - {name} ({info['type']}): {info['size']:.2f} MB")
        else:
            self.logger.info("No trained models found in runs directory")
            
        return sorted_models

    def _infer_model_type_from_path(self, path_name):
        """Infer model type from directory/file name"""
        path_lower = path_name.lower()
        if 'yolov5' in path_lower or 'v5' in path_lower:
            return 'yolov5'
        elif 'yolov7' in path_lower or 'v7' in path_lower:
            return 'yolov7'
        elif 'yolov8' in path_lower or 'v8' in path_lower:
            return 'yolov8'
        else:
            return self.model_type  # Default to pipeline's model type

    def load_or_train_model(self, force_train=False, epochs=15, batch_size=4):
        """Load trained model or train new one"""
        self.logger.info("Looking for trained models or training new one...")
        
        # Find trained models for the current model type
        trained_models = self.find_trained_models()
        
        # Filter models by current model type
        compatible_models = {k: v for k, v in trained_models.items() 
                           if v['type'] == self.model_type}
        
        if compatible_models and not force_train:
            # Load the most recent compatible model
            most_recent_name = next(iter(compatible_models))
            most_recent_info = compatible_models[most_recent_name]
            
            self.logger.info(f"Loading trained model: {most_recent_name}")
            
            try:
                self.model = self._load_model(most_recent_info['path'])
                self.logger.info(f"Successfully loaded {self.model_type} model")
                return self.model
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.logger.info("Will train new model instead.")
        else:
            reason = f"No trained {self.model_type} models found." if not compatible_models else "Force training enabled."
            self.logger.info(f"{reason} Training new model...")
        
        # Train new model
        self.train_model(epochs=epochs, batch_size=batch_size)
        return self.model

    def _load_model(self, model_path):
        """Load model based on type with better error handling"""
        try:
            if self.model_type == "yolov8":
                from ultralytics import YOLO
                model = YOLO(model_path)
                self.logger.info(f"YOLOv8 model loaded from {model_path}")
                return model
                
            elif self.model_type == "yolov5":
                # For YOLOv5, we need to handle the loading more carefully
                if Path(model_path).exists():
                    # Load custom trained model
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                    self.logger.info(f"YOLOv5 custom model loaded from {model_path}")
                else:
                    # Load pretrained model
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    self.logger.info("YOLOv5 pretrained model loaded")
                
                # Store the model path for later use
                model.pt_path = model_path
                return model
                
            elif self.model_type == "yolov7":
                # For YOLOv7, we store the path as it uses different loading mechanism
                if Path(model_path).exists():
                    self.logger.info(f"YOLOv7 model path set to {model_path}")
                    return model_path
                else:
                    self.logger.error(f"YOLOv7 model file not found: {model_path}")
                    return None
            else:
                raise ValueError(f"Unsupported model type for loading: {self.model_type}")
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def train_model(self, epochs: int = 15, batch_size: int = 8):
        """Train model with optimized settings"""
        self.logger.info(f"Starting {self.model_type} model training with {epochs} epochs...")
        
        if self.model_type == "yolov5":
            self._train_yolov5_optimized(epochs, batch_size)
        elif self.model_type == "yolov7":
            self._train_yolov7_optimized(epochs, batch_size)
        elif self.model_type == "yolov8":
            self._train_yolov8_optimized(epochs, batch_size)
            
        self.logger.info("Training complete")

    def _train_yolov5_optimized(self, epochs: int, batch_size: int):
        """Optimized YOLOv5 training with proper path handling"""
        try:
            yolov5_dir = self.base_path / 'yolov5'
            data_yaml = self.base_path / 'data.yaml'
            
            # Verify files exist
            if not yolov5_dir.exists():
                raise FileNotFoundError(f"YOLOv5 directory not found: {yolov5_dir}")
            if not data_yaml.exists():
                raise FileNotFoundError(f"Data config not found: {data_yaml}")
            
            # Check for pre-trained weights
            weights_file = yolov5_dir / 'yolov5s.pt'
            if not weights_file.exists():
                self.logger.info("Downloading YOLOv5s weights...")
                success, _, _ = self._run_command(
                    'python -c "import torch; model = torch.hub.load(\'ultralytics/yolov5\', \'yolov5s\', pretrained=True)"',
                    cwd=yolov5_dir
                )
                if not success:
                    self.logger.warning("Failed to download weights, trying alternative method")
                    weights_file = 'yolov5s.pt'  # Use online weights
            
            # Training parameters
            img_size = 640
            device = '0' if torch.cuda.is_available() else 'cpu'
            project_dir = self.base_path / 'runs' / 'train'
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Build training command using Python subprocess instead of os.system
            train_script = yolov5_dir / 'train.py'
            if not train_script.exists():
                raise FileNotFoundError(f"Training script not found: {train_script}")
            
            cmd_args = [
                sys.executable, str(train_script),
                '--data', str(data_yaml),
                '--weights', str(weights_file) if weights_file.exists() else 'yolov5s.pt',
                '--img', str(img_size),
                '--batch-size', str(batch_size),  
                '--epochs', str(epochs),
                '--device', device,
                '--project', str(project_dir),
                '--name', 'bee_behavior_yolov5',
                '--cache',
                '--save-period', '5',
                '--patience', '10'
            ]

            self.logger.info(f"Training command: {' '.join(cmd_args)}")
            
            # Run training
            process = subprocess.Popen(
                cmd_args,
                cwd=yolov5_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                self.logger.info(line.strip())
                
            process.wait()
            
            if process.returncode == 0:
                # Load the trained model
                best_model_path = project_dir / 'bee_behavior_yolov5' / 'weights' / 'best.pt'
                if best_model_path.exists():
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(best_model_path))
                    self.logger.info("YOLOv5 training completed successfully")
                    
                    # Copy to weights directory
                    target_path = self.dirs['weights'] / 'best_yolov5_bee_behavior.pt'
                    shutil.copy(best_model_path, target_path)
                    self.logger.info(f"Model copied to {target_path}")
                else:
                    self.logger.error("Training completed but best.pt not found")
            else:
                self.logger.error(f"YOLOv5 training failed with return code: {process.returncode}")
                
        except Exception as e:
            self.logger.error(f"Error during YOLOv5 training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _train_yolov7_optimized(self, epochs: int, batch_size: int):
        """Optimized YOLOv7 training with proper path handling"""
        try:
            yolov7_dir = self.base_path / 'yolov7'
            data_yaml = self.base_path / 'data.yaml'
            
            # Verify files exist
            if not yolov7_dir.exists():
                raise FileNotFoundError(f"YOLOv7 directory not found: {yolov7_dir}")
            if not data_yaml.exists():
                raise FileNotFoundError(f"Data config not found: {data_yaml}")
            
            # Check for pre-trained weights
            weights_file = yolov7_dir / 'yolov7.pt'
            if not weights_file.exists():
                self.logger.warning("YOLOv7 weights not found, training may fail")
            
            # Training parameters
            img_size = 640
            device = '0' if torch.cuda.is_available() else 'cpu'
            project_dir = self.base_path / 'runs' / 'train'
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Build training command
            train_script = yolov7_dir / 'train.py'
            if not train_script.exists():
                raise FileNotFoundError(f"Training script not found: {train_script}")
            
            cmd_args = [
                sys.executable, str(train_script),
                '--data', str(data_yaml),
                '--weights', str(weights_file) if weights_file.exists() else 'yolov7.pt',
                '--img', str(img_size), str(img_size),
                '--batch-size', str(batch_size),
                '--epochs', str(epochs),
                '--device', device,
                '--project', str(project_dir),
                '--name', 'bee_behavior_yolov7',
                '--cache-images',
            ]

            self.logger.info(f"Training command: {' '.join(cmd_args)}")
            
            # Run training
            process = subprocess.Popen(
                cmd_args,
                cwd=yolov7_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                self.logger.info(line.strip())
                
            process.wait()
            
            if process.returncode == 0:
                # For YOLOv7, store the path as model reference
                best_model_path = project_dir / 'bee_behavior_yolov7' / 'weights' / 'best.pt'
                if best_model_path.exists():
                    self.model = str(best_model_path)
                    self.logger.info("YOLOv7 training completed successfully")
                    
                    # Copy to weights directory
                    target_path = self.dirs['weights'] / 'best_yolov7_bee_behavior.pt'
                    shutil.copy(best_model_path, target_path)
                    self.logger.info(f"Model copied to {target_path}")
                else:
                    self.logger.error("Training completed but best.pt not found")
            else:
                self.logger.error(f"YOLOv7 training failed with return code: {process.returncode}")
                
        except Exception as e:
            self.logger.error(f"Error during YOLOv7 training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _train_yolov8_optimized(self, epochs: int, batch_size: int):
        """Optimized YOLOv8 training"""
        try:
            from ultralytics import YOLO
            
            data_yaml = self.base_path / 'data.yaml'
            
            # Verify data file exists
            if not data_yaml.exists():
                raise FileNotFoundError(f"Data config file not found: {data_yaml}")
            
            # Initialize model with pre-trained weights
            model = YOLO('yolov8n.pt')  # Start with nano for faster training
            
            # Optimized training parameters
            training_args = {
                'data': str(data_yaml),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 0 if torch.cuda.is_available() else 'cpu',
                'project': str(self.base_path / 'runs' / 'train'),
                'name': 'bee_behavior_yolov8',
                'cache': True,
                'save_period': 5,
                'patience': 10,
                'cos_lr': True,
                'label_smoothing': 0.1,
                'mixup': 0.2,
                'augment': True,
                'lr0': 0.01,
                'lrf': 0.1,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.9,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'copy_paste': 0.3
            }
            
            self.logger.info("Starting YOLOv8 training with optimized parameters...")
            results = model.train(**training_args)
            
            # Save reference to trained model
            best_model_path = self.base_path / 'runs' / 'train' / 'bee_behavior_yolov8' / 'weights' / 'best.pt'
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                self.logger.info("YOLOv8 training completed successfully")
                
                # Also save to weights directory for easy access
                target_path = self.dirs['weights'] / f'best_yolov8_bee_behavior.pt'
                shutil.copy(best_model_path, target_path)
                self.logger.info(f"Model copied to {target_path}")
            else:
                self.logger.error("Training completed but best.pt not found")
                
        except Exception as e:
            self.logger.error(f"Error during YOLOv8 training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def predict_video(self, video_path: str, conf_threshold: float = 0.20):
        """Run prediction on a video with proper error handling and video processing"""
        self.logger.info(f"Running prediction on {video_path}")
        
        if self.model is None:
            self.logger.error("No model loaded. Please train or load a model first.")
            return None
        
        # Validate video path
        video_file = Path(video_path)
        if not video_file.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None
        
        if not video_file.is_file():
            self.logger.error(f"Path is not a file: {video_path}")
            return None
        
        # Check file permissions
        if not os.access(video_file, os.R_OK):
            self.logger.error(f"No read permission for file: {video_path}")
            return None
        
        # Validate video file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        if video_file.suffix.lower() not in valid_extensions:
            self.logger.error(f"Unsupported video format: {video_file.suffix}")
            return None
        
        try:
            if self.model_type == "yolov8":
                # Create output directory with proper permissions
                output_dir = self.dirs['output'] / f"prediction_{video_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results = self.model.predict(
                    source=str(video_file),
                    save=True,
                    project=str(self.dirs['output']),
                    name=f"prediction_{video_file.stem}",
                    conf=conf_threshold,
                    verbose=True
                )
                
            elif self.model_type == "yolov5":
                # For YOLOv5, we need to process video frame by frame or use detect.py
                self.logger.info("Running YOLOv5 inference on video...")
                
                yolov5_dir = self.base_path / 'yolov5'
                detect_script = yolov5_dir / 'detect.py'
                
                if not detect_script.exists():
                    self.logger.error(f"YOLOv5 detect script not found: {detect_script}")
                    # Try alternative method with cv2
                    return self._process_video_with_cv2(video_file, conf_threshold)
                
                # Create output directory
                output_dir = self.dirs['output'] / f"yolov5_prediction_{video_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Find the best trained model
                trained_models = self.find_trained_models()
                yolov5_models = {k: v for k, v in trained_models.items() if v['type'] == 'yolov5'}
                
                if yolov5_models:
                    model_path = list(yolov5_models.values())[0]['path']
                else:
                    # Use the loaded model path or default
                    if hasattr(self.model, 'pt_path'):
                        model_path = self.model.pt_path
                    else:
                        model_path = str(yolov5_dir / 'yolov5s.pt')
                
                cmd_args = [
                    sys.executable, str(detect_script),
                    '--weights', model_path,
                    '--source', str(video_file),
                    '--project', str(self.dirs['output']),
                    '--name', f"yolov5_prediction_{video_file.stem}",
                    '--conf-thres', str(conf_threshold),
                    '--save-txt',
                    '--save-conf',
                    '--exist-ok'
                ]
                
                self.logger.info(f"YOLOv5 detection command: {' '.join(cmd_args)}")
                success, stdout, stderr = self._run_command(' '.join(cmd_args), cwd=yolov5_dir)
                
                if not success:
                    self.logger.error(f"YOLOv5 detection failed: {stderr}")
                    # Fallback to frame-by-frame processing
                    return self._process_video_with_cv2(video_file, conf_threshold)
                
                results = f"YOLOv5 detection completed successfully"
                
            elif self.model_type == "yolov7":
                # For YOLOv7, use detect.py script with proper path handling
                yolov7_dir = self.base_path / 'yolov7'
                detect_script = yolov7_dir / 'detect.py'
                
                if not detect_script.exists():
                    raise FileNotFoundError(f"Detect script not found: {detect_script}")
                
                # Create output directory
                output_dir = self.dirs['output'] / f"yolov7_prediction_{video_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                cmd_args = [
                    sys.executable, str(detect_script),
                    '--weights', str(self.model),
                    '--source', str(video_file),
                    '--project', str(self.dirs['output']),
                    '--name', f"yolov7_prediction_{video_file.stem}",
                    '--conf-thres', str(conf_threshold),
                    '--save-txt',
                    '--save-conf',
                    '--exist-ok'
                ]
                
                success, stdout, stderr = self._run_command(' '.join(cmd_args), cwd=yolov7_dir)
                if not success:
                    self.logger.error(f"YOLOv7 detection failed: {stderr}")
                    return None
                results = f"YOLOv7 detection completed successfully"
            
            self.logger.info(f"Prediction complete. Results saved to output directory")
            return results
            
        except PermissionError as pe:
            self.logger.error(f"Permission error during prediction: {pe}")
            self.logger.error("Please check file/directory permissions")
            return None
        except FileNotFoundError as fe:
            self.logger.error(f"File not found error: {fe}")
            return None
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def _process_video_with_cv2(self, video_file: Path, conf_threshold: float):
        """Process video frame by frame using OpenCV as fallback"""
        try:
            import cv2
            self.logger.info("Processing video frame by frame using OpenCV...")
            
            # Open video
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_file}")
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Create output directory
            output_dir = self.dirs['output'] / f"cv2_prediction_{video_file.stem}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer for output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = output_dir / f"detected_{video_file.name}"
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            frame_count = 0
            detections_log = []
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference on frame
                if self.model_type == "yolov5":
                    # YOLOv5 inference
                    results = self.model(frame, size=640)
                    
                    # Draw detections on frame
                    annotated_frame = results.render()[0]
                    
                    # Extract detection info
                    detections = results.xyxy[0].cpu().numpy()
                    frame_detections = []
                    
                    for detection in detections:
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, cls = detection[:6]
                            if conf >= conf_threshold:
                                frame_detections.append({
                                    'frame': frame_count,
                                    'class': int(cls),
                                    'confidence': float(conf),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                })
                    
                    detections_log.extend(frame_detections)
                    out.write(annotated_frame)
                    
                elif self.model_type == "yolov8":
                    # YOLOv8 inference
                    results = self.model.predict(frame, conf=conf_threshold, verbose=False)
                    
                    if results and results[0].boxes is not None:
                        # Draw detections
                        annotated_frame = results[0].plot()
                        
                        # Extract detection info
                        boxes = results[0].boxes
                        for i in range(len(boxes)):
                            conf = float(boxes.conf[i])
                            cls = int(boxes.cls[i])
                            bbox = boxes.xyxy[i].cpu().numpy().tolist()
                            
                            detections_log.append({
                                'frame': frame_count,
                                'class': cls,
                                'confidence': conf,
                                'bbox': bbox
                            })
                        
                        out.write(annotated_frame)
                    else:
                        out.write(frame)
                
                # Progress logging
                if frame_count % 30 == 0:  # Log every 30 frames
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            # Clean up
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Save detection log
            import json
            log_file = output_dir / "detections_log.json"
            with open(log_file, 'w') as f:
                json.dump(detections_log, f, indent=2)
            
            self.logger.info(f"Video processing complete. Output saved to {output_dir}")
            self.logger.info(f"Total detections: {len(detections_log)}")
            
            return f"Frame-by-frame processing completed. {len(detections_log)} detections found."
            
        except ImportError:
            self.logger.error("OpenCV not installed. Cannot process video frame by frame.")
            self.logger.info("Install OpenCV with: pip install opencv-python")
            return None
        except Exception as e:
            self.logger.error(f"Error in frame-by-frame processing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def analyze_results(self, results):
        """Analyze prediction results with improved YOLOv7 support"""
        if results is None:
            self.logger.error("No results to analyze")
            return None
            
        class_names = ['foraging', 'defense', 'fanning', 'washboarding']
        class_counts = {name: 0 for name in class_names}
        
        try:
            if self.model_type == "yolov8":
                # YOLOv8 results analysis (unchanged)
                total_detections = 0
                confidence_scores = []
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                        
                        for cls_id, conf in zip(classes, confidences):
                            if int(cls_id) < len(class_names):
                                class_counts[class_names[int(cls_id)]] += 1
                                confidence_scores.append(conf)
                                total_detections += 1
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                self.logger.info(f"Analysis Results:")
                self.logger.info(f"Total detections: {total_detections}")
                self.logger.info(f"Average confidence: {avg_confidence:.3f}")
                self.logger.info("Class distribution:")
                for class_name, count in class_counts.items():
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
                    
            elif self.model_type == "yolov5":
                # YOLOv5 results analysis (unchanged)
                total_detections = 0
                confidence_scores = []
                
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                    for detection in results.xyxy[0]:
                        if len(detection) >= 6:
                            conf = float(detection[4])
                            cls_id = int(detection[5])
                            
                            if cls_id < len(class_names):
                                class_counts[class_names[cls_id]] += 1
                                confidence_scores.append(conf)
                                total_detections += 1
                else:
                    self._parse_yolov5_saved_results(class_counts, confidence_scores)
                    total_detections = sum(class_counts.values())
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                self.logger.info(f"YOLOv5 Analysis Results:")
                self.logger.info(f"Total detections: {total_detections}")
                self.logger.info(f"Average confidence: {avg_confidence:.3f}")
                self.logger.info("Class distribution:")
                for class_name, count in class_counts.items():
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
                    
            elif self.model_type == "yolov7":
                # Enhanced YOLOv7 results analysis
                confidence_scores = []
                
                self.logger.info("Starting YOLOv7 results analysis...")
                
                # Parse saved detection files
                self._parse_yolov7_saved_results(class_counts, confidence_scores)
                
                total_detections = sum(class_counts.values())
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
                self.logger.info(f"YOLOv7 Analysis Results:")
                self.logger.info(f"Total detections: {total_detections}")
                self.logger.info(f"Average confidence: {avg_confidence:.3f}")
                self.logger.info("Class distribution:")
                for class_name, count in class_counts.items():
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
                
                # If no detections found, provide debugging information
                if total_detections == 0:
                    self.logger.warning("No detections found in YOLOv7 results. Debugging information:")
                    
                    # List all directories in output folder
                    output_dirs = [d for d in self.dirs['output'].iterdir() if d.is_dir()]
                    self.logger.info(f"Output directories found: {[str(d.name) for d in output_dirs]}")
                    
                    # Check for any YOLOv7 related directories
                    yolov7_dirs = [d for d in output_dirs if 'yolov7' in d.name.lower()]
                    for yv7_dir in yolov7_dirs:
                        self.logger.info(f"YOLOv7 directory contents: {[f.name for f in yv7_dir.iterdir()]}")
                        
                        labels_dir = yv7_dir / 'labels'
                        if labels_dir.exists():
                            label_files = list(labels_dir.glob('*.txt'))
                            self.logger.info(f"Label files in {labels_dir}: {[f.name for f in label_files]}")
                            
                            # Check if label files have content
                            for label_file in label_files[:3]:  # Check first 3 files
                                try:
                                    with open(label_file, 'r') as f:
                                        content = f.read().strip()
                                        if content:
                                            self.logger.info(f"Sample content from {label_file.name}: {content[:100]}...")
                                        else:
                                            self.logger.info(f"Empty label file: {label_file.name}")
                                except Exception as e:
                                    self.logger.error(f"Error reading {label_file}: {e}")
                    
            return {
                'total_detections': sum(class_counts.values()),
                'class_counts': class_counts,
                'average_confidence': avg_confidence if 'avg_confidence' in locals() else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _parse_yolov5_saved_results(self, class_counts, confidence_scores):
        """Parse YOLOv5 saved detection results from output directory"""
        try:
            # Look for saved results in the output directory
            output_dirs = list(self.dirs['output'].glob("*"))
            
            for output_dir in output_dirs:
                if output_dir.is_dir():
                    # Look for labels directory (contains detection txt files)
                    labels_dir = output_dir / 'labels'
                    if labels_dir.exists():
                        # Parse each detection file
                        for label_file in labels_dir.glob('*.txt'):
                            self._parse_detection_file(label_file, class_counts, confidence_scores)
                    else:
                        # Check if detection files are directly in output directory
                        for txt_file in output_dir.glob('*.txt'):
                            if txt_file.name != 'results.txt':  # Skip results summary
                                self._parse_detection_file(txt_file, class_counts, confidence_scores)
        except Exception as e:
            self.logger.warning(f"Could not parse YOLOv5 saved results: {e}")

    def _parse_yolov7_saved_results(self, class_counts, confidence_scores):
        """Parse YOLOv7 saved detection results from output directory"""
        try:
            # Look for the specific YOLOv7 prediction directory
            yolov7_dirs = list(self.dirs['output'].glob("yolov7_prediction_*"))
            
            if not yolov7_dirs:
                # Fallback: look for any directory that might contain YOLOv7 results
                yolov7_dirs = [d for d in self.dirs['output'].iterdir() if d.is_dir() and 'yolov7' in d.name.lower()]
            
            self.logger.info(f"Looking for YOLOv7 results in directories: {[str(d) for d in yolov7_dirs]}")
            
            for output_dir in yolov7_dirs:
                if output_dir.is_dir():
                    # YOLOv7 typically saves detection files in a 'labels' subdirectory
                    labels_dir = output_dir / 'labels'
                    
                    if labels_dir.exists() and labels_dir.is_dir():
                        self.logger.info(f"Found labels directory: {labels_dir}")
                        # Parse detection files from labels directory
                        txt_files = list(labels_dir.glob('*.txt'))
                        self.logger.info(f"Found {len(txt_files)} label files")
                        
                        for txt_file in txt_files:
                            self.logger.debug(f"Parsing detection file: {txt_file}")
                            self._parse_detection_file(txt_file, class_counts, confidence_scores)
                    else:
                        # If no labels subdirectory, check the main directory
                        self.logger.info(f"No labels subdirectory found, checking main directory: {output_dir}")
                        txt_files = [f for f in output_dir.glob('*.txt') if f.name not in ['results.txt', 'summary.txt']]
                        self.logger.info(f"Found {len(txt_files)} detection files in main directory")
                        
                        for txt_file in txt_files:
                            self.logger.debug(f"Parsing detection file: {txt_file}")
                            self._parse_detection_file(txt_file, class_counts, confidence_scores)
                            
            total_parsed = sum(class_counts.values())
            self.logger.info(f"Successfully parsed {total_parsed} total detections from YOLOv7 results")
            
        except Exception as e:
            self.logger.error(f"Error parsing YOLOv7 saved results: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _parse_detection_file(self, file_path, class_counts, confidence_scores):
        """Parse individual detection file in YOLO format"""
        try:
            class_names = ['foraging', 'defense', 'fanning', 'washboarding']
            
            self.logger.debug(f"Parsing file: {file_path}")
            
            # Check if file exists and is readable
            if not file_path.exists():
                self.logger.warning(f"Detection file does not exist: {file_path}")
                return
                
            if not os.access(file_path, os.R_OK):
                self.logger.warning(f"No read permission for file: {file_path}")
                return
            
            # Check file size
            if file_path.stat().st_size == 0:
                self.logger.debug(f"Empty detection file: {file_path}")
                return
            
            detections_in_file = 0
            
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        
                        # YOLO format: class_id x_center y_center width height [confidence]
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                
                                # Validate class ID
                                if cls_id < 0 or cls_id >= len(class_names):
                                    self.logger.warning(f"Invalid class ID {cls_id} in {file_path} line {line_num}")
                                    continue
                                
                                # Check if confidence is included (6th element)
                                if len(parts) >= 6:
                                    try:
                                        confidence = float(parts[5])
                                        # Validate confidence range
                                        if 0.0 <= confidence <= 1.0:
                                            confidence_scores.append(confidence)
                                        else:
                                            self.logger.warning(f"Invalid confidence {confidence} in {file_path} line {line_num}")
                                    except (ValueError, IndexError):
                                        self.logger.debug(f"Could not parse confidence from line {line_num} in {file_path}")
                                
                                # Count the class detection
                                class_counts[class_names[cls_id]] += 1
                                detections_in_file += 1
                                
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                                continue
                        else:
                            self.logger.warning(f"Invalid line format in {file_path} line {line_num}: {line}")
            
            self.logger.debug(f"Parsed {detections_in_file} detections from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error parsing detection file {file_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def run_full_pipeline(self, video_path: str = None, force_train: bool = False, 
                 epochs: int = 15, batch_size: int = 4, conf_threshold: float = 0.25):
        """Run the complete pipeline from setup to prediction with better error handling"""
        try:
            self.logger.info("Starting Enhanced Bee Tracking Pipeline")
            
            # Setup environment
            self.setup_environment()
            
            # Check dataset
            self.check_dataset()
            
            # Load or train model
            self.load_or_train_model(force_train=force_train, epochs=epochs, batch_size=batch_size)
            
            # Run prediction if video provided
            if video_path:
                # Handle both file and directory paths
                video_file_path = Path(video_path)
                
                if video_file_path.is_dir():
                    # If it's a directory, look for video files
                    video_files = []
                    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
                    
                    for ext in video_extensions:
                        video_files.extend(list(video_file_path.glob(ext)))
                        # Also check case variations
                        video_files.extend(list(video_file_path.glob(ext.upper())))
                    
                    if not video_files:
                        self.logger.error(f"No video files found in directory: {video_path}")
                        self.logger.info("Supported formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v")
                        
                        # List all files in directory for debugging
                        all_files = list(video_file_path.glob('*'))
                        self.logger.info(f"Files found in directory: {[f.name for f in all_files if f.is_file()]}")
                        
                        return {'model': self.model}
                    
                    # Use the first video file found
                    actual_video_path = str(video_files[0])
                    self.logger.info(f"Found video file: {actual_video_path}")
                else:
                    actual_video_path = video_path
                
                # Check if the video file exists and is accessible
                if os.path.exists(actual_video_path) and os.access(actual_video_path, os.R_OK):
                    # Verify it's actually a video file
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
                    file_ext = Path(actual_video_path).suffix.lower()
                    
                    if file_ext not in video_extensions:
                        self.logger.error(f"File is not a supported video format: {file_ext}")
                        self.logger.info(f"Supported formats: {', '.join(video_extensions)}")
                        return {'model': self.model}
                    
                    results = self.predict_video(actual_video_path, conf_threshold=conf_threshold)
                    analysis = self.analyze_results(results)
                    
                    self.logger.info("Pipeline completed successfully!")
                    return {
                        'model': self.model,
                        'results': results,
                        'analysis': analysis
                    }
                else:
                    self.logger.error(f"Video file not accessible: {actual_video_path}")
                    if not os.path.exists(actual_video_path):
                        self.logger.error("File does not exist")
                    elif not os.access(actual_video_path, os.R_OK):
                        self.logger.error("No read permission for file")
                    
                    self.logger.info("Pipeline setup completed. Ready for predictions.")
                    return {'model': self.model}
            else:
                self.logger.info("Pipeline setup completed. Ready for predictions.")
                return {'model': self.model}
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def export_model(self, format: str = "onnx"):
        """Export trained model to different formats"""
        if self.model is None:
            self.logger.error("No model loaded to export")
            return False
            
        try:
            if self.model_type == "yolov8":
                export_path = self.dirs['weights'] / f"bee_behavior_model.{format}"
                self.model.export(format=format, imgsz=640)
                self.logger.info(f"Model exported to {format} format")
                return True
            else:
                self.logger.warning(f"Export not implemented for {self.model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return False

    def benchmark_model(self, test_images_dir: str):
        """Benchmark model performance on test images"""
        if self.model is None:
            self.logger.error("No model loaded for benchmarking")
            return None
            
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            self.logger.error(f"Test directory not found: {test_dir}")
            return None
            
        try:
            import time
            
            image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if not image_files:
                self.logger.error("No image files found in test directory")
                return None
            
            total_time = 0
            total_detections = 0
            
            self.logger.info(f"Benchmarking on {len(image_files)} images...")
            
            for img_path in image_files[:10]:  # Test on first 10 images
                start_time = time.time()
                
                if self.model_type == "yolov8":
                    results = self.model.predict(str(img_path), verbose=False)
                    if results and results[0].boxes is not None:
                        total_detections += len(results[0].boxes)
                elif self.model_type == "yolov5":
                    results = self.model(str(img_path))
                    total_detections += len(results.xyxy[0])
                
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time = total_time / len(image_files[:10])
            fps = 1 / avg_time if avg_time > 0 else 0
            
            self.logger.info(f"Benchmark Results:")
            self.logger.info(f"Average inference time: {avg_time:.3f}s")
            self.logger.info(f"Average FPS: {fps:.1f}")
            self.logger.info(f"Total detections: {total_detections}")
            
            return {
                'avg_inference_time': avg_time,
                'fps': fps,
                'total_detections': total_detections,
                'images_tested': len(image_files[:10])
            }
            
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {e}")
            return None
        
   
    def cleanup(self):
        """Clean up temporary files and free resources"""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset model
            self.model = None
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Usage example and main execution
if __name__ == "__main__":
    base_path = os.getcwd()
    # Example usage
    pipeline = EnhancedBeeTrackingPipeline(
        base_path = base_path, 
        model_type="yolov7"  # or "yolov5", "yolov7"
    )
    
    test_path = base_path + "/testing"
    # Run full pipeline
    result = pipeline.run_full_pipeline(
        video_path=test_path,  # Optional
        force_train=False,  # Set to True to force retraining
        epochs=15,
        batch_size=4,
        conf_threshold=0.25
    )
    
    if result:
        print("Pipeline completed successfully!")
        if 'analysis' in result and result['analysis']:
            print(f"Total detections: {result['analysis']['total_detections']}")
    pipeline.cleanup()
    
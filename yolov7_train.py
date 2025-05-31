import os
import torch
import yaml
import logging
import shutil
import subprocess
import sys
import requests
from pathlib import Path
import hashlib
import warnings

# Suppress pickle warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")

class EnhancedBeeTrackingPipeline:
    def __init__(self, base_path: str, model_type: str = "yolov7"):
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

        if self.model_type == "yolov7":
            self._setup_yolov7()
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

    # def _download_file_with_retry(self, url, file_path, max_retries=3, chunk_size=8192):
    #     """Download file with retry mechanism and progress tracking"""
    #     for attempt in range(max_retries):
    #         try:
    #             self.logger.info(f"Downloading attempt {attempt + 1}/{max_retries}: {url}")
                
    #             response = requests.get(url, stream=True, timeout=30)
    #             response.raise_for_status()
                
    #             total_size = int(response.headers.get('content-length', 0))
    #             downloaded_size = 0
                
    #             with open(file_path, 'wb') as f:
    #                 for chunk in response.iter_content(chunk_size=chunk_size):
    #                     if chunk:
    #                         f.write(chunk)
    #                         downloaded_size += len(chunk)
                            
    #                         # Progress logging every 10MB
    #                         if downloaded_size % (10 * 1024 * 1024) == 0:
    #                             progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
    #                             self.logger.info(f"Downloaded: {downloaded_size / (1024*1024):.1f}MB ({progress:.1f}%)")
                
    #             # Verify file size
    #             if total_size > 0 and downloaded_size != total_size:
    #                 self.logger.warning(f"Size mismatch: expected {total_size}, got {downloaded_size}")
    #                 if attempt < max_retries - 1:
    #                     continue
                
    #             # Verify file exists and has content
    #             if file_path.exists() and file_path.stat().st_size > 0:
    #                 self.logger.info(f"Download successful: {file_path} ({downloaded_size / (1024*1024):.1f}MB)")
    #                 return True
    #             else:
    #                 self.logger.error(f"Downloaded file is empty or doesn't exist")
                    
    #         except requests.exceptions.RequestException as e:
    #             self.logger.error(f"Download attempt {attempt + 1} failed: {e}")
    #             if file_path.exists():
    #                 file_path.unlink()  # Remove partial download
                    
    #         except Exception as e:
    #             self.logger.error(f"Unexpected error during download: {e}")
    #             if file_path.exists():
    #                 file_path.unlink()
                    
    #         if attempt < max_retries - 1:
    #             self.logger.info("Retrying download in 5 seconds...")
    #             import time
    #             time.sleep(5)
        
    #     return False

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
                
    # def _verify_weights_file(self, weights_path):
    #     """Verify that a weights file is valid and can be loaded"""
    #     try:
    #         if not weights_path.exists() or weights_path.stat().st_size < 1024:  # At least 1KB
    #             return False
            
    #         # Try to load with torch to verify it's a valid PyTorch file
    #         import torch
    #         try:
    #             # Try loading with weights_only=True for newer PyTorch versions
    #             checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    #             return True
    #         except TypeError:
    #             # Fallback for older PyTorch versions
    #             try:
    #                 checkpoint = torch.load(weights_path, map_location='cpu')
    #                 return True
    #             except:
    #                 return False
    #         except:
    #             return False
                
    #     except Exception as e:
    #         self.logger.warning(f"Could not verify weights file {weights_path}: {e}")
    #         return False
        

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

    # def find_trained_models(self):
    #     """Find trained models in runs directory (not pre-trained models)"""
    #     self.logger.info("Searching for trained models...")
        
    #     trained_models = {}
    #     runs_dir = self.base_path / 'runs'
        
    #     if not runs_dir.exists():
    #         self.logger.info("No runs directory found")
    #         return trained_models
        
    #     # Look in train subdirectories
    #     train_dir = runs_dir / 'train'
    #     if train_dir.exists():
    #         for exp_dir in train_dir.iterdir():
    #             if exp_dir.is_dir():
    #                 # Check for weights directory
    #                 weights_dir = exp_dir / 'weights'
    #                 if weights_dir.exists():
    #                     # Look for best.pt or last.pt
    #                     for weight_file in ['best.pt', 'last.pt']:
    #                         weight_path = weights_dir / weight_file
    #                         if weight_path.exists():
    #                             # Determine model type from directory name
    #                             model_type = self._infer_model_type_from_path(exp_dir.name)
    #                             model_name = f"{exp_dir.name}_{weight_file.split('.')[0]}"
                                
    #                             trained_models[model_name] = {
    #                                 'path': str(weight_path),
    #                                 'type': model_type,
    #                                 'size': weight_path.stat().st_size / (1024 * 1024),
    #                                 'last_modified': weight_path.stat().st_mtime,
    #                                 'experiment': exp_dir.name
    #                             }
        
    #     # Sort by last modified time
    #     sorted_models = dict(sorted(
    #         trained_models.items(), 
    #         key=lambda item: item[1]['last_modified'], 
    #         reverse=True
    #     ))
        
    #     if sorted_models:
    #         self.logger.info(f"Found {len(sorted_models)} trained models:")
    #         for name, info in sorted_models.items():
    #             self.logger.info(f"  - {name} ({info['type']}): {info['size']:.2f} MB")
    #     else:
    #         self.logger.info("No trained models found in runs directory")
            
    #     return sorted_models

    # def _infer_model_type_from_path(self, path_name):
    #     """Infer model type from directory/file name"""
    #     path_lower = path_name.lower()
    #     if 'yolov7' in path_lower or 'v7' in path_lower:
    #         return 'yolov7'
    #     return self.model_type  # Default to pipeline's model type

    def load_or_train_model(self, force_train=False, epochs=15, batch_size=4):
        self.train_model(epochs=epochs, batch_size=batch_size)
        return self.model

    def _load_model(self, model_path):
        """Load model based on type with better error handling"""
        try: 
            if self.model_type == "yolov7":
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

        if self.model_type == "yolov7":
            self._train_yolov7_optimized(epochs, batch_size)
            
        self.logger.info("Training complete")

    def _train_yolov7_optimized(self, epochs: int, batch_size: int):
        """Optimized YOLOv7 training with better error handling and model saving"""
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
            use_pretrained = weights_file.exists() and weights_file.stat().st_size > 1024
            
            if not use_pretrained:
                self.logger.warning("No valid pretrained weights available, training from scratch")
            else:
                self.logger.info(f"Using pretrained weights: {weights_file}")
            
            # Training parameters
            img_size = 640
            device = '0' if torch.cuda.is_available() else 'cpu'
            project_dir = self.base_path / 'runs' / 'train'
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create experiment name with timestamp
            import time
            timestamp = int(time.time())
            experiment_name = f'bee_behavior_yolov7_{timestamp}'
            
            # Build training command
            train_script = yolov7_dir / 'train.py'  # Use original train.py
            if not train_script.exists():
                raise FileNotFoundError(f"Training script not found: {train_script}")
            
            cmd_args = [
                sys.executable, str(train_script),
                '--data', str(data_yaml),
                '--img', str(img_size), str(img_size),
                '--batch-size', str(4),
                '--epochs', str(epochs),
                '--device', device,
                '--project', str(project_dir),
                '--name', experiment_name,
                '--exist-ok'
            ]
            
            # Add weights parameter only if file exists
            if use_pretrained:
                cmd_args.extend(['--weights', str(weights_file)])
            else:
                # Use model config for training from scratch
                model_cfg = yolov7_dir / 'cfg' / 'training' / 'yolov7.yaml'
                if model_cfg.exists():
                    cmd_args.extend(['--cfg', str(model_cfg)])
                else:
                    # Look for alternative config files
                    config_files = list((yolov7_dir / 'cfg').glob('**/yolov7*.yaml'))
                    if config_files:
                        cmd_args.extend(['--cfg', str(config_files[0])])
                        self.logger.info(f"Using config file: {config_files[0]}")

            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(yolov7_dir) + os.pathsep + env.get('PYTHONPATH', '')
            
            self.logger.info(f"Training command: {' '.join(cmd_args)}")
            
            # Run training with better error handling
            process = subprocess.Popen(
                cmd_args,
                cwd=yolov7_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Stream output and monitor training progress
            training_started = False
            training_completed = False
            last_epoch = 0
            
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Filter out known warnings that don't affect functionality
                    skip_patterns = [
                        'weights_only',
                        'WeightsUnpickler', 
                        'FutureWarning',
                        'maximum recursion depth exceeded'
                    ]
                    
                    if not any(pattern in line for pattern in skip_patterns):
                        self.logger.info(line)
                        
                    # Check for training progress
                    if 'Epoch' in line and ('/' in line or 'GPU_mem' in line):
                        training_started = True
                        # Extract epoch info
                        try:
                            if 'Epoch' in line:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if 'Epoch' in part and i + 1 < len(parts):
                                        epoch_part = parts[i + 1].split('/')
                                        if len(epoch_part) >= 1:
                                            last_epoch = int(epoch_part[0])
                                        break
                        except:
                            pass
                    
                    # Check for training completion indicators
                    if any(keyword in line.lower() for keyword in ['training complete', 'results saved', 'optimizer stripped']):
                        training_completed = True
            
            process.wait()
            
            # Look for model files
            expected_weights_dir = project_dir / experiment_name / 'weights'
            
            # Check multiple possible locations for weights
            possible_weight_locations = [
                expected_weights_dir / 'best.pt',
                expected_weights_dir / 'last.pt',
                project_dir / experiment_name / 'best.pt',
                project_dir / experiment_name / 'last.pt'
            ]
            
            # Also search in yolov7 runs directory
            yolov7_runs = yolov7_dir / 'runs' / 'train'
            if yolov7_runs.exists():
                for exp_dir in yolov7_runs.iterdir():
                    if exp_dir.is_dir() and experiment_name in exp_dir.name:
                        possible_weight_locations.extend([
                            exp_dir / 'weights' / 'best.pt',
                            exp_dir / 'weights' / 'last.pt'
                        ])
            
            model_found = False
            final_model_path = None
            
            # Find the best available model
            for weight_path in possible_weight_locations:
                if weight_path.exists() and weight_path.stat().st_size > 1024:  # At least 1KB
                    final_model_path = weight_path
                    model_found = True
                    self.logger.info(f"Found model at: {weight_path}")
                    break
            
            # If no specific model found but training started, search broadly
            if not model_found and training_started:
                self.logger.info("Searching for any recent model files...")
                search_dirs = [project_dir, yolov7_dir / 'runs']
                
                current_time = time.time()
                for search_dir in search_dirs:
                    if search_dir.exists():
                        for pt_file in search_dir.rglob('*.pt'):
                            # Skip original pretrained weights
                            if pt_file.name == 'yolov7.pt' and pt_file.parent == yolov7_dir:
                                continue
                            # Check if created recently (within last 2 hours)
                            if (current_time - pt_file.stat().st_mtime < 7200 and 
                                pt_file.stat().st_size > 1024):
                                final_model_path = pt_file
                                model_found = True
                                self.logger.info(f"Found recent model: {pt_file}")
                                break
                    if model_found:
                        break
            
            if model_found and final_model_path:
                self.model = str(final_model_path)
                self.logger.info("YOLOv7 training completed successfully")
                
                # Copy to weights directory
                model_type = 'best' if 'best' in final_model_path.name else 'last'
                target_name = f"{model_type}_yolov7_bee_behavior_{timestamp}.pt"
                target_path = self.dirs['weights'] / target_name
                
                try:
                    shutil.copy(final_model_path, target_path)
                    self.logger.info(f"Model copied to {target_path}")
                    self.model = str(target_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy model: {e}")
                    
            elif training_started:
                self.logger.warning(f"Training made progress (last epoch: {last_epoch}) but no model weights found")
            else:
                self.logger.error("Training failed - no progress detected")
                
        except Exception as e:
            self.logger.error(f"Error during YOLOv7 training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def _create_compatible_train_script(self, yolov7_dir):
        """Create a PyTorch-compatible version of the training script"""
        original_train = yolov7_dir / 'train.py'
        compatible_train = yolov7_dir / 'train_compatible.py'
        
        if compatible_train.exists():
            return  # Already created
        
        try:
            if not original_train.exists():
                self.logger.error("Original train.py not found")
                return
                
            # Read the original training script
            with open(original_train, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add PyTorch compatibility fixes at the beginning
            compatibility_fixes = '''import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*weights_only.*")

    def safe_torch_load(path, map_location='cpu'):
        """Safely load PyTorch models with version compatibility"""
        try:
            # Try with weights_only=False for compatibility
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            try:
                return torch.load(path, map_location=map_location)
            except Exception as e:
                print(f"Warning: Could not load weights from {path}: {e}")
                return None
        except Exception as e:
            print(f"Warning: Could not load weights from {path}: {e}")
            return None

    # Monkey patch torch.load at module level
    import torch
    original_torch_load = torch.load
    torch.load = safe_torch_load

    '''
            
            # Find where imports end (look for the first function or class definition)
            import_patterns = ['\ndef ', '\nclass ', '\nif __name__']
            insert_position = len(content)
            
            for pattern in import_patterns:
                pos = content.find(pattern)
                if pos != -1 and pos < insert_position:
                    insert_position = pos
            
            # Insert compatibility fixes after imports
            modified_content = content[:insert_position] + '\n' + compatibility_fixes + '\n' + content[insert_position:]
            
            # Write the compatible version
            with open(compatible_train, 'w', encoding='utf-8') as f:
                f.write(modified_content)
                
            self.logger.info("Created PyTorch-compatible training script")
            
        except Exception as e:
            self.logger.warning(f"Could not create compatible training script: {e}")
            # Fall back to using original script
            if not compatible_train.exists():
                try:
                    shutil.copy(original_train, compatible_train)
                except:
                    pass
                
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
            if self.model_type == "yolov7":
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
        
    def analyze_results(self, results):
        """Analyze prediction results"""
        if results is None:
            self.logger.error("No results to analyze")
            return None
            
        class_names = ['foraging', 'defense', 'fanning', 'washboarding']
        class_counts = {name: 0 for name in class_names}
        confidence_scores = []
        
        try:    
            if self.model_type == "yolov7":
                # Parse YOLOv7 results from saved detection files
                total_detections = self._parse_yolov7_saved_results(class_counts, confidence_scores)
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                self.logger.info(f"YOLOv7 Analysis Results:")
                self.logger.info(f"Total detections: {total_detections}")
                self.logger.info(f"Average confidence: {avg_confidence:.3f}")
                self.logger.info("Class distribution:")
                for class_name, count in class_counts.items():
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
                    
            return {
                'total_detections': sum(class_counts.values()),
                'class_counts': class_counts,
                'average_confidence': avg_confidence if confidence_scores else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _parse_yolov7_saved_results(self, class_counts, confidence_scores):
        """Parse YOLOv7 saved detection results from output directory"""
        total_detections = 0
        try:
            # Look for saved results in the output directory
            output_dirs = list(self.dirs['output'].glob("*"))
            
            for output_dir in output_dirs:
                if output_dir.is_dir():
                    # YOLOv7 saves detection files in labels subdirectory
                    labels_dir = output_dir / 'labels'
                    if labels_dir.exists():
                        for txt_file in labels_dir.glob('*.txt'):
                            detections = self._parse_detection_file(txt_file, class_counts, confidence_scores)
                            total_detections += detections
                    else:
                        # Also check directly in output directory
                        for txt_file in output_dir.glob('*.txt'):
                            if txt_file.name not in ['results.txt', 'summary.txt']:
                                detections = self._parse_detection_file(txt_file, class_counts, confidence_scores)
                                total_detections += detections
                                
        except Exception as e:
            self.logger.warning(f"Could not parse YOLOv7 saved results: {e}")
        
        return total_detections

    def _parse_detection_file(self, file_path, class_counts, confidence_scores):
        """Parse individual detection file in YOLO format"""
        detections_count = 0
        try:
            class_names = ['foraging', 'defense', 'fanning', 'washboarding']
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:  # class_id x_center y_center width height [confidence]
                            try:
                                cls_id = int(parts[0])
                                
                                # Check if confidence is included
                                if len(parts) >= 6:
                                    try:
                                        confidence = float(parts[5])
                                        confidence_scores.append(confidence)
                                    except (ValueError, IndexError):
                                        pass
                                
                                # Count the class
                                if 0 <= cls_id < len(class_names):
                                    class_counts[class_names[cls_id]] += 1
                                    detections_count += 1
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Invalid detection format in {file_path}: {line}")
                                continue
                                
        except Exception as e:
            self.logger.warning(f"Error parsing detection file {file_path}: {e}")
        
        return detections_count

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
                        if not video_files:
                            self.logger.error(f"No video files found in directory: {video_path}")
                        return None
                    
                    self.logger.info(f"Found {len(video_files)} video files in directory")
                    
                    # Process each video file
                    all_results = []
                    for video_file in video_files:
                        self.logger.info(f"Processing video: {video_file.name}")
                        results = self.predict_video(str(video_file), conf_threshold=conf_threshold)
                        if results:
                            analysis = self.analyze_results(results)
                            all_results.append({
                                'video': video_file.name,
                                'results': results,
                                'analysis': analysis
                            })
                    
                    return all_results
                
                elif video_file_path.is_file():
                    # Single video file
                    results = self.predict_video(video_path, conf_threshold=conf_threshold)
                    if results:
                        analysis = self.analyze_results(results)
                        return {
                            'video': video_file_path.name,
                            'results': results,
                            'analysis': analysis
                        }
                    else:
                        self.logger.error("Prediction failed")
                        return None
                else:
                    self.logger.error(f"Invalid video path: {video_path}")
                    return None
            else:
                self.logger.info("No video path provided. Pipeline setup complete.")
                return {"status": "setup_complete", "model_loaded": self.model is not None}
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def cleanup(self):
        """Cleanup resources and temporary files"""
        try:
            self.logger.info("Cleaning up resources...")
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared")
            
            # Clean up temporary files if any
            temp_dirs = [self.base_path / 'temp', self.base_path / '.temp']
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Removed temporary directory: {temp_dir}")
            
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
            
    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": self.model_type,
            "model_loaded": True
        }
        
        if self.model_type == "yolov7" and isinstance(self.model, str):
            model_path = Path(self.model)
            if model_path.exists():
                info.update({
                    "model_path": str(model_path),
                    "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                    "last_modified": model_path.stat().st_mtime
                })
        
        return info
    
    def validate_environment(self):
        """Validate that the environment is properly set up"""
        self.logger.info("Validating environment...")
        
        validation_results = {
            "directories": True,
            "dependencies": True,
            "dataset": True,
            "cuda": torch.cuda.is_available(),
            "model_framework": True
        }
        
        # Check directories
        try:
            required_dirs = ['weights', 'results', 'output', 'runs']
            for dir_name in required_dirs:
                if not (self.base_path / dir_name).exists():
                    validation_results["directories"] = False
                    self.logger.error(f"Missing directory: {dir_name}")
        except Exception as e:
            validation_results["directories"] = False
            self.logger.error(f"Directory validation error: {e}")
        
        # Check dataset
        try:
            self.check_dataset()
        except Exception as e:
            validation_results["dataset"] = False
            self.logger.error(f"Dataset validation failed: {e}")
        
        # Check model framework dependencies
        try:
            if self.model_type == "yolov7":
                yolov7_dir = self.base_path / 'yolov7'
                if not yolov7_dir.exists():
                    validation_results["model_framework"] = False
                    self.logger.error("YOLOv7 directory not found")
        except Exception as e:
            validation_results["model_framework"] = False
            self.logger.error(f"Model framework validation error: {e}")
        
        # Summary
        all_valid = all(validation_results.values())
        self.logger.info(f"Environment validation {'PASSED' if all_valid else 'FAILED'}")
        
        for component, status in validation_results.items():
            status_str = "✓" if status else "✗"
            self.logger.info(f"  {component}: {status_str}")
        
        return validation_results
    
def main():
    """Main execution function with example usage"""
    base_path = os.getcwd()
    # Initialize pipeline
    pipeline = EnhancedBeeTrackingPipeline(
        base_path=base_path,
        model_type="yolov7"
    )
    test_path = base_path + "/testing"
    try:
        # Validate environment first
        # validation = pipeline.validate_environment()
        # if not all(validation.values()):
        #     print("Environment validation failed. Please check the logs.")
        #     return
        
        # Run full pipeline
        # Option 1: Train new model and predict on single video
        results = pipeline.run_full_pipeline(
            video_path=test_path,
            force_train=False,  # Set to True to force new training
            epochs=15,
            batch_size=8,
            conf_threshold=0.25
        )
        
        # Option 2: Just setup and train (no prediction)
        # results = pipeline.run_full_pipeline(force_train=True, epochs=20)
        
        # Option 3: Process multiple videos in a directory
        # results = pipeline.run_full_pipeline(video_path="path/to/video/directory/")
        
        if results:
            print("Pipeline completed successfully!")
            print(f"Results: {results}")
        else:
            print("Pipeline failed. Check logs for details.")
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        # Cleanup
        pipeline.cleanup()

if __name__ == "__main__":
    main()
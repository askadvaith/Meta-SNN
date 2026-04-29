"""
Unified Data Loader for Meta-Learning with Olfactory E-Nose Datasets

Handles heterogeneous datasets:
- Beef: Excel with 12 sheets (cuts), 10 MQ sensors, 4 freshness classes
- Seafood: Excel with 4 sheets (species), 4 MQ sensors + env, 2 classes (accept/reject)
- Chinese Vinegar: Excel files in 6 brand folders, 8 sensors, 6 classes
- Wine: TXT files in 3 quality folders, 6 MQ sensors (removing temp/humidity)

All data is converted to a unified format:
- Time series: [samples, timesteps, features]
- Features: Zero-padded to max_features (10 for this dataset collection)
- Labels: Integer encoded per task
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TaskData:
    """Container for a single task's data"""
    name: str
    domain: str
    X: np.ndarray  # [samples, timesteps, features]
    y: np.ndarray  # [samples]
    class_names: List[str]
    feature_names: List[str]
    num_classes: int
    metadata: Dict[str, Any]


class UnifiedOlfactoryDataLoader:
    """
    Unified data loader for multiple olfactory datasets.
    Converts all formats to a common representation suitable for meta-learning.
    
    Supports adaptive window selection by storing original lengths and allowing
    reloading with task-specific window sizes.
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 500,
        downsample_factor: int = 2,
        max_features: int = 10,  # Max number of sensor features across all datasets
        stabilization_period: int = 100,  # Initial timesteps to skip
        normalize: bool = True,
        random_state: int = 42,
        adaptive_windows: Dict[str, int] = None  # Task-specific window sizes
    ):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing all dataset folders
            sequence_length: Target sequence length after downsampling
            downsample_factor: Factor to downsample time series
            max_features: Maximum features (for zero-padding smaller datasets)
            stabilization_period: Initial timesteps to skip (sensor stabilization)
            normalize: Whether to normalize features to [0, 1]
            random_state: Random seed for reproducibility
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.max_features = max_features
        self.stabilization_period = stabilization_period
        self.normalize = normalize
        self.random_state = random_state
        self.adaptive_windows = adaptive_windows or {}  # Task-specific window sizes
        
        # Cache for raw sequences (before windowing) for adaptive window support
        self._raw_sequences_cache: Dict[str, List[np.ndarray]] = {}
        
        np.random.seed(random_state)
        
        # Dataset paths
        self.beef_path = os.path.join(data_root, 'beef', 'e-nose_dataset_12_beef_cuts.xlsx')
        self.seafood_path = os.path.join(data_root, 'seafood', 'seafood_dataset.xlsx')
        self.vinegar_path = os.path.join(data_root, 'chinese_vinegar')
        self.wine_path = os.path.join(data_root, 'wine')
        
        # Vinegar sensor order (from readme.txt)
        self.vinegar_sensors = ['MQ136', 'MQ9B', 'MQ7B', 'MQ2', 'MQ8', 'MQ138', 'MQ137', 'MQ5']
        
        # Wine sensor columns (excluding temp/humidity as requested)
        self.wine_sensors = [
            'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
            'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
        ]
        
    def load_all_tasks(self) -> Dict[str, List[TaskData]]:
        """
        Load all datasets and organize as meta-learning tasks.
        
        Returns:
            Dictionary mapping domain names to lists of TaskData objects
        """
        tasks = {
            'beef': self._load_beef_tasks(),
            'seafood': self._load_seafood_tasks(),
            'vinegar': self._load_vinegar_tasks(),
            'wine': self._load_wine_tasks()
        }
        
        return tasks
    
    def _process_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Process a single time series sequence:
        1. Skip stabilization period
        2. Downsample
        3. Pad/truncate to target length
        4. Pad features to max_features
        
        Args:
            sequence: Raw time series [timesteps, features]
            
        Returns:
            Processed sequence [sequence_length, max_features]
        """
        # Skip stabilization period
        if len(sequence) > self.stabilization_period:
            sequence = sequence[self.stabilization_period:]
        
        # Downsample
        if self.downsample_factor > 1:
            sequence = sequence[::self.downsample_factor]
        
        # Pad or truncate to target length
        if len(sequence) < self.sequence_length:
            # Pad by repeating last value
            padding = np.repeat(sequence[-1:], self.sequence_length - len(sequence), axis=0)
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        
        # Pad features to max_features (zero-padding)
        if sequence.shape[1] < self.max_features:
            padding = np.zeros((sequence.shape[0], self.max_features - sequence.shape[1]))
            sequence = np.hstack([sequence, padding])
        
        return sequence
    
    def _normalize_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize all sequences using MinMax scaling"""
        if not self.normalize:
            return sequences
        
        # Stack all data for fitting scaler
        all_data = np.vstack(sequences)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_data)
        
        # Transform each sequence
        normalized = []
        for seq in sequences:
            norm_seq = scaler.transform(seq)
            normalized.append(norm_seq)
        
        return normalized
    
    def _load_beef_tasks(self) -> List[TaskData]:
        """
        Load beef freshness dataset.
        Creates one task per beef cut (sheet), each with 4 freshness classes.
        
        Beef Labels:
        - 1: Excellent
        - 2: Good  
        - 3: Acceptable
        - 4: Spoiled
        """
        tasks = []
        
        if not os.path.exists(self.beef_path):
            print(f"Warning: Beef dataset not found at {self.beef_path}")
            return tasks
        
        xls = pd.ExcelFile(self.beef_path)
        
        # Sensor columns in the beef dataset
        sensor_cols = ['MQ135', 'MQ136', 'MQ137', 'MQ138', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ8', 'MQ9']
        # Use only the first 10 (max_features)
        sensor_cols = sensor_cols[:self.max_features]
        
        class_names = ['Excellent', 'Good', 'Acceptable', 'Spoiled']
        
        # Beef-specific window parameters:
        # - 60-minute windows to capture meaningful sensor patterns
        # - 15-minute stride (75% overlap) for more samples
        # - No stabilization period skip (data already starts at minute 1)
        beef_window_size = 60  # 60 minutes per window
        beef_stride = 15  # 75% overlap, generates ~144 samples per cut
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Filter to only use available sensor columns
            available_sensors = [col for col in sensor_cols if col in df.columns]
            
            sequences = []
            labels = []
            
            # Get raw time series
            sensor_data = df[available_sensors].values
            label_data = df['Label'].values
            
            # Apply rolling mean smoothing (window size 5, min_periods 1)
            df_sensors = df[available_sensors].copy()
            df_smoothed = df_sensors.rolling(window=5, min_periods=1).mean()
            sensor_data = df_smoothed.values
            
            # Create sliding windows with beef-specific parameters
            for start in range(0, len(sensor_data) - beef_window_size, beef_stride):
                end = start + beef_window_size
                window = sensor_data[start:end]
                label = int(label_data[end - 1])  # Label at end of window
                
                # Beef-specific processing: direct slicing without stabilization skip
                # Pad features to max_features if needed
                if window.shape[1] < self.max_features:
                    padding = np.zeros((window.shape[0], self.max_features - window.shape[1]))
                    window = np.hstack([window, padding])
                
                sequences.append(window)
                labels.append(label - 1)  # Convert 1-4 to 0-3
            
            if len(sequences) == 0:
                continue
            
            # Normalize
            sequences = self._normalize_sequences(sequences)
            
            X = np.stack(sequences)
            y = np.array(labels)
            
            # Log class distribution
            class_counts = np.bincount(y, minlength=len(class_names))
            dist_str = ", ".join([f"{class_names[i]}:{c}" for i, c in enumerate(class_counts)])
            print(f"    Class distribution: {dist_str}")
            
            # Create feature names (padded)
            feature_names = available_sensors + ['pad_' + str(i) for i in range(self.max_features - len(available_sensors))]
            
            task = TaskData(
                name=f"beef_{sheet_name.replace('.', '_').replace(' ', '_')}",
                domain='beef',
                X=X,
                y=y,
                class_names=class_names,
                feature_names=feature_names,
                num_classes=len(class_names),
                metadata={
                    'cut_name': sheet_name,
                    'original_sensors': available_sensors,
                    'window_size': beef_window_size,
                    'stride': beef_stride
                }
            )
            tasks.append(task)
            print(f"  Loaded beef task: {task.name} - {len(X)} samples, {len(np.unique(y))} classes")
        
        return tasks
    
    def _load_seafood_tasks(self) -> List[TaskData]:
        """
        Load seafood freshness dataset.
        Creates one task per seafood type (sheet), each with 2 classes (accept/reject).
        """
        tasks = []
        
        if not os.path.exists(self.seafood_path):
            print(f"Warning: Seafood dataset not found at {self.seafood_path}")
            return tasks
        
        xls = pd.ExcelFile(self.seafood_path)
        
        # Sensor columns (excluding humidity and temperature as per user request for wine,
        # we keep them here for seafood as they weren't mentioned)
        sensor_cols = ['MQ136', 'MQ137', 'MQ5', 'MQ8']
        
        class_names = ['Accept', 'Reject']
        label_map = {'accept': 0, 'reject': 1}
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Filter to sensor columns
            available_sensors = [col for col in sensor_cols if col in df.columns]
            
            # Seafood data has Duration column indicating time
            # Group by sample: Each continuous duration sequence is one sample
            sequences = []
            labels = []
            
            # The data appears to be one long time series
            # We need to split it into samples based on label changes or fixed windows
            sensor_data = df[available_sensors].values
            label_data = df['Label'].values
            
            # Create samples using sliding windows
            window_size = self.sequence_length * self.downsample_factor + self.stabilization_period
            stride = window_size // 2
            
            for start in range(0, len(sensor_data) - window_size, stride):
                end = start + window_size
                window = sensor_data[start:end]
                label_str = label_data[end - 1]
                
                if label_str not in label_map:
                    continue
                
                processed = self._process_sequence(window)
                sequences.append(processed)
                labels.append(label_map[label_str])
            
            if len(sequences) == 0:
                continue
            
            # Normalize
            sequences = self._normalize_sequences(sequences)
            
            X = np.stack(sequences)
            y = np.array(labels)
            
            # Create feature names (padded)
            feature_names = available_sensors + ['pad_' + str(i) for i in range(self.max_features - len(available_sensors))]
            
            task = TaskData(
                name=f"seafood_{sheet_name.replace(' ', '_')}",
                domain='seafood',
                X=X,
                y=y,
                class_names=class_names,
                feature_names=feature_names,
                num_classes=len(class_names),
                metadata={
                    'species': sheet_name,
                    'original_sensors': available_sensors
                }
            )
            tasks.append(task)
            print(f"  Loaded seafood task: {task.name} - {len(X)} samples, {len(np.unique(y))} classes")
        
        return tasks
    
    def _load_vinegar_tasks(self) -> List[TaskData]:
        """
        Load Chinese vinegar dataset.
        Creates one multi-class classification task (6 vinegar brands).
        Each brand folder contains 25 sample files.
        """
        tasks = []
        
        if not os.path.exists(self.vinegar_path):
            print(f"Warning: Vinegar dataset not found at {self.vinegar_path}")
            return tasks
        
        # Brand folders
        brand_folders = ['JSHS', 'SCBN', 'TJTL', 'SXLF', 'LNGQ', 'SXDH']
        class_names = brand_folders
        
        sequences = []
        labels = []
        
        for brand_idx, brand in enumerate(brand_folders):
            brand_path = os.path.join(self.vinegar_path, brand)
            
            if not os.path.exists(brand_path):
                continue
            
            # Load all sample files
            sample_files = [f for f in os.listdir(brand_path) 
                          if f.endswith('.xlsx') and not f.startswith('~')]
            
            for sample_file in sample_files:
                file_path = os.path.join(brand_path, sample_file)
                
                try:
                    # No header in vinegar files
                    df = pd.read_excel(file_path, header=None)
                    
                    # Ensure we have 8 sensor columns
                    if df.shape[1] < 8:
                        continue
                    
                    # Use first 8 columns (sensors)
                    sensor_data = df.iloc[:, :8].values
                    
                    processed = self._process_sequence(sensor_data)
                    sequences.append(processed)
                    labels.append(brand_idx)
                    
                except Exception as e:
                    print(f"    Warning: Could not load {file_path}: {e}")
                    continue
        
        if len(sequences) == 0:
            return tasks
        
        # Normalize
        sequences = self._normalize_sequences(sequences)
        
        X = np.stack(sequences)
        y = np.array(labels)
        
        # Create feature names (padded)
        feature_names = self.vinegar_sensors + ['pad_' + str(i) for i in range(self.max_features - len(self.vinegar_sensors))]
        
        task = TaskData(
            name='vinegar_brand_classification',
            domain='vinegar',
            X=X,
            y=y,
            class_names=class_names,
            feature_names=feature_names,
            num_classes=len(class_names),
            metadata={
                'brands': brand_folders,
                'samples_per_brand': 25,
                'original_sensors': self.vinegar_sensors
            }
        )
        tasks.append(task)
        print(f"  Loaded vinegar task: {task.name} - {len(X)} samples, {len(np.unique(y))} classes")
        
        return tasks
    
    def _load_wine_tasks(self) -> List[TaskData]:
        """
        Load wine quality dataset.
        Creates one task for wine quality classification (HQ, LQ, AQ).
        Temperature and Humidity columns are excluded as requested.
        """
        tasks = []
        
        if not os.path.exists(self.wine_path):
            print(f"Warning: Wine dataset not found at {self.wine_path}")
            return tasks
        
        # Quality folders (labels derived from folder names)
        quality_folders = {
            'HQ_Wines': 'HQ',
            'LQ_Wines': 'LQ', 
            'AQ_Wines': 'AQ'
        }
        class_names = ['AQ', 'HQ', 'LQ']  # Alphabetical order
        label_map = {v: idx for idx, v in enumerate(class_names)}
        
        sequences = []
        labels = []
        
        for folder_name, quality_label in quality_folders.items():
            folder_path = os.path.join(self.wine_path, folder_name)
            
            if not os.path.exists(folder_path):
                continue
            
            # Load all txt files
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            for txt_file in txt_files:
                file_path = os.path.join(folder_path, txt_file)
                
                try:
                    # Wine files are tab-separated with no header
                    # Columns: Humidity, Temperature, MQ-3_R1, MQ-4_R1, MQ-6_R1, MQ-3_R2, MQ-4_R2, MQ-6_R2
                    df = pd.read_csv(file_path, sep='\t', header=None)
                    
                    # Skip first 2 columns (Humidity, Temperature) as requested
                    sensor_data = df.iloc[:, 2:].values
                    
                    processed = self._process_sequence(sensor_data)
                    sequences.append(processed)
                    labels.append(label_map[quality_label])
                    
                except Exception as e:
                    print(f"    Warning: Could not load {file_path}: {e}")
                    continue
        
        if len(sequences) == 0:
            return tasks
        
        # Normalize
        sequences = self._normalize_sequences(sequences)
        
        X = np.stack(sequences)
        y = np.array(labels)
        
        # Create feature names (padded) - 6 sensors, pad to 10
        sensor_names = ['MQ3_R1', 'MQ4_R1', 'MQ6_R1', 'MQ3_R2', 'MQ4_R2', 'MQ6_R2']
        feature_names = sensor_names + ['pad_' + str(i) for i in range(self.max_features - len(sensor_names))]
        
        task = TaskData(
            name='wine_quality_classification',
            domain='wine',
            X=X,
            y=y,
            class_names=class_names,
            feature_names=feature_names,
            num_classes=len(class_names),
            metadata={
                'quality_labels': list(quality_folders.values()),
                'original_sensors': sensor_names,
                'class_distribution': {class_names[i]: int(c) for i, c in enumerate(np.bincount(y))}
            }
        )
        tasks.append(task)
        
        # Log class distribution to help diagnose imbalance issues
        class_dist = np.bincount(y)
        dist_str = ", ".join([f"{class_names[i]}:{c}" for i, c in enumerate(class_dist)])
        print(f"  Loaded wine task: {task.name} - {len(X)} samples, {len(np.unique(y))} classes")
        print(f"    Class distribution: {dist_str}")
        
        return tasks


def create_meta_learning_tasks(
    data_root: str,
    sequence_length: int = 500,
    downsample_factor: int = 2,
    adaptive_windows: Dict[str, int] = None,
    **kwargs
) -> Tuple[List[TaskData], Dict[str, List[TaskData]]]:
    """
    Convenience function to load all tasks for meta-learning.
    
    Args:
        data_root: Root directory containing datasets
        sequence_length: Default sequence length (used if no adaptive window specified)
        downsample_factor: Downsampling factor for time series
        adaptive_windows: Optional dict mapping task names to specific window sizes
        **kwargs: Additional arguments for UnifiedOlfactoryDataLoader
    
    Returns:
        all_tasks: Flat list of all TaskData objects
        tasks_by_domain: Dictionary mapping domain to list of tasks
    """
    loader = UnifiedOlfactoryDataLoader(
        data_root=data_root,
        sequence_length=sequence_length,
        downsample_factor=downsample_factor,
        adaptive_windows=adaptive_windows,
        **kwargs
    )
    
    print("Loading olfactory datasets...")
    if adaptive_windows:
        print(f"Using adaptive windows for {len(adaptive_windows)} tasks")
    print("=" * 60)
    
    tasks_by_domain = loader.load_all_tasks()
    
    # Flatten to single list
    all_tasks = []
    for domain_tasks in tasks_by_domain.values():
        all_tasks.extend(domain_tasks)
    
    print("=" * 60)
    print(f"Total tasks loaded: {len(all_tasks)}")
    
    # Print summary
    for domain, domain_tasks in tasks_by_domain.items():
        total_samples = sum(t.X.shape[0] for t in domain_tasks)
        print(f"  {domain}: {len(domain_tasks)} tasks, {total_samples} total samples")
    
    return all_tasks, tasks_by_domain


def apply_adaptive_windows(
    all_tasks: List[TaskData],
    optimal_windows: Dict[str, int]
) -> List[TaskData]:
    """
    Apply task-specific window sizes to loaded tasks.
    
    This is useful when you want to:
    1. Load tasks with default window first
    2. Run window optimization
    3. Apply optimal windows to the tasks
    
    Args:
        all_tasks: List of TaskData objects with default windows
        optimal_windows: Dict mapping task names to optimal window sizes
        
    Returns:
        List of TaskData objects with updated window sizes
    """
    updated_tasks = []
    
    for task in all_tasks:
        if task.name in optimal_windows:
            target_window = optimal_windows[task.name]
            current_length = task.X.shape[1]
            
            if target_window == current_length:
                # No change needed
                updated_tasks.append(task)
            else:
                # Apply new window
                X = task.X
                if target_window < current_length:
                    # Truncate
                    X_new = X[:, :target_window, :]
                else:
                    # Pad by repeating last value
                    padding_length = target_window - current_length
                    padding = np.repeat(X[:, -1:, :], padding_length, axis=1)
                    X_new = np.concatenate([X, padding], axis=1)
                
                # Create updated task
                updated_task = TaskData(
                    name=task.name,
                    domain=task.domain,
                    X=X_new,
                    y=task.y,
                    class_names=task.class_names,
                    feature_names=task.feature_names,
                    num_classes=task.num_classes,
                    metadata={
                        **task.metadata,
                        'original_window': current_length,
                        'adaptive_window': target_window
                    }
                )
                updated_tasks.append(updated_task)
                print(f"  Applied window {target_window} to {task.name} (was {current_length})")
        else:
            updated_tasks.append(task)
    
    return updated_tasks


if __name__ == "__main__":
    # Test the data loader
    import sys
    
    # Get data root from command line or use default
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    all_tasks, tasks_by_domain = create_meta_learning_tasks(
        data_root=data_root,
        sequence_length=500,
        downsample_factor=2
    )
    
    # Print detailed info about each task
    print("\n" + "=" * 60)
    print("TASK DETAILS")
    print("=" * 60)
    
    for task in all_tasks:
        print(f"\nTask: {task.name}")
        print(f"  Domain: {task.domain}")
        print(f"  Shape: {task.X.shape}")
        print(f"  Classes: {task.num_classes} - {task.class_names}")
        print(f"  Features: {task.feature_names[:task.X.shape[2]]}")
        print(f"  Label distribution: {np.bincount(task.y)}")

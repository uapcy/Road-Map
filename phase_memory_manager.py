"""
phase_memory_manager.py
Memory-efficient batch processing for 3D tomography.
"""

import numpy as np
import os
import tempfile
import warnings
from typing import List, Tuple, Dict, Any, Optional, Generator

class MemoryManager:
    """
    Manages memory-efficient processing of large 3D tomography datasets.
    """
    
    def __init__(self, max_memory_gb: float = 2.0, temp_dir: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            max_memory_gb: Maximum memory to use in GB
            temp_dir: Directory for temporary files (None = system temp)
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        self.temp_dir = temp_dir
        self.temp_files = []
        
        # Estimate available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            self.available_memory_bytes = min(available_memory * 0.8, self.max_memory_bytes)
        except ImportError:
            # Conservative estimate if psutil not available
            self.available_memory_bytes = self.max_memory_bytes
            warnings.warn("psutil not installed. Using conservative memory estimates.")
    
    def __del__(self):
        """Clean up temporary files."""
        self.cleanup()
    
    def cleanup(self):
        """Remove all temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files = []
    
    def estimate_batch_size(self, data_shape: Tuple[int, ...], dtype=np.complex64) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            data_shape: Shape of full dataset (e.g., (rows, cols, looks))
            dtype: Data type
            
        Returns:
            int: Recommended batch size (number of columns)
        """
        # Size of one element in bytes
        if dtype == np.complex64:
            element_size = 8  # 4 bytes per float, 2 floats per complex
        elif dtype == np.float32:
            element_size = 4
        elif dtype == np.float64:
            element_size = 8
        else:
            element_size = 8  # Default
        
        # Memory for one full column
        rows, cols, looks = data_shape
        column_memory = rows * looks * element_size
        
        # Memory for output tomogram (per column)
        # Assuming depth dimension similar to rows
        output_memory = rows * rows * element_size  # Rough estimate
        
        # Total memory per column
        total_per_column = column_memory + output_memory
        
        # Add overhead (50%)
        total_per_column *= 1.5
        
        # Calculate batch size
        batch_size = max(1, int(self.available_memory_bytes // total_per_column))
        
        # Don't exceed number of columns
        batch_size = min(batch_size, cols)
        
        # Ensure minimum batch size for efficiency
        batch_size = max(batch_size, min(10, cols))
        
        print(f"[MEMORY MANAGER] Estimated batch size: {batch_size} columns "
              f"(available: {self.available_memory_bytes/1024**3:.1f} GB, "
              f"per column: {total_per_column/1024**3:.3f} GB)")
        
        return batch_size
    
    def save_to_disk(self, data: np.ndarray, filename: str, 
                    group: str = 'data', dataset: str = 'array') -> str:
        """
        Save array to disk (numpy format).
        
        Args:
            data: Array to save
            filename: Output filename
            group: For compatibility with h5py API (ignored)
            dataset: For compatibility with h5py API (ignored)
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.temp_dir, filename)
        
        # Save with numpy
        np.save(filepath, data)
        
        self.temp_files.append(filepath)
        return filepath
    
    def load_from_disk(self, filepath: str, group: str = 'data', 
                      dataset: str = 'array') -> np.ndarray:
        """
        Load array from disk.
        
        Args:
            filepath: Path to file
            group: For compatibility with h5py API (ignored)
            dataset: For compatibility with h5py API (ignored)
            
        Returns:
            np.ndarray: Loaded array
        """
        # Load with numpy
        data = np.load(filepath, allow_pickle=True)
        return data
    
    def process_in_batches(self, data_3d: np.ndarray, 
                          processing_func: callable,
                          batch_size: Optional[int] = None,
                          dimension: int = 1,
                          **processing_kwargs) -> np.ndarray:
        """
        Process 3D data in memory-efficient batches.
        
        Args:
            data_3d: 3D input data [rows, cols, looks]
            processing_func: Function to apply to each batch
            batch_size: Number of columns per batch (None = auto)
            dimension: Dimension to batch over (1 = columns)
            **processing_kwargs: Arguments to pass to processing_func
            
        Returns:
            np.ndarray: Processed 3D data
        """
        rows, cols, looks = data_3d.shape
        
        if batch_size is None:
            batch_size = self.estimate_batch_size(data_3d.shape, data_3d.dtype)
        
        # Initialize output array
        # We don't know output shape yet, so we'll collect results
        results = []
        column_indices = []
        
        # Process in batches
        for batch_start in range(0, cols, batch_size):
            batch_end = min(batch_start + batch_size, cols)
            batch_cols = list(range(batch_start, batch_end))
            
            print(f"[BATCH PROCESSING] Processing columns {batch_start}-{batch_end-1} "
                  f"({len(batch_cols)} columns)...")
            
            # Extract batch
            if dimension == 1:  # Batch over columns
                batch_data = data_3d[:, batch_start:batch_end, :]
            else:
                raise ValueError(f"Batching over dimension {dimension} not implemented")
            
            # Process batch
            batch_result = processing_func(batch_data, **processing_kwargs)
            
            # Store results
            results.append(batch_result)
            column_indices.extend(batch_cols)
        
        # Combine results
        # Determine output shape from first batch
        if results:
            # Assuming output is [rows, depth] per column
            if results[0].ndim == 2:  # [batch_cols, depth]
                output_rows = results[0].shape[1]
                output = np.zeros((cols, output_rows), dtype=results[0].dtype)
                
                # Fill output
                batch_start = 0
                for batch_result in results:
                    batch_cols = batch_result.shape[0]
                    output[batch_start:batch_start+batch_cols, :] = batch_result
                    batch_start += batch_cols
                    
                # Transpose to [rows, cols, depth] if needed
                output = output.T  # [depth, cols]
                output = output[np.newaxis, :, :]  # [1, depth, cols]
                
            elif results[0].ndim == 3:  # [rows, batch_cols, depth]
                output_rows, _, output_depth = results[0].shape
                output = np.zeros((output_rows, cols, output_depth), dtype=results[0].dtype)
                
                # Fill output
                col_idx = 0
                for batch_result in results:
                    batch_cols = batch_result.shape[1]
                    output[:, col_idx:col_idx+batch_cols, :] = batch_result
                    col_idx += batch_cols
            else:
                raise ValueError(f"Unexpected result shape: {results[0].shape}")
            
            return output
        else:
            return np.array([])
    
    def batch_iterator(self, data_3d: np.ndarray, 
                      batch_size: Optional[int] = None,
                      dimension: int = 1) -> Generator[Tuple[np.ndarray, List[int]], None, None]:
        """
        Iterate over 3D data in batches (generator version).
        
        Args:
            data_3d: 3D input data
            batch_size: Number of columns per batch
            dimension: Dimension to batch over
            
        Yields:
            tuple: (batch_data, column_indices)
        """
        rows, cols, looks = data_3d.shape
        
        if batch_size is None:
            batch_size = self.estimate_batch_size(data_3d.shape, data_3d.dtype)
        
        for batch_start in range(0, cols, batch_size):
            batch_end = min(batch_start + batch_size, cols)
            batch_cols = list(range(batch_start, batch_end))
            
            # Extract batch
            if dimension == 1:  # Batch over columns
                batch_data = data_3d[:, batch_start:batch_end, :]
            else:
                raise ValueError(f"Batching over dimension {dimension} not implemented")
            
            yield batch_data, batch_cols
    
    def save_intermediate_results(self, results: Dict[str, np.ndarray], 
                                prefix: str = "batch_result") -> str:
        """
        Save intermediate batch results to disk.
        
        Args:
            results: Dictionary of result arrays
            prefix: Prefix for filename
            
        Returns:
            str: Path to saved file
        """
        import time
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.npz"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Save with numpy
        np.savez_compressed(filepath, **results)
        
        self.temp_files.append(filepath)
        return filepath
    
    def merge_batch_results(self, result_files: List[str], 
                          output_shape: Tuple[int, ...],
                          output_key: str = 'tomogram') -> np.ndarray:
        """
        Merge results from multiple batch files.
        
        Args:
            result_files: List of numpy files with batch results
            output_shape: Expected shape of final output
            output_key: Key for the output array in numpy files
            
        Returns:
            np.ndarray: Merged result
        """
        output = np.zeros(output_shape, dtype=np.complex64)
        current_col = 0
        
        for filepath in result_files:
            data = np.load(filepath, allow_pickle=True)
            
            if output_key in data:
                batch_data = data[output_key]
                batch_cols = batch_data.shape[1] if batch_data.ndim >= 2 else 1
                
                if batch_data.ndim == 2:  # [depth, cols]
                    output_depth, _ = batch_data.shape
                    output[:output_depth, current_col:current_col+batch_cols] = batch_data
                elif batch_data.ndim == 3:  # [rows, cols, depth]
                    output_rows, batch_cols, output_depth = batch_data.shape
                    output[:output_rows, current_col:current_col+batch_cols, :output_depth] = batch_data
                
                current_col += batch_cols
        
        return output


def example_processing_function(batch_data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Example processing function for batch processing.
    
    Args:
        batch_data: Batch of 3D data [rows, batch_cols, looks]
        **kwargs: Additional parameters
        
    Returns:
        np.ndarray: Processed batch
    """
    # This is where you would call your actual tomography function
    # For example:
    # results = []
    # for col in range(batch_data.shape[1]):
    #     y_data = batch_data[:, col, :]
    #     tomogram, z_vec, _ = focus_sonic_tomogram(y_data, ...)
    #     results.append(tomogram)
    # return np.stack(results, axis=1)
    
    # Return dummy data for example
    rows, batch_cols, looks = batch_data.shape
    return np.random.randn(rows, batch_cols, 50)  # [rows, batch_cols, depth]


# Quick test function
def test_memory_manager():
    """Test the memory manager with synthetic data."""
    print("Testing Memory Manager...")
    
    # Create synthetic data
    rows, cols, looks = 1000, 500, 100
    data_3d = np.random.randn(rows, cols, looks).astype(np.complex64)
    
    # Initialize memory manager
    mm = MemoryManager(max_memory_gb=1.0)  # 1 GB limit
    
    # Estimate batch size
    batch_size = mm.estimate_batch_size(data_3d.shape)
    print(f"Estimated batch size: {batch_size}")
    
    # Test batch iterator
    for i, (batch, col_indices) in enumerate(mm.batch_iterator(data_3d, batch_size)):
        print(f"  Batch {i}: columns {col_indices[0]}-{col_indices[-1]}, shape {batch.shape}")
        if i >= 2:  # Just test first few batches
            break
    
    # Clean up
    mm.cleanup()
    print("Memory manager test complete.")


if __name__ == "__main__":
    test_memory_manager()
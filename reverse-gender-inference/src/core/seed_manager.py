#!/usr/bin/env python3
"""
Random Seed Management System for Reverse Gender Inference Detection

Ensures reproducibility of all test results through comprehensive seed management.
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np


@dataclass
class SeedConfig:
    """Configuration for random seed management."""
    
    seed: int
    timestamp: str
    description: str
    version: str = "1.0"
    test_scale: int = 19800
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedConfig":
        """Create instance from dictionary."""
        return cls(**data)


class SeedManager:
    """
    Comprehensive random seed management system.
    
    Provides deterministic randomization for all aspects of data generation
    and evaluation to ensure complete reproducibility.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        """
        Initialize seed manager.
        
        Args:
            config_dir: Directory to store seed configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.current_seed: Optional[int] = None
        self.seed_config: Optional[SeedConfig] = None
        self.config_file = self.config_dir / "seed_config.json"
        
    def set_seed(
        self, 
        seed: Optional[int] = None, 
        strategy: str = "default",
        description: str = "Reverse gender inference evaluation",
        save_config: bool = True
    ) -> int:
        """
        Set random seed for all randomization components.
        
        Args:
            seed: Specific seed value. If None, uses strategy to determine
            strategy: Strategy for seed generation ("default", "time", "custom")
            description: Description of this seed usage
            save_config: Whether to save seed configuration to file
            
        Returns:
            The seed value that was set
        """
        if seed is None:
            seed = self._generate_seed(strategy)
        
        # Set seed for all randomization libraries
        self._apply_seed(seed)
        
        # Create and store configuration
        self.current_seed = seed
        self.seed_config = SeedConfig(
            seed=seed,
            timestamp=datetime.now().isoformat(),
            description=description
        )
        
        if save_config:
            self.save_config()
            
        return seed
    
    def _generate_seed(self, strategy: str) -> int:
        """
        Generate seed based on strategy.
        
        Args:
            strategy: Seed generation strategy
            
        Returns:
            Generated seed value
        """
        if strategy == "default":
            return 42  # Consistent with existing system
        elif strategy == "time":
            return int(time.time()) % (2**31)  # Use timestamp
        elif strategy == "random":
            return random.randint(0, 2**31 - 1)
        else:
            return 42  # Fallback to default
    
    def _apply_seed(self, seed: int) -> None:
        """
        Apply seed to all randomization components.
        
        Args:
            seed: Seed value to apply
        """
        # Python built-in random
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # Store for manual randomization
        self.current_seed = seed
    
    def get_random_state(self) -> Dict[str, Any]:
        """
        Get current random state for debugging.
        
        Returns:
            Dictionary containing random states
        """
        return {
            "current_seed": self.current_seed,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_config(self) -> None:
        """Save current seed configuration to file."""
        if self.seed_config is None:
            raise ValueError("No seed configuration to save")
        
        config_data = self.seed_config.to_dict()
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> SeedConfig:
        """
        Load seed configuration from file.
        
        Args:
            config_file: Path to configuration file. If None, uses default
            
        Returns:
            Loaded seed configuration
        """
        if config_file is None:
            config_file = self.config_file
        else:
            config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Seed configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.seed_config = SeedConfig.from_dict(config_data)
        return self.seed_config
    
    def restore_from_config(self, config_file: Optional[Union[str, Path]] = None) -> int:
        """
        Restore random state from saved configuration.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Restored seed value
        """
        config = self.load_config(config_file)
        seed = config.seed
        
        # Apply the loaded seed
        self._apply_seed(seed)
        self.current_seed = seed
        
        return seed
    
    def create_deterministic_sequence(self, length: int, range_max: int = 1000) -> list[int]:
        """
        Create a deterministic sequence of random numbers.
        
        Useful for consistent sampling and ordering operations.
        
        Args:
            length: Length of sequence to generate
            range_max: Maximum value in sequence
            
        Returns:
            List of deterministic random integers
        """
        if self.current_seed is None:
            raise ValueError("No seed set. Call set_seed() first.")
        
        # Use a separate Random instance to avoid affecting global state
        rng = random.Random(self.current_seed)
        return [rng.randint(0, range_max - 1) for _ in range(length)]
    
    def shuffle_deterministic(self, items: list, seed_modifier: int = 0) -> list:
        """
        Shuffle a list deterministically.
        
        Args:
            items: List to shuffle
            seed_modifier: Modifier to add to seed for different shuffles
            
        Returns:
            Shuffled copy of the list
        """
        if self.current_seed is None:
            raise ValueError("No seed set. Call set_seed() first.")
        
        # Create copy and shuffle with modified seed
        items_copy = items.copy()
        rng = random.Random(self.current_seed + seed_modifier)
        rng.shuffle(items_copy)
        return items_copy
    
    def sample_deterministic(
        self, 
        items: list, 
        k: int, 
        seed_modifier: int = 0
    ) -> list:
        """
        Sample items deterministically.
        
        Args:
            items: List to sample from
            k: Number of items to sample
            seed_modifier: Modifier to add to seed for different samples
            
        Returns:
            Sampled items
        """
        if self.current_seed is None:
            raise ValueError("No seed set. Call set_seed() first.")
        
        rng = random.Random(self.current_seed + seed_modifier)
        return rng.sample(items, k)
    
    def get_seed_info(self) -> Dict[str, Any]:
        """
        Get comprehensive seed information.
        
        Returns:
            Dictionary with seed information
        """
        return {
            "current_seed": self.current_seed,
            "config": self.seed_config.to_dict() if self.seed_config else None,
            "config_file_exists": self.config_file.exists(),
            "config_file_path": str(self.config_file)
        }


# Global seed manager instance
_global_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """Get or create global seed manager instance."""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager()
    return _global_seed_manager


def set_global_seed(
    seed: Optional[int] = None,
    strategy: str = "default",
    description: str = "Global seed for reverse gender inference"
) -> int:
    """
    Convenience function to set global seed.
    
    Args:
        seed: Specific seed value
        strategy: Seed generation strategy
        description: Description of seed usage
        
    Returns:
        The seed value that was set
    """
    manager = get_seed_manager()
    return manager.set_seed(seed, strategy, description)


if __name__ == "__main__":
    # Demo usage
    print("Random Seed Management System Demo")
    print("=" * 40)
    
    # Create seed manager
    manager = SeedManager("../config")
    
    # Set seed and demonstrate reproducibility
    seed = manager.set_seed(42, description="Demo test")
    print(f"Set seed: {seed}")
    
    # Generate some random data
    sequence1 = manager.create_deterministic_sequence(10, 100)
    print(f"Sequence 1: {sequence1}")
    
    shuffled_list = manager.shuffle_deterministic(list(range(10)))
    print(f"Shuffled list: {shuffled_list}")
    
    # Reset with same seed
    manager.set_seed(42)
    sequence2 = manager.create_deterministic_sequence(10, 100)
    print(f"Sequence 2: {sequence2}")
    print(f"Sequences identical: {sequence1 == sequence2}")
    
    # Show seed info
    info = manager.get_seed_info()
    print(f"Seed info: {info}")
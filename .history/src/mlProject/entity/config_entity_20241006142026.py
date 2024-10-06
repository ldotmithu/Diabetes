from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir:Path
    URL:str
    local_data_path:Path
    unzip_dir:Path
    
@dataclass
class DataTransfomationConfig:
    root_dir:Path
    data_path:Path    
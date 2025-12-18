import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import csv
import json
import itertools
from typing import Optional, Callable, Tuple, List, Union
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import torch.nn.functional as F
import numpy as np

# Try importing trimesh, provide a helpful error if missing
try:
    import trimesh
except ImportError:
    trimesh = None

# Try importing av
try:
    import av
except ImportError:
    av = None
import torchvision

# -----------------------------------------------------------------------------
# IMAGE DATASET
# -----------------------------------------------------------------------------

class MyImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and transforming images from a flat directory.

    This dataset scans a directory for valid images. If no custom transform is provided,
    it automatically resizes images to `img_size` and converts them to tensors.

    Attributes:
        _img_size (tuple[int, int]): The target dimensions (Height, Width).
        _img_transform (Callable): The transform function (custom or default).
        _files (list[str]): A list of valid image file paths found in the directory.
    """

    def __init__(self,
                 directory: str,
                 img_size: tuple[int, int] = (256, 256),
                 transform: Optional[Callable] = None):
        """
        Initializes the MyImageDataset.

        Args:
            directory (str): The system path to the folder containing the images.
            img_size (tuple[int, int], optional): Target (height, width) for resizing.
                Used ONLY if `transform` is None. Defaults to (256, 256).
            transform (Callable, optional): A custom transform function. If provided,
                `img_size` is ignored. If None, a default Resize+ToTensor transform
                is created using `img_size`.
        """
        super().__init__()

        # --- Parameter Validation ---
        if not isinstance(directory, str):
            raise TypeError(f"directory must be a string, got {directory}")

        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2:
            raise TypeError(f"img_size must be a tuple/list of two integers (H, W)")

        if not isinstance(img_size[0], int) or img_size[0] < 1:
            raise TypeError(f"Height must be a non-zero positive integer")

        if not isinstance(img_size[1], int) or img_size[1] < 1:
            raise TypeError(f"Width must be a non-zero positive integer")

        if transform is not None and not callable(transform):
            raise TypeError(f"transform must be callable")
        # --- End Validation ---

        self._img_size = img_size

        # --- Transform Logic ---
        # If no transform is provided, create a default one that enforces img_size
        if transform is None:
            self._img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
        else:
            self._img_transform = transform

        # --- File Loading ---
        search_pattern = os.path.join(directory, '*')
        # If img_dir is "./data/train", this becomes "./data/train/*"
        # This returns a list: ['./data/train/img1.jpg', './data/train/img2.png', ...]

        all_files = glob.glob(search_pattern)
        # Use glob to find files MATCHING that pattern

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        self._files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]


        if len(self._files) == 0:
            raise FileNotFoundError(f"No valid images found in {directory}")

    def __len__(self) -> int:
        """
        Returns the total number of image files found in the dataset directory.

        Returns:
            int: The length of the file list.
        """

        return len(self._files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Loads and transforms the image at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The image data converted to a PyTorch tensor.
                If a transform was provided in __init__, the tensor shape depends on the transform.
                Otherwise, it defaults to a standard (C, H, W) tensor via `to_tensor`.
        """

        img_path = self._files[idx]

        # Ensure 3-channel RGB (fixes greyscale/transparency issues)
        img = Image.open(img_path).convert("RGB")

        # Apply the transform (either the user's custom one, or our default resize)
        return self._img_transform(img)


# -----------------------------------------------------------------------------
# AUDIO DATASET
# -----------------------------------------------------------------------------

class MyAudioDataset(Dataset):
    """
    A custom PyTorch Dataset for loading, preprocessing, and transforming audio files.

    This dataset handles the entire audio pipeline:
    1. Loads audio from a directory.
    2. Resamples to a target sample rate.
    3. Mixes multiple channels down to Mono.
    4. Pads or Crops the audio to a fixed duration.
    5. Converts the raw waveform into a Log-Mel Spectrogram.

    Attributes:
        _mel_transform (torch.nn.Sequential): The pipeline converting waveform to Mel DB.
        _target_sr (int): The expected sample rate (e.g., 16000 Hz).
        _target_seconds (int): The expected duration in seconds.
        _files (list[str]): List of valid audio file paths.
        _patch_size (int): Patch size used by the audioAdapter. Needed to make sure the logic using it in the imageAdapter of the audioAdapter works.
        _hop_length(int): The distance (in samples) the window slides forward to create the next time frame
    """

    def __init__(self,
                 audio_dir: str,
                 target_sr: int = 16000,
                 target_seconds: int = 5,
                 patch_size = 16,
                 hop_length = 200):
        """
        Initializes the MyAudioDataset.

        Args:
            audio_dir (str): Path to the directory containing audio files.
            target_sr (int, optional): Desired sample rate. Defaults to 16000.
            target_seconds (int, optional): Desired audio length in seconds. Defaults to 5.
            patch_size(int, optional): Patch size used by the audioAdapter
            hop_length(int, optional): The distance (in samples) the window slides forward to create the next time frame

        Raises:
            TypeError: If arguments are of incorrect types.
            ValueError: If `target_sr`, `target_seconds`, 'patch_size' or 'hop_length' are not positive.
            FileNotFoundError: If no valid audio files are found in `audio_dir`.
        """
        super().__init__()

        # --- Parameter Validation ---
        if not isinstance(audio_dir, str):
            raise TypeError(f"audio_dir must be a string, got {audio_dir}")

        for param, value in {"Target sample":target_sr,
                             "Target seconds": target_seconds,
                             "Patch size":patch_size,
                             "Hop length":hop_length}.items():

            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{param} must be a positive integer, got {value}")

        # --- End Validation ---


        # Define the transformation pipeline
        # 1. Convert to Mel Spectrogram (Power)
        # 2. Convert Power to Decibels (Log Scale Normalization)
        self._mel_transform = torch.nn.Sequential(
            T.MelSpectrogram(sample_rate=target_sr, n_mels=128, hop_length = hop_length),
            T.AmplitudeToDB(stype='power', top_db=80)
        )

        self._patch_size = patch_size
        self._hop_length = hop_length
        self._target_sr = target_sr
        self._target_seconds = target_seconds

        # Populate file list
        # Change 1: Add '**' to look in subfolders
        search_pattern = os.path.join(audio_dir, '**', '*')

        # Change 2: Add recursive=True
        all_files = list(glob.glob(search_pattern, recursive=True))


        valid_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.riff', '.raw'}

        self._files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if len(self._files) == 0:
            raise FileNotFoundError(f"No valid audio files found in {audio_dir}")

    def __len__(self) -> int:
        """
        Returns the total number of audio files in the dataset.
        """
        return len(self._files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Loads and processes a single audio file.

        The processing steps are:
        1. Load file.
        2. Resample to `target_sr`.
        3. Mix to Mono (average channels).
        4. Pad (with zeros) or Random Crop to match `target_seconds`.
        5. Generate Log-Mel Spectrogram.

        Args:
            idx (int): Index of the file to load.

        Returns:
            torch.Tensor: A Log-Mel Spectrogram tensor.
            Shape: (1, n_mels, time_frames).
            Example shape for 5s @ 16kHz: (1, 128, ~400)
        """
        audio_path = self._files[idx]

        # 1. Load waveform
        waveform, sr = torchaudio.load(audio_path)

        # 2. Resample (to make all audio consistent)
        if sr != self._target_sr:
            waveform = F_audio.resample(waveform, sr, self._target_sr)

        # 3. Mix to Mono
        # Because parameter of load: channels_first = True by default, waveform shape is [channels, time]
        if waveform.shape[0] > 1:
            # Average across the channel dimension (dim=0) to get Mono
            # Result shape: [1, Time] (we keep dim=0 to maintain 2D shape)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Fix seconds

        # --- Geometric Math ---
        # 1. Estimate frames
        n_frames = (self._target_seconds * self._target_sr) // self._hop_length
        # 2. Round UP to the next multiple of patch_size (e.g. 16)
        n_frames = ((n_frames // self._patch_size) + 1) * self._patch_size

        target_length = n_frames * self._hop_length

        length_difference = waveform.size(1) - target_length

        if length_difference > 0:
            # Positive difference means audio is too long -> Crop
            # torch.randint returns a tensor, so we use .item() to get the int value
            start_idx = torch.randint(0, length_difference + 1, (1,)).item()
            # Slice: Keep all channels (:), take chunk from start_idx
            waveform = waveform[:, start_idx : start_idx + target_length]
        elif length_difference < 0:
            # when negative (which mean we pad), length_difference becomes the amount of value to pad
            # (0, amount_to_pad) means:
            # 0 zeros added to the LEFT (beginning)
            # amount_to_pad zeros added to the RIGHT (end)
            waveform = F.pad(waveform, (0, -length_difference))

        # 5. Convert to Spectrogram
        mel_spectrogram = self._mel_transform(waveform)

        # 6. SAFETY CLIP: Ensure we have exactly n_frames (removes the +1 STFT artifact of the mel transform)
        mel_spectrogram = mel_spectrogram[..., :n_frames]

        return mel_spectrogram

# -----------------------------------------------------------------------------
# TEXT DATASET
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

class MyTextDataset(Dataset):
    """
    A custom Dataset for loading text from a directory of .txt, .json, .csv, or .tsv files.

    - .txt / .json: Each file is treated as a single sample.
    - .csv / .tsv: Each row in each file is treated as a single sample.
    """

    def __init__(self,
                 text_dir: str,
                 tokenizer_name: str,
                 max_length: int = 512,
                 file_type: str = "txt",
                 begin: Union[int, List[int], Tuple[int, ...], None] = None,
                 end: Union[int, List[int], Tuple[int, ...], None] = None):
        """
        Args:
            text_dir (str): Path to directory containing files.
            tokenizer_name (str): Name of the Hugging Face tokenizer.
            max_length (int): Fixed sequence length for the output tensors.
            file_type (str): "txt", "json", "csv", or "tsv".
            begin (int | iterable):
                - int: Starting column index (for csv/tsv).
                - [col]: Starting column index.
                - [col, row]: Starting column index and starting row (skip `row` lines at start of every file).
            end (int | iterable):
                - int: Ending column index (exclusive).
                - [col]: Ending column index.
                - [col, row]: Ending column index and ending row (stop reading at `row` line of every file).
        """
        super().__init__()

        # --- Parameter Validation ---
        if not isinstance(text_dir, str):
            raise TypeError(f"text_dir must be a string, got {text_dir}")
        if not isinstance(tokenizer_name, str):
            raise TypeError(f"tokenizer_name must be a string, got {tokenizer_name}")
        if not isinstance(max_length, int) or max_length < 1:
            raise TypeError(f"max_length must be a positive integer, got {max_length}")

        accepted_file_types = ["txt", "json", "csv", "tsv"]
        if not isinstance(file_type, str) or file_type not in accepted_file_types:
            raise TypeError(f"file_type must be {accepted_file_types}. Got {file_type}.")

        # Validate 'begin' and 'end'
        for param_name, param_val in [("begin", begin), ("end", end)]:
            if param_val is not None:
                if isinstance(param_val, int):
                    if param_val < 0:
                        raise ValueError(f"{param_name} must be non-negative.")
                elif isinstance(param_val, (list, tuple)):
                    if file_type not in ["csv", "tsv"]:
                        raise ValueError(f"Iterable '{param_name}' only supported for csv/tsv.")
                    if len(param_val) not in [1, 2]:
                        raise ValueError(f"'{param_name}' iterable must have length 1 or 2.")
                    if not all(isinstance(x, int) for x in param_val):
                        raise TypeError(f"Elements in '{param_name}' must be integers.")
                else:
                    raise TypeError(f"{param_name} must be int, list/tuple of ints, or None.")

        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._file_type = file_type

        # Parse begin/end into explicit (col, row) logic
        # Defaults: Start at row 0, col 0. End at last row, last col.
        self._start_col = 0
        self._start_row = 0
        self._end_col = None # None implies "till the end"
        self._end_row = None # None implies "till the end"

        if begin is not None:
            if isinstance(begin, int):
                self._start_col = begin
            elif len(begin) >= 1:
                self._start_col = begin[0]
                if len(begin) == 2:
                    self._start_row = begin[1]

        if end is not None:
            if isinstance(end, int):
                self._end_col = end
            elif len(end) >= 1:
                self._end_col = end[0]
                if len(end) == 2:
                    self._end_row = end[1]

        # Find files
        search_pattern = os.path.join(text_dir, '**', f'*.{file_type}')
        self._files = sorted(glob.glob(search_pattern, recursive=True))

        if len(self._files) == 0:
            raise FileNotFoundError(f"No .{file_type} files found in {text_dir}")

        # --- Pre-calculate Dataset Size ---
        # For CSV/TSV, we must map global index -> (File, Row_in_File)
        self._file_map = [] # List of tuples: (start_index, num_samples, file_path)
        self._total_samples = 0

        if self._file_type in ["csv", "tsv"]:
            self._precalculate_csv_offsets()
        else:
            # For txt/json, 1 file = 1 sample
            self._total_samples = len(self._files)

    def _precalculate_csv_offsets(self):
        """Scans all CSV/TSV files to count valid rows and build an index map."""
        print(f"Indexing {len(self._files)} {self._file_type} files...")
        current_idx = 0

        delimiter = '\t' if self._file_type == 'tsv' else ','

        for f_path in self._files:
            try:
                # We interpret file line-by-line to avoid loading into RAM
                with open(f_path, 'r', encoding='utf-8', errors='replace') as f:
                    # Logic: Count lines, subtract start_row.
                    # If end_row is specified, limit count to (end_row - start_row).

                    # Fast line counting
                    total_lines = sum(1 for _ in f)

                    # Calculate valid samples in this file
                    # 1. Skip rows at start
                    available_lines = max(0, total_lines - self._start_row)

                    # 2. Cut off at end_row if specified
                    if self._end_row is not None:
                        # Effective lines is min(available, end_row - start_row)
                        # Ensure we don't go negative
                        max_allowed = max(0, self._end_row - self._start_row)
                        valid_samples = min(available_lines, max_allowed)
                    else:
                        valid_samples = available_lines

                    if valid_samples > 0:
                        self._file_map.append({
                            "start": current_idx,
                            "count": valid_samples,
                            "path": f_path
                        })
                        current_idx += valid_samples

            except Exception as e:
                print(f"Warning: Could not read {f_path}. Skipping. Error: {e}")

        self._total_samples = current_idx
        print(f"Indexed {self._total_samples} samples.")

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self._total_samples}")

        text_content = ""

        # --- Case A: TXT or JSON (1 File = 1 Sample) ---
        if self._file_type in ["txt", "json"]:
            file_path = self._files[idx]
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if self._file_type == "txt":
                    text_content = f.read()
                else:
                    # Load JSON and dump back to string to standardize formatting
                    data = json.load(f)
                    text_content = json.dumps(data, ensure_ascii=False)

        # --- Case B: CSV or TSV (1 Row = 1 Sample) ---
        else:
            # 1. Find which file contains this global index
            target_file_info = None
            local_idx = 0

            # Simple linear search (fast enough for <10k files)
            # Binary search could be used for massive numbers of files
            for info in self._file_map:
                if info["start"] <= idx < info["start"] + info["count"]:
                    target_file_info = info
                    local_idx = idx - info["start"]
                    break

            if target_file_info is None:
                raise IndexError(f"Internal Error: Index {idx} not found in file map.")

            # 2. Extract the specific row from the file
            file_path = target_file_info["path"]
            delimiter = '\t' if self._file_type == 'tsv' else ','

            # The actual row number in the file = skipped_header + local_index
            target_row_number = self._start_row + local_idx

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f, delimiter=delimiter)

                # Efficiently skip to the target row
                # islice(iterable, start, stop) -> here we skip 'target_row_number' items
                row_iterator = itertools.islice(reader, target_row_number, None)
                try:
                    row = next(row_iterator)
                except StopIteration:
                    # Should not happen if pre-calc was correct, but safety first
                    row = []

                # 3. Select Columns
                if row:
                    # Slice columns based on start_col and end_col
                    # row is a list of strings
                    selection = row[self._start_col : self._end_col]
                    text_content = " ".join(selection)

        # --- Tokenization ---
        encoding = self._tokenizer(
            text_content,
            padding='max_length',
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt'
        )

        return encoding['input_ids'].squeeze(0)

# -----------------------------------------------------------------------------
# POINT CLOUD DATASET
# -----------------------------------------------------------------------------

class MyPointCloudDataset(Dataset):
    """
    A generic Dataset for loading and processing 3D Point Clouds from common file formats
    (.off, .obj, .ply, .stl).

    It handles:
    1. Loading meshes or point clouds.
    2. Sampling points from mesh surfaces (if faces exist) or vertices.
    3. Resampling to a fixed number of points (num_points).
    4. Normalizing to a unit sphere (essential for Fourier encodings).

    Attributes:
        files (list): List of file paths.
        num_points (int): The target number of points (N) per sample.
        normalize (bool): Whether to center and scale to unit sphere.
    """

    def __init__(self,
                 root_dir: str,
                 num_points: int = 2048,
                 normalize: bool = True,
                 valid_extensions: Optional[List[str]] = None):
        """
        Args:
            root_dir (str): Root directory containing the 3D files (searches recursively).
            num_points (int): Number of points to sample from the object.
            normalize (bool): If True, center and scale to fit in unit sphere [-1, 1].
            valid_extensions (list, optional): List of extensions to load.
                                               Defaults to ['.off', '.obj', '.ply', '.stl'].
        """
        if trimesh is None:
            raise ImportError("MyPointCloudDataset requires 'trimesh'. Please install it via `pip install trimesh`.")

        self.root_dir = root_dir
        self.num_points = num_points
        self.normalize = normalize

        if valid_extensions is None:
            self.valid_extensions = ['.off', '.obj', '.ply', '.stl']
        else:
            self.valid_extensions = valid_extensions

        # Recursively find all valid files
        # We search for **/* to include subdirectories
        search_pattern = os.path.join(root_dir, '**', '*')
        all_files = glob.glob(search_pattern, recursive=True)

        self.files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in self.valid_extensions
        ]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No 3D files found in {root_dir} with extensions {self.valid_extensions}")

        print(f"Found {len(self.files)} 3D files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            # 1. Load the 3D file
            # force='mesh' ensures we get a Trimesh object even if it's a scene
            # (Trimesh handles the complexity of merging scene geometries if needed)
            mesh = trimesh.load(path, force='mesh')
        except Exception as e:
            # Fallback for handling bad files without crashing the training
            print(f"Error loading {path}: {e}. Returning zeros.")
            return torch.zeros((self.num_points, 3), dtype=torch.float32)

        # 2. Sample Points
        # If the object is a mesh (has faces), sample from the SURFACE (better representation)
        # If the object is a point cloud (no faces), sample from vertices.

        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Efficient barycentric sampling from surface
            points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        else:
            # It's a point cloud, we must resample vertices
            vertices = mesh.vertices
            n_vertices = len(vertices)

            if n_vertices >= self.num_points:
                # Random choice without replacement (downsample)
                choice_idx = np.random.choice(n_vertices, self.num_points, replace=False)
            else:
                # Random choice WITH replacement (upsample/pad)
                choice_idx = np.random.choice(n_vertices, self.num_points, replace=True)

            points = vertices[choice_idx]

        # 3. Normalize (Unit Sphere)
        # This is critical for the Fourier Features to work well (they expect inputs roughly in [-1, 1])
        if self.normalize:
            # Center at origin
            centroid = np.mean(points, axis=0)
            points = points - centroid

            # Scale to unit sphere
            # Find the point furthest from origin
            m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))

            # Avoid division by zero
            if m > 0:
                points = points / m

        # 4. Convert to Tensor
        # Shape: [num_points, 3]
        return torch.tensor(points, dtype=torch.float32)

# -----------------------------------------------------------------------------
# VIDEO DATASET
# -----------------------------------------------------------------------------

class MyVideoDataset(Dataset):
    """
    A memory-efficient Video Dataset using PyAV for lazy loading.

    It handles:
    1. Loading video metadata without decoding the file.
    2. Lazy-loading only specific frames (Uniform or Clip) to save RAM.
    3. Resizing and normalizing.
    """

    def __init__(self,
                 root_dir: str,
                 num_frames: int = 16,
                 img_size: tuple[int, int] = (224, 224),
                 sample_mode: str = 'uniform',
                 valid_extensions: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            root_dir (str): Root directory containing video files.
            num_frames (int): Number of frames to sample.
            img_size (tuple): Target (Height, Width).
            sample_mode (str): 'uniform' (evenly spaced) or 'clip' (consecutive frames).
            valid_extensions (list): Extensions to look for.
            transform (callable): Optional transform.
        """
        super().__init__()

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.sample_mode = sample_mode
        self.transform = transform

        if valid_extensions is None:
            self.valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        else:
            self.valid_extensions = valid_extensions

        search_pattern = os.path.join(root_dir, '**', '*')
        all_files = glob.glob(search_pattern, recursive=True)

        self.files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in self.valid_extensions
        ]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No video files found in {root_dir}")

        print(f"Found {len(self.files)} video files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            frames = self._read_specific_frames(path)
        except Exception as e:
            print(f"Error loading {path}: {e}. Returning zeros.")
            # Return placeholder tensor (C, T, H, W)
            return torch.zeros((3, self.num_frames, *self.img_size), dtype=torch.float32)

        # frames is currently a list of [C, H, W] tensors. Stack them -> [T, C, H, W]
        video = torch.stack(frames)

        # Permute to [C, T, H, W] for Perceiver Adapter
        video = video.permute(1, 0, 2, 3)

        if self.transform:
            video = self.transform(video)

        return video

    def _read_specific_frames(self, path: str) -> List[torch.Tensor]:
        """
        Uses PyAV to decode only the required frames.
        """
        container = av.open(path)

        # 1. Get Video Stream and Metadata
        stream = container.streams.video[0]

        # 'frames' property can be 0 sometimes, fallback to distinct decoding or estimation
        total_frames = stream.frames
        if total_frames == 0:
            # Fallback: estimate from duration and fps
            # This is an approximation but fast
            if stream.duration and stream.average_rate:
                total_frames = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
            else:
                # Last resort: Decode everything to count (slow, but safe fallback)
                total_frames = sum(1 for _ in container.decode(video=0))
                # Re-open for actual reading
                container.close()
                container = av.open(path)
                stream = container.streams.video[0]

        # 2. Determine Indices
        indices = []
        if total_frames <= self.num_frames:
            # Pad by repeating indices if video is too short
            indices = np.arange(total_frames)
            # Repeat to fill num_frames
            if total_frames > 0:
                indices = np.resize(indices, self.num_frames)
            else:
                indices = np.zeros(self.num_frames, dtype=int)
        else:
            if self.sample_mode == 'uniform':
                # Evenly spaced indices
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            elif self.sample_mode == 'clip':
                # Random start or Center start? Let's do Center for validation stability
                start = (total_frames - self.num_frames) // 2
                indices = np.arange(start, start + self.num_frames)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        # Sort indices to allow efficient seeking/looping
        indices = np.sort(indices)
        target_indices = set(indices)

        frames = []

        # 3. Decode Loop
        # Strategy: Seek to the first index, then decode forward.
        # This prevents decoding the whole video if we only need the end.

        # Seek to the first target frame (PyAV seeks to nearest keyframe before this timestamp)
        # We calculate timestamp roughly:
        # pts = frame_index * (duration / total_frames) ... inaccurate.
        # simpler: seek by frame index logic is complex in PyAV.

        # ROBUST STRATEGY:
        # For 'Clip' (consecutive): Seek once, read N.
        # For 'Uniform' (scattered): Iterate through stream, skipping unneeded.
        # (Seeking repeatedly is slow, decoding everything is OOM).
        # We compromise: We iterate, but we close early.

        # Optimization: If uniform sampling asks for frame 0, 100, 200...
        # We must decode 0..200. We cannot easily skip 1..99 without seeking.
        # For simplicity and stability in this notebook: we iterate.
        # NOTE: For very long videos (hours), this is slow. For standard datasets (seconds), it's fine.

        current_idx = 0
        collected_count = 0

        # To speed up, we can seek to the start index first
        start_target = indices[0]

        # Rough seek to near the start frame (using time_base)
        if start_target > 0 and stream.average_rate > 0:
             # timestamp = frame / fps
             timestamp_sec = start_target / float(stream.average_rate)
             # PyAV seek expects time in stream.time_base
             target_pts = int(timestamp_sec / float(stream.time_base))
             container.seek(target_pts, stream=stream)
             # Reset current_idx estimate (seeking is inexact, so we might be slightly before)
             # In a robust system, we'd check frame.pts. Here we assume we are close enough
             # and rely on the fact that we grab 'num_frames' frames from here.

             # Actually, precise seeking is hard.
             # FALLBACK: Just standard decoding loop. It is much lighter than read_video
             # because we discard frames immediately, never storing them in RAM.
             pass

        # Resize transform (applied on individual PIL images to save RAM)
        # We do resizing HERE, before stacking, to minimize tensor size
        resize_transform = torchvision.transforms.Resize(self.img_size)
        to_tensor = torchvision.transforms.ToTensor()

        for frame in container.decode(video=0):
            if current_idx in target_indices:
                # Convert AVFrame to PIL/Tensor
                img = frame.to_image() # PIL Image

                # Resize immediately (PIL is fast/efficient)
                img = resize_transform(img)

                # Convert to Tensor [C, H, W] and normalize [0, 1]
                t_img = to_tensor(img)
                frames.append(t_img)
                collected_count += 1

                if collected_count >= self.num_frames:
                    break

            current_idx += 1

            # Safety break for infinite loops or massive videos
            if current_idx > indices[-1] + 100:
                break

        container.close()

        # Handle case where seeking/decoding failed to get enough frames (e.g. video shorter than expected)
        if len(frames) < self.num_frames:
            # Pad with last frame or zeros
            diff = self.num_frames - len(frames)
            if len(frames) > 0:
                frames.extend([frames[-1]] * diff)
            else:
                frames.extend([torch.zeros(3, *self.img_size)] * diff)

        return frames

# -----------------------------------------------------------------------------
# TASK DATASET ADAPTER
# -----------------------------------------------------------------------------

class TaskDatasetAdapter(Dataset):
    """
    Unify image dataset into one type of dataset, so that they can be processed the same way
    regardless of the task type or the origin of the dataset.

    This allows multiple different datasets of images to be concatenized and be fed to a dataloader
    in an uniform way.
    """

    def __init__(self, source_dataset, task_id, transform=None, task_type='classification'):
        """
        Args:
            source_dataset: The original PyTorch/HuggingFace dataset.
            task_id (int): A unique integer identifying this task (e.g., 0).
            transform: Optional transform to apply to the image (resize/normalize/...).
            task_type (str): 'classification', 'segmentation' or 'autoencoding' to determine parsing logic.
        """

         # --- Parameter Validation ---

        if not isinstance(source_dataset,torch.utils.data.Dataset):
            raise TypeError(f"source_dataset is not of type torch.utils.data.Dataset, \
            instead it is {type(source_dataset)}")

        if not isinstance(task_id, int):
            raise TypeError(f"task_id: {task_id} is not an integer, instead it is {type(task_id)}")

        if transform is not None and not callable(transform):
            raise TypeError(f"transform must be callable, got {type(transform)}")

        tasks = ["classification", "autoencoding", "segmentation"]
        if not isinstance(task_type, str) or task_type not in tasks:
            raise TypeError(f"task_type: {task_type} is either not a string \
            (current type: {type(task_type)} or \
            doesn't belong in the available tasks: {tasks}")

         # --- End Validation ---

        self.source_dataset = source_dataset
        self.task_id = task_id
        self.transform = transform
        self.task_type = task_type

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        # 1. Fetch raw data from the source
        raw_data = self.source_dataset[idx]

        # 2. Parse based on Source Type
        # Handle Hugging Face (dict) vs Torchvision (tuple)
        if isinstance(raw_data, dict):
            # Hugging Face usually returns dicts like {'image': ..., 'label': ...}
            image = raw_data['image']
            target = raw_data['label'] if 'label' in raw_data else None
            mask = raw_data['segmentation'] if 'segmentation' in raw_data else None
        elif isinstance(raw_data, (tuple, list)):
            # Torchvision usually returns (image, label) or (image, mask)
            image = raw_data[0]
            target = raw_data[1]
            mask = raw_data[1] if self.task_type == 'segmentation' else None
        else:
            # Fallback for custom datasets returning just image
            image = raw_data
            target = raw_data if self.task_type == 'autoencoding' else None
            mask = None

        # 3. Apply Transforms (Crucial for Perceiver: images must be same size)
        if self.transform:
            image = self.transform(image)

        # 4. Construct the Unified Dictionary

        handling_label = lambda label: label if isinstance(label, torch.Tensor) else torch.tensor(label)

        # We fill missing keys with distinct placeholders (e.g., -100 is standard ignore_index)
        return {
            "pixel_values": image,
            "label": handling_label(target) if (target is not None and self.task_type in ['classification', 'autoencoding']) else torch.tensor(-100),
            "mask": torch.tensor(mask) if (mask is not None and self.task_type == 'segmentation') else torch.zeros_like(image[0]), # Placeholder mask
            "task_id": torch.tensor(self.task_id, dtype=torch.long)
        }
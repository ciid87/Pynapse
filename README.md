# Pynapse

Headless Python recreation of SynapseJ macro for automated synapse detection and colocalization analysis in fluorescence microscopy images.

## Requirements

- **Python 3.8+**
- **Linux or macOS** (Windows users: see WSL below)
- **uv** (recommended) or pip

Fiji is downloaded automatically on first run if not already installed.

### Windows Users

Pynapse runs in WSL (Windows Subsystem for Linux). We recommend a minimal WSL2 setup with Arch Linux:

```powershell
# In PowerShell (as Administrator)
wsl --install
# Recommend using archlinux to minimize footprint:
# wsl --install -d archlinux
# Some configuration required to get it dependencies working
```

Then run all commands inside the WSL terminal.

## Installation

### Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies automatically.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
cd pynapse

# With uv (creates virtual environment automatically)
uv sync --index-url https://pypi.org/simple

# Or with pip
pip install numpy pandas pillow
```

## Quick Start

### 1. Configure your analysis

Copy and edit the example config:

```bash
cp config_example.txt my_config.txt
```

Edit `my_config.txt` with your paths and parameters:

```ini
# Required paths (use absolute paths)
source_dir=/path/to/your/images
dest_dir=/path/to/output

# Channel assignments (1-indexed)
pre_channel=4
post_channel=3

# Detection thresholds (adjust for your images)
pre_min=658
post_min=578
```

### 2. Run Pynapse

```bash
python Pynapse.py /path/to/my_config.txt
```

On first run, if Fiji isn't found, you'll be prompted to download it or provide a path. The path is saved to `~/.pynapse_config` for future runs.

### 3. Validate output (optional)

Compare your output against a reference dataset:

```bash
# With uv
uv run python validate_pynapse.py /path/to/output /path/to/reference

# Or with pip-installed dependencies
python validate_pynapse.py /path/to/output /path/to/reference
```

Run `python validate_pynapse.py --help` for all options.

## Output Files

### Main results (in `dest_dir/`)

| File | Description |
|------|-------------|
| `Syn Pre Results.txt` | Synaptic pre-synaptic puncta measurements |
| `Syn Post Results.txt` | Synaptic post-synaptic puncta measurements |
| `All Pre Results.txt` | All detected pre-synaptic puncta |
| `All Post Results.txt` | All detected post-synaptic puncta |
| `CorrResults.txt` | Correlation/colocalization results |
| `Collated ResultsIF.txt` | Summary statistics per image |

### Per-image files

| Pattern | Description |
|---------|-------------|
| `*Pre.txt` / `*Post.txt` | Individual image measurements |
| `*PreF.tif` / `*PostF.tif` | Filtered images |
| `*RoiSet.zip` | ROI collections (loadable in ImageJ/Fiji) |

### Presentation outputs (in `dest_dir/present/`)

| File | Description |
|------|-------------|
| `Batch_Synapse_Report.csv` | Comprehensive report for all images |
| `*_Report.csv` | Per-image detailed reports |

## Configuration Reference

See `config_example.txt` for all available parameters. Key settings:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `source_dir` | Input image directory | (required) |
| `dest_dir` | Output directory | (required) |
| `pre_channel` | Pre-synaptic marker channel (1-indexed) | 4 |
| `post_channel` | Post-synaptic marker channel (1-indexed) | 3 |
| `pre_min` | Pre-synaptic intensity threshold | 658 |
| `post_min` | Post-synaptic intensity threshold | 578 |
| `pre_size_low` / `pre_size_high` | Puncta size range (µm²) | 0.08 / 2.5 |
| `overlap_pixels` | Min overlap to define synapse | 1 |

## License

MIT License

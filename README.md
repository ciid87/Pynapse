# Pynapse

Headless Python recreation of SynapseJ macro for automated synapse detection and colocalization analysis in fluorescence microscopy images.

## Requirements

- **Python 3.8+**
- **Linux or macOS** (Windows users: see WSL below)
- **uv** (recommended) or pip

Fiji is downloaded automatically on first run if not already installed.

### Windows Users

Pynapse runs in WSL (Windows Subsystem for Linux). We recommend using the default WSL distro, Ubuntu:

```powershell
# In PowerShell (as Administrator)
wsl --install
```

Then run all commands inside the WSL terminal.

## Installation

### Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies automatically.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Exit WSL and reopen
```bash
# Enter each line one at a time
exit
wsl
cd
```

### Add missing dependencies and binaries to WSL Ubuntu
```bash
# Enter each line one at a time
sudo apt update
sudo apt install unzip gedit libx11-dev
```

### Copy Pynapse from GitHub
```bash
git clone https://github.com/ciid87/Pynapse.git 
```

### Install dependencies

```bash
cd Pynapse

# With uv (creates virtual environment automatically)
uv sync
```

## Quick Start

### 1. Configure your analysis

You will want to move or copy your .TIF files into Linux in their own folder using Windows Explorer. (ie open Windows Explorer, click Linux > Ubuntu > home > [UserName] > [Make Folder(s) Here])

Copy and make mandatory changes to the config file `my_config.txt` to direct it to your input directory (folder containing .TIF files for analysis) and output directory (empty folder where output will be generated).

```bash
cp config_example.txt my_config.txt
gedit my_config.txt
```
To view the filepath and select the correct portion for the config file, open Windows Explorer, right click one of the images, and select "Properties" to see the file path. 

Example:
If images are located in "\\\\wsl.localhost\Ubuntu\home\bob\study1\images", 
the resulting necessary path for the config file is:

```bash
/home/bob/study1/images
```
(Note forward rather than back slashes when copying from windows)

```ini
# Required paths (use absolute paths)
source_dir=/path/to/your/images
dest_dir=/path/to/output_folder
```
If making no further chages, save and close the gedit window. Otherwise, you may edit `my_config.txt` with your specific parameters if you wish to deviate from default values. Only changed parameters need to be included. Some examples of changes:

```ini
# Channel assignments 
pre_channel=4
post_channel=3

# Detection thresholds (adjust for your images)
pre_min=658
post_min=578
```


### 2. Run Pynapse

```bash
uv run Pynapse.py my_config.txt
```

On first run, if Fiji isn't found, you'll be prompted to download it or provide a path. The path is saved to `~/.pynapse_config` for future runs.

As Pynapse is running, it will generate debug statements indicating that it is progressing through the images. The batch is finished when the output text in the WSL terminal reads "All images processed; outputs saved to [Output/file/path]." and the prompt returns to the terminal.

### 3. Validate output (optional)

Compare your output against a reference dataset:

```bash
# With uv
uv run python validate_pynapse.py /path/to/output /path/to/reference

# Or with pip-installed dependencies
python validate_pynapse.py /path/to/output /path/to/reference
```

Run `python validate_pynapse.py --help` for all options.

## Future Runs After Install
If Pynapse is already installed and you are running data without needing to reiterate installation steps:
### 1. Open Ubuntu
Open Windows Powershell > Downwards arrow to open a new tab > Select Ubuntu
### 2. Set working directory to Pynapse folder
```bash
# Confirm Pynapse folder is in the current home directory
ls

# Set working directory
cd Pynapse
```
### 3. Run Pynapse
```bash
uv run Pynapse.py my_config.txt

# or run with uv
uv run python validate_pynapse.py /path/to/output /path/to/reference

```
## Output Files

### Main results (in `dest_dir/`)

| File | Description |
|------|-------------|
| `Syn Pre Results.txt` | Synaptic pre-synaptic puncta measurements |
| `Syn Post Results.txt` | Synaptic post-synaptic puncta measurements |
| `All Pre Results.txt` | All detected pre-synaptic puncta |
| `All Post Results.txt` | All detected post-synaptic puncta |
| `CorrResults.txt` | Pre-to-Post correlation/colocalization results |
| `CorrResults2.txt` | Post-to-Pre correlation/colocalization results |
| `Collated ResultsIF.txt` | Summary statistics per image |
| `IFALog.txt` | Time-stamped log of batch run |

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

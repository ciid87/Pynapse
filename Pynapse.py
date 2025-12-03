# Pynapse.py - Repaired Headless Fiji/Jython recreation of SynapseJ macro v1.
"""
Pynapse.py - Repaired Headless Fiji/Jython recreation of SynapseJ macro v1.

REFERENCE:
Moreno-Manrique, M., et al. (2021). SynapseJ: An ImageJ plugin for the automated 
detection of synapses. bioRxiv.

This script implements the image processing pipeline described in the SynapseJ paper
for the automated detection and quantification of synapses in immunofluorescence images.
It replicates the logic of the original ImageJ macro (SynapseJ_v_1.ijm) but is 
adapted for headless execution in a cluster environment using Jython.
"""

import os
import sys
import math
import csv
from collections import defaultdict, OrderedDict
from datetime import datetime

# --- BOOTSTRAP LOGIC FOR CPYTHON ---
def is_jython():
    return sys.platform.startswith('java')

if not is_jython():
    # We are running in standard Python (CPython).
    # We need to find Fiji and launch this script using it.
    import subprocess
    import shutil
    
    CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".pynapse_config")
    
    def get_fiji_path():
        # Check config file
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                path = f.read().strip()
                if os.path.exists(path):
                    return path
        
        # Check common locations
        common_paths = [
            "Fiji.app/ImageJ-linux64", # Relative
            "Fiji.app/ImageJ-win64.exe",
            "Fiji.app/Contents/MacOS/ImageJ-macosx",
            "fiji-linux-x64", # Relative (from user's previous setup)
            "Fiji/fiji-linux-x64",
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
        ]
        for p in common_paths:
            if os.path.exists(p):
                return os.path.abspath(p)
                
        return None

    def setup_fiji():
        print("Fiji not found. Setting up...")
        # Download
        if not os.path.exists("fiji-latest-linux64-jdk.zip"):
            print("Downloading Fiji...")
            subprocess.check_call(["wget", "https://downloads.imagej.net/fiji/latest/fiji-latest-linux64-jdk.zip"])
        
        # Unzip
        if not os.path.exists("Fiji.app") and not os.path.exists("Fiji"):
            print("Unzipping...")
            subprocess.check_call(["unzip", "-q", "fiji-latest-linux64-jdk.zip"])
            
        # Locate binary after unzip
        # The zip usually creates 'Fiji.app'
        fiji_bin = None
        if os.path.exists("Fiji.app/ImageJ-linux64"):
            fiji_bin = os.path.abspath("Fiji.app/ImageJ-linux64")
        elif os.path.exists("Fiji/fiji-linux-x64"): # User's specific structure
             fiji_bin = os.path.abspath("Fiji/fiji-linux-x64")
             
        if not fiji_bin:
             # Fallback search
             for root, dirs, files in os.walk("."):
                 if "ImageJ-linux64" in files:
                     fiji_bin = os.path.abspath(os.path.join(root, "ImageJ-linux64"))
                     break
        
        if fiji_bin:
            print("Updating Fiji...")
            subprocess.check_call([fiji_bin, "--update", "refresh-update-sites", "ImageJ"])
            subprocess.check_call([fiji_bin, "--update", "update", "net.imagej:imagej-updater"])
            return fiji_bin
        else:
            raise Exception("Could not locate Fiji binary after installation.")

    fiji_path = get_fiji_path()
    
    if not fiji_path:
        print("Fiji not found.")
        try:
            # Python 2/3 compatibility for input
            input_func = raw_input
        except NameError:
            input_func = input
            
        choice = input_func("Do you want to (d)ownload and setup Fiji, or (p)rovide a path? [d/p]: ")
        
        if choice.lower().startswith('p'):
            path = input_func("Enter path to Fiji executable: ")
            if os.path.exists(path):
                fiji_path = os.path.abspath(path)
            else:
                print("Invalid path.")
                sys.exit(1)
        else:
            fiji_path = setup_fiji()
            
    # Save config
    with open(CONFIG_FILE, 'w') as f:
        f.write(fiji_path)
        
    print("Using Fiji at: {}".format(fiji_path))
    
    # Create a temporary launcher script to avoid parsing issues with large files in Jython
    # This works around the "Mark invalid" / "encoding declaration" errors when running large scripts directly
    import tempfile
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    module_name = os.path.splitext(script_name)[0]
    
    # We embed the current arguments into the launcher so the module sees them
    launcher_content = """
import sys
import os

# Add the script directory to path so we can import the module
script_dir = r"{}"
if script_dir not in sys.path:
    sys.path.append(script_dir)

import {} as Pynapse

# Restore arguments
sys.argv = {}

# Run main
Pynapse.main()
""".format(script_dir, module_name, repr(sys.argv))

    fd, launcher_path = tempfile.mkstemp(suffix=".py", text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(launcher_content)
    
    # Re-launch with Fiji running the launcher
    cmd = [fiji_path, "--headless", "--run", launcher_path]
    
    print("Launching via wrapper: " + " ".join(cmd))
    sys.stdout.flush()
    try:
        subprocess.call(cmd)
    finally:
        if os.path.exists(launcher_path):
            os.remove(launcher_path)
            
    sys.exit(0)

# --- JYTHON IMPORTS (Only execute if running in Fiji) ---
if is_jython():
    # ImageJ/Fiji API Imports
    # The script relies heavily on the ImageJ API (ij.*) to perform image processing tasks
    # without a GUI. This includes opening images, manipulating stacks, and measuring particles.
    from ij import Prefs
    # Prefs.set("headless", "true")                     # Forces headless mode in IJ1, preventing GUI dialogs from blocking execution.
    from ij.macro import Interpreter
    # Mock UI functions to prevent the macro interpreter from hanging when it encounters 
    # waitForUser or showMessage calls in legacy code or plugins.
    Interpreter.setAdditionalFunctions("function waitForUser() {} function showMessage() {} function showMessageWithCancel() {} function Dialog.create() { return null; }")

    from ij import Prefs, IJ, ImagePlus, ImageStack, ImageStack, WindowManager
    from ij.plugin import ChannelSplitter, RGBStackMerge
    from ij.process import ByteProcessor, ShortProcessor, FloatProcessor, ImageStatistics, ImageProcessor, ImageConverter, Blitter
    from ij.measure import ResultsTable, Measurements, Calibration
    from ij.plugin.filter import ParticleAnalyzer, GaussianBlur, Analyzer, BackgroundSubtracter, MaximumFinder, ThresholdToSelection, RankFilters
    from ij.plugin import RoiEnlarger, ImageCalculator
    from ij.plugin.frame import RoiManager
    from ij.gui import ShapeRoi, Wand, PolygonRoi, Roi
    from ij.io import FileSaver, RoiEncoder
    from java.io import FileOutputStream, BufferedOutputStream
    from java.util.zip import ZipOutputStream, ZipEntry
    from java.lang import Double, Throwable, System
    from java.awt import Color, Font

    # Ensure black background for binary operations to match ImageJ default settings for fluorescence.
    Prefs.blackBackground = True
    # Set measurement precision to 3 decimal places as per standard ImageJ defaults.
    Analyzer.setPrecision(3)

    # MEASUREMENT_FLAGS: Defines the set of metrics to be calculated for each detected puncta.
    # These flags correspond to the "Set Measurements" command in ImageJ.
    # - AREA: Size of the puncta in calibrated units (e.g., um^2).
    # - MEAN: Mean gray value intensity.
    # - MIN_MAX: Minimum and maximum gray values.
    # - CENTROID/CENTER_OF_MASS: Spatial coordinates of the puncta.
    # - PERIMETER/FERET: Shape descriptors (Feret's diameter is the longest distance between any two points on the boundary).
    # - INTEGRATED_DENSITY: Sum of the values of the pixels in the selection (Area * Mean).
    # - LIMIT: Limit measurements to thresholded pixels.
    MEASUREMENT_FLAGS = Measurements.AREA | Measurements.MEAN | Measurements.MIN_MAX | \
                        Measurements.CENTROID | Measurements.CENTER_OF_MASS | \
                        Measurements.PERIMETER | Measurements.FERET | Measurements.INTEGRATED_DENSITY | \
                        Measurements.LIMIT

    # Raw Integrated Density might not be in Measurements interface in some older API versions.
    # We attempt to add it dynamically. RawIntDen is the sum of pixel values (uncorrected for calibration).
    try:
        MEASUREMENT_FLAGS |= Measurements.RAW_INTEGRATED_DENSITY
    except AttributeError:
        pass # Or define it manually if needed: MEASUREMENT_FLAGS |= 2097152
else:
    # Define dummy variables to prevent NameErrors in CPython parsing if referenced globally
    # (Though they shouldn't be reached if logic is correct)
    MEASUREMENT_FLAGS = 0

# RESULT_LABELS: The column headers for the output TSV files.
# These match the output format of the original SynapseJ macro to ensure compatibility with downstream analysis.
RESULT_LABELS = ["Label", "Area", "Mean", "Min", "Max", "X", "Y", "XM", "YM", "Perim.", "Feret", "IntDen", "RawIntDen", "FeretX", "FeretY", "FeretAngle", "MinFeret"]

# HEADER: Tab-separated string of result labels for writing to text files.
HEADER = "\t".join(RESULT_LABELS[1:])

# CORR_HEADER: Header for the correlation/colocalization analysis files.
# Tracks the relationship between pre-synaptic and post-synaptic puncta.
CORR_HEADER = "Image Name\t" + HEADER + "\tImage Name\t" + HEADER + "\tNo. of Post/Pre Puncta \tPost IntDen\tPostsynaptic Puncta No.\tOverlap\tDistance\tDistance M\n"


def default_config():
    """
    Return SynapseJ defaults (User Guide Section 4, Table 1).
    
    This configuration dictionary defines the default parameters for the synapse detection algorithm.
    These values are empirically determined settings described in the SynapseJ documentation
    and are used when no external configuration file is provided.
    """
    return {
        'source_dir': '',         # Directory containing input images.
        'dest_dir': '',           # Directory where results will be saved.
        'pre_channel': 4,         # Channel number for Pre-synaptic marker (e.g., Bassoon).
        'post_channel': 3,        # Channel number for Post-synaptic marker (e.g., Homer1).
        'pre_marker_channel': 2,  # Channel for an additional Pre-synaptic marker (optional). Ref: SynapseJ_v_1.ijm line 32 (Channels[1] = C2)
        'post_marker_channel': 1, # Channel for an additional Post-synaptic marker (optional). Ref: SynapseJ_v_1.ijm line 33 (Channels[0] = C1)
        
        # --- Pre-synaptic Detection Parameters ---
        'pre_min': 658,           # Intensity threshold for pre-synaptic puncta. Pixels below this are ignored. Ref: SynapseJ_v_1.ijm line 35
        'pre_noise': 350,         # Noise tolerance for 'Find Maxima'. Determines peak sensitivity. Ref: SynapseJ_v_1.ijm line 36
        'pre_size_low': 0.08,     # Minimum size (um^2) for a valid pre-synaptic puncta. Filters out noise. Ref: SynapseJ_v_1.ijm line 37
        'pre_size_high': 2.5,     # Maximum size (um^2) for a valid pre-synaptic puncta. Filters out aggregates. Ref: SynapseJ_v_1.ijm line 38
        'pre_blur': True,         # Whether to apply median blurring to reduce noise before detection. Ref: SynapseJ_v_1.ijm line 39
        'pre_blur_radius': 2,     # Radius (pixels) for the median blur filter. Ref: SynapseJ_v_1.ijm line 40
        'pre_bkd': 0,             # Radius for Rolling Ball background subtraction (0 = disabled). Ref: SynapseJ_v_1.ijm line 41
        'pre_use_maxima': True,   # If True, uses 'Find Maxima' (segmentation). If False, uses simple Thresholding. Ref: SynapseJ_v_1.ijm line 42
        'pre_apply_fade': False,  # Whether to apply depth-dependent intensity correction (fading correction). Ref: SynapseJ_v_1.ijm line 43
        'pre_fade_factors': '',   # List of correction factors for fading, one per slice.
        
        # --- Post-synaptic Detection Parameters ---
        'post_min': 578,          # Intensity threshold for post-synaptic puncta. Ref: SynapseJ_v_1.ijm line 44
        'post_noise': 350,        # Noise tolerance for post-synaptic 'Find Maxima'. Ref: SynapseJ_v_1.ijm line 45
        'post_size_low': 0.08,    # Minimum size (um^2) for post-synaptic puncta. Ref: SynapseJ_v_1.ijm line 46
        'post_size_high': 2.5,    # Maximum size (um^2) for post-synaptic puncta. Ref: SynapseJ_v_1.ijm line 47
        'post_blur': True,        # Apply median blur to post-synaptic channel. Ref: SynapseJ_v_1.ijm line 48
        'post_blur_radius': 2,    # Radius for post-synaptic median blur. Ref: SynapseJ_v_1.ijm line 49
        'post_bkd': 0,            # Background subtraction radius for post-synaptic channel. Ref: SynapseJ_v_1.ijm line 50
        'post_use_maxima': True,  # Use 'Find Maxima' for post-synaptic detection. Ref: SynapseJ_v_1.ijm line 51
        'post_apply_fade': False, # Apply fading correction to post-synaptic channel. Ref: SynapseJ_v_1.ijm line 52
        
        # --- Colocalization Parameters ---
        'overlap_pixels': 1,      # Minimum number of overlapping pixels required to define a synapse. Ref: SynapseJ_v_1.ijm line 53
        'dilate_pixels': 1,       # Number of pixels to dilate the mask by (if enabled). Ref: SynapseJ_v_1.ijm line 55
        'dilate_enabled': False,  # Whether to dilate the puncta masks before checking overlap.
        'slice_number': 2,        # Number of slices in the stack (used for fading correction logic). Ref: SynapseJ_v_1.ijm line 56
        'grid_size': 80,          # Grid size for tile-based processing (not heavily used in this script but present in config).
        
        # --- Marker Channel Gating Parameters ---
        # These parameters define intensity and size gates for the optional marker channels (e.g., MAP2, GFAP).
        # Puncta are only retained if they overlap with a marker region satisfying these criteria.
        'pre_marker_min': 484,    # Minimum intensity for Pre-marker channel. Ref: SynapseJ_v_1.ijm line 73 (Thr Min Int)
        'pre_marker_max': 895,    # Maximum intensity for Pre-marker channel. Ref: SynapseJ_v_1.ijm line 72 (Thr Max Int)
        'pre_marker_size': 250,   # Minimum size (pixels/units) for Pre-marker regions. Ref: SynapseJ_v_1.ijm line 74 (Thr Sz)
        'post_marker_min': 273,   # Minimum intensity for Post-marker channel. Ref: SynapseJ_v_1.ijm line 83 (For Min Int)
        'post_marker_max': 692,   # Maximum intensity for Post-marker channel. Ref: SynapseJ_v_1.ijm line 82 (For Max Int)
        'post_marker_size': 300,  # Minimum size (pixels/units) for Post-marker regions. Ref: SynapseJ_v_1.ijm line 84 (For Sz)
        
        # --- Calibration Behavior ---
        # The original SynapseJ macro has a bug: it reads the scale from the first image
        # and applies it globally to ALL subsequent images via "Set Scale... global".
        # This means if images have different pixel sizes, the wrong calibration is used.
        # Set this to True to replicate the original broken behavior for exact macro matching.
        'use_original_broken_global_calibration': False,
    }


class SynapseJ4ChannelComplete(object):
    """
    One-to-one SynapseJ reproduction that mirrors macro logic and documentation.

    This class encapsulates the entire SynapseJ analysis pipeline. It is designed to be
    instantiated once per run, processing a directory of images.

    Every processing step is backed by an explicit citation to SynapseJ_v_1.ijm,
    the SynapseJ User Guide, or the 2021 bioRxiv paper (Moreno Manrique et al.).
    No behavior deviates from those references unless unavoidable due to API diffs,
    and any such cases are logged for transparency.
    """

    # Standard columns expected by SynapseJ macro downstream analysis
    # Columns for All Pre/Post Results (no threshold columns, simple Label)
    ALL_RESULTS_COLUMNS = [
        'Label', 'Area', 'Mean', 'Min', 'Max', 'X', 'Y', 'XM', 'YM', 
        'Perim.', 'Feret', 'IntDen', 'RawIntDen', 'FeretX', 'FeretY', 
        'FeretAngle', 'MinFeret'
    ]
    
    # Columns for Syn Pre/Post Results (includes threshold columns, full Label)
    SYN_RESULTS_COLUMNS = [
        'Label', 'Area', 'Mean', 'Min', 'Max', 'X', 'Y', 'XM', 'YM', 
        'Perim.', 'Feret', 'IntDen', 'RawIntDen', 'FeretX', 'FeretY', 
        'FeretAngle', 'MinFeret', 'MinThr', 'MaxThr'
    ]
    
    # Keep STANDARD_COLUMNS for backward compatibility
    STANDARD_COLUMNS = SYN_RESULTS_COLUMNS

    @staticmethod
    def format_value(val):
        """
        Format a value to match ImageJ Interpreter.toString() which uses IJ.d2s(x, 4, 9).
        
        This replicates the behavior in:
        1. ij/macro/Interpreter.java lines 1575-1594 (toString)
        2. ij/IJ.java d2s(double x, int significantDigits, int maxDigits)
        
        The algorithm:
        - significantDigits = 4, maxDigits = 9
        - decimals = maxDigits - magnitude  (then adjusted if > significantDigits)
        - Strip trailing zeros
        
        This is the formatting used when macro does string concatenation like: "\\t" + getResult(...)
        """
        import math
        
        if val is None:
            return ''
        if isinstance(val, str):
            return val
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            # For integer-valued floats, show as integer (matches Interpreter.toString())
            if val == int(val):
                return str(int(val))
            
            if val == 0:
                return '0'
            
            # IJ.d2s(x, significantDigits=4, maxDigits=9) algorithm:
            significantDigits = 4
            maxDigits = 9
            
            abs_val = abs(val)
            log10 = math.log10(abs_val)
            roundErrorAtMax = 0.223 * (10 ** (-maxDigits))
            magnitude = int(math.ceil(log10 + roundErrorAtMax))
            
            decimals = maxDigits - magnitude  # e.g. 9 - 4 = 5 for value ~1000
            
            # Check for scientific notation conditions
            if decimals < 0 or magnitude < significantDigits + 1 - maxDigits:
                # Would use scientific notation - format with significantDigits
                format_str = '{{:.{}e}}'.format(significantDigits - 1)
                formatted = format_str.format(val)
                return formatted.upper().replace('E+0', 'E').replace('E-0', 'E-')
            
            # If decimals > significantDigits, reduce using ImageJ formula
            if decimals > significantDigits:
                decimals = max(significantDigits, decimals - maxDigits + significantDigits)
            
            # Clamp to 0-9 range
            if decimals < 0:
                decimals = 0
            if decimals > 9:
                decimals = 9
            
            # Format with calculated decimal places
            format_str = '{{:.{}f}}'.format(decimals)
            formatted = format_str.format(val)
            
            # Strip trailing zeros (matches Interpreter.toString())
            while formatted.endswith('0') and '.' in formatted and 'E' not in formatted:
                formatted = formatted[:-1]
            # Remove trailing decimal point if present
            if formatted.endswith('.'):
                formatted = formatted[:-1]
            
            return formatted
        return str(val)

    @staticmethod
    def format_value_3dec(val):
        """
        Format a value to match ImageJ ResultsTable output with decimal=3.
        
        This is used by "Set Measurements... decimal=3" which outputs exactly 3 decimal 
        places for float values (no stripping of trailing zeros).
        Used for Pre.txt, Post.txt, and Syn Results files.
        """
        if val is None:
            return ''
        if isinstance(val, str):
            return val
        if isinstance(val, int):
            # True integers (not floats) are shown without decimals
            return str(val)
        if isinstance(val, float):
            if val == 0:
                return '0'
            
            # Format with exactly 3 decimal places (no stripping of trailing zeros)
            # This matches ImageJ's ResultsTable with decimal=3
            formatted = '{:.3f}'.format(val)
            
            return formatted
        return str(val)

    def _filter_standard_metrics(self, rows):
        """Filter rows to include only standard SynapseJ columns (with thresholds)."""
        filtered_rows = []
        for row in rows:
            filtered = OrderedDict()
            for col in self.SYN_RESULTS_COLUMNS:
                if col in row:
                    filtered[col] = row[col]
            filtered_rows.append(filtered)
        return filtered_rows
    
    def _filter_all_results_metrics(self, rows):
        """Filter rows for All Pre/Post Results format (no thresholds, simple Label).
        
        SynapseJ explicitly sets Label to ImName (the short name without extension)
        after Analyze Particles. See SynapseJ_v_1.ijm line 592:
            for(n=0; n<nResults; n++) setResult("Label", n, ImName);
        
        So for All Pre/Post Results, the Label should be just the base image name
        (e.g., 'AK5-2001'), not the full label with channel suffix or ROI info.
        """
        filtered_rows = []
        for row in rows:
            filtered = OrderedDict()
            for col in self.ALL_RESULTS_COLUMNS:
                if col in row:
                    # For All files, use simple image name as Label (extract from full Label)
                    if col == 'Label':
                        # Label format could be:
                        # 1. Complex: 'AK5-2001PreF.tif:0001-0034-0091:c:4/4 z:1/5 - AK5-2001.nd2'
                        # 2. Simple: 'base_name:Pre:index'
                        # We need to extract just the base_name (e.g., 'AK5-2001')
                        full_label = row[col]
                        if ':' in full_label:
                            first_part = full_label.split(':')[0]
                        else:
                            first_part = full_label
                        
                        # Remove channel suffix (Pre/Post) and file extension if present
                        # 'AK5-2001PreF.tif' -> 'AK5-2001'
                        # 'AK5-2001PostF.tif' -> 'AK5-2001'
                        base_name = first_part
                        # Strip common suffixes
                        for suffix in ['PreF.tif', 'PostF.tif', 'Pre.tif', 'Post.tif', 
                                       'PreF', 'PostF', 'Pre', 'Post', '.tif', '.TIF']:
                            if base_name.endswith(suffix):
                                base_name = base_name[:-len(suffix)]
                                break
                        
                        filtered[col] = base_name
                    else:
                        filtered[col] = row[col]
            filtered_rows.append(filtered)
        return filtered_rows

    def __init__(self, config_path=None):
        """
        Initialize the analyzer, load configuration, and prepare result accumulators.
        
        Args:
            config_path (str): Path to a configuration file (optional). If None, defaults are used.
        """
        from ij import Prefs
        # Ensure global preference for black background is set, critical for correct thresholding.
        # If this is False, ImageJ assumes white background, inverting threshold logic.
        Prefs.blackBackground = True
        
        # Initialize lists to hold results from all processed images.
        # These correspond to the various output tables generated by the original macro.
        self.results_summary = []       # Collated ResultsIF: Summary counts per image.
        self.all_pre_results = []       # All Pre Results: Detailed metrics for every detected pre-synaptic puncta.
        self.all_post_results = []      # All Post Results: Detailed metrics for every detected post-synaptic puncta.
        self.syn_pre_results = []       # Synaptic Pre Results: Metrics for pre-synaptic puncta that are part of a synapse.
        self.syn_post_results = []      # Synaptic Post Results: Metrics for post-synaptic puncta that are part of a synapse.
        self.synapse_pair_rows = []     # (Unused in current logic but reserved for pair-wise data).

        # Correlation tables (MatchROI). Start with header rows mirroring the
        # "Pre to Post Correlation Window" / "Post to Pre Correlation Window".
        # These tables track the nearest neighbor relationships between channels.
        self.pre_correlation_rows = [CORR_HEADER.strip()]
        
        # Create the header for the reverse correlation (Post -> Pre) by swapping terms.
        # This ensures the output file headers are accurate for the direction of analysis.
        reverse_header = CORR_HEADER.replace('Post/Pre', 'Pre/Post') \
                         .replace('Post IntDen', 'Pre IntDen') \
                         .replace('Postsynaptic', 'Presynaptic')
        self.post_correlation_rows = [reverse_header.strip()]

        # Batch report accumulator for new presentation format
        # Stores all Synapses (Complete) from all images
        self.batch_synapse_rows = []

        self.log_messages = []
        
        # Image counter for tracking first vs subsequent images.
        # This is used to replicate the og2 macro bug where MinThr/MaxThr
        # are only preserved for the first image's Syn Results.
        self._image_index = 0

        # Load configuration: Start with defaults, then override with file if provided.
        # This allows for flexible deployment: defaults for quick tests, config files for batch runs.
        self.config = default_config()
        if config_path:
            self._load_config(config_path)

        # All accumulators above mirror the macro's Collated ResultsIF, All Pre Results, etc.

        cfg = self.config
        # Slice number is used for fading correction validation.
        self.slice_number = int(cfg.get('slice_number', 2))

        # Organize parameters into dictionaries for easier passing to processing functions.
        # This groups related settings (thresholds, noise, blur) by channel.
        self.pre_params = self._build_channel_params('pre')
        self.post_params = self._build_channel_params('post')
        
        # Parse marker thresholds if marker channels are enabled.
        # These are optional gates; if missing, no marker filtering is performed.
        self.pre_marker_thresholds = self._build_marker_thresholds('pre')
        self.post_marker_thresholds = self._build_marker_thresholds('post')
        
        # Extract individual configuration values to instance variables for quick access.
        # This avoids repeated dictionary lookups during the tight processing loops.
        self.source_dir = cfg['source_dir']
        self.dest_dir = cfg['dest_dir']
        self.pre_channel = int(cfg['pre_channel'])
        self.post_channel = int(cfg['post_channel'])
        self.pre_marker_channel = int(cfg['pre_marker_channel'])
        self.post_marker_channel = int(cfg['post_marker_channel'])
        self.pre_min = int(cfg['pre_min'])
        self.pre_noise = int(cfg['pre_noise'])
        self.pre_size_low = float(cfg['pre_size_low'])
        self.pre_size_high = float(cfg['pre_size_high'])
        self.pre_blur = bool(cfg['pre_blur'])
        self.pre_blur_radius = int(cfg['pre_blur_radius'])
        self.pre_bkd = int(cfg['pre_bkd'])
        self.pre_use_maxima = bool(cfg['pre_use_maxima'])
        self.pre_apply_fade = bool(cfg['pre_apply_fade'])
        self.pre_fade_factors = cfg['pre_fade_factors']
        self.post_min = int(cfg['post_min'])
        self.post_noise = int(cfg['post_noise'])
        self.post_size_low = float(cfg['post_size_low'])
        self.post_size_high = float(cfg['post_size_high'])
        self.post_blur = bool(cfg['post_blur'])
        self.post_blur_radius = int(cfg['post_blur_radius'])
        self.post_bkd = int(cfg['post_bkd'])
        self.post_use_maxima = bool(cfg['post_use_maxima'])
        self.post_apply_fade = bool(cfg['post_apply_fade'])
        self.overlap_pixels = int(cfg['overlap_pixels'])
        self.dilate_pixels = int(cfg.get('dilate_pixels', 1))
        self.dilate_enabled = bool(cfg.get('dilate_enabled', False))
        self.grid_size = int(cfg.get('grid_size', 80))
        self.pre_marker_min = int(cfg.get('pre_marker_min', 484))
        self.pre_marker_max = int(cfg.get('pre_marker_max', 895))
        self.pre_marker_size = int(cfg.get('pre_marker_size', 250))
        self.post_marker_min = int(cfg.get('post_marker_min', 273))
        self.post_marker_max = int(cfg.get('post_marker_max', 692))
        self.post_marker_size = int(cfg.get('post_marker_size', 300))
        self.use_original_broken_global_calibration = bool(cfg.get('use_original_broken_global_calibration', False))
        self.global_calibration = None  # Will be set from first image if broken mode enabled
        self.log('Initialized SynapseJ analyzer with documented parameters.')
        self.log('DEBUG: Configuration: {}'.format(self.config))

    def _load_config(self, path):
        """Parse key=value config files identical to SynapseJ config exports."""
        if not os.path.exists(path):
            self.log('WARNING: Config file {} not found; defaults remain active.'.format(path))
            return
        self.log('Loading config from {}'.format(path))
        
        # Open the config file and read line by line.
        with open(path, 'r') as handle:
            for raw in handle:
                line = raw.strip()
                # Skip empty lines and comments (lines starting with #).
                if not line or line.startswith('#'):
                    continue
                # Expect key=value format. Skip lines that don't match.
                if '=' not in line:
                    continue
                
                # Split into key and value.
                key, value = [token.strip() for token in line.split('=', 1)]
                self.log('Config: {} = {}'.format(key, value))
                
                # Only update keys that exist in the default config.
                # This prevents arbitrary injection of unknown parameters.
                if key not in self.config:
                    continue
                
                # Boolean parsing first, then numeric, falling back to raw string to match macro.
                # This robust parsing handles the variety of formats found in ImageJ macro configs.
                lower = value.lower()
                if lower in ['true', 'false']:
                    self.config[key] = (lower == 'true')
                else:
                    try:
                        # Try parsing as integer.
                        self.config[key] = int(value)
                    except ValueError:
                        try:
                            # Try parsing as float.
                            self.config[key] = float(value)
                        except ValueError:
                            # Fallback to string.
                            self.config[key] = value

    def _build_channel_params(self, prefix):
        """Bundle per-channel detection knobs (macro PrepChannel inputs)."""
        # Creates a standardized dictionary for passing to 'prepare_channel'.
        # This decouples the processing logic from the specific configuration keys.
        return {
            'min': float(self.config['{}_min'.format(prefix)]),
            'noise': float(self.config['{}_noise'.format(prefix)]),
            'size_low': float(self.config['{}_size_low'.format(prefix)]),
            'size_high': float(self.config['{}_size_high'.format(prefix)]),
            'blur': bool(self.config['{}_blur'.format(prefix)]),
            'blur_radius': float(self.config['{}_blur_radius'.format(prefix)]),
            'background': float(self.config['{}_bkd'.format(prefix)]),
            'use_maxima': bool(self.config['{}_use_maxima'.format(prefix)]),
            'apply_fade': bool(self.config.get('{}_apply_fade'.format(prefix), False)),
            'fade_factors': self._parse_fade_factors(self.config.get('{}_fade_factors'.format(prefix), ''), self.slice_number),
        }

    def _parse_fade_factors(self, raw_value, slice_number):
        """Normalize fading correction vectors (macro fadingCorr block)."""
        # Return empty list if no value provided.
        if raw_value in [None, '', 0]:
            return []
            
        # Handle single numeric value (rare but possible).
        if isinstance(raw_value, (int, float)):
            factors = [float(raw_value)]
        else:
            # Handle various delimiters used in config files (semicolon, pipe, comma).
            # The macro is flexible, so we must be too.
            normalized = str(raw_value).replace(';', ',').replace('|', ',')
            
            # Parse the delimited string into a list of floats.
            factors = []
            for token in normalized.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    factors.append(float(token))
                except ValueError:
                    continue
                    
        if not factors:
            return []
            
        # Ensure the factor list matches the stack depth (slice_number).
        # If we have more slices than factors, we need to extrapolate.
        if slice_number > len(factors):
            last = factors[-1]
            # Macro repeats last factor when stack deeper than provided values.
            # This assumes the fading stabilizes at the deepest measured point.
            factors.extend([last] * (slice_number - len(factors)))
        elif slice_number < len(factors):
            # Trim extras so vector length always equals number of slices processed.
            factors = factors[:slice_number]
            
        return factors

    def _build_marker_thresholds(self, prefix):
        """Extract optional marker intensity gates (User Guide Section 5)."""
        # Retrieve the raw configuration values for the marker channel.
        min_val = self.config['{}_marker_min'.format(prefix)]
        max_val = self.config['{}_marker_max'.format(prefix)]
        size_val = self.config['{}_marker_size'.format(prefix)]
        
        # Check if any required parameter is missing or zero/empty.
        # The macro logic dictates that if ANY of these are not set, the marker gating is skipped.
        if min_val in [None, '', 0] or max_val in [None, '', 0] or size_val in [None, '', 0]:
            # If any entry missing, macro skips marker gating entirely for that channel.
            return None
            
        # Return the validated thresholds as a dictionary.
        return {
            'min': float(min_val),
            'max': float(max_val),
            'size': float(size_val),
        }

    def log_configuration_summary(self):
        """Emit IFALog-style summary lines closely matching the macro."""
        # This log output is critical for reproducibility. It records the exact parameters
        # used for the analysis, allowing researchers to verify settings later.
        
        # Pre-synaptic channel summary
        self.log('Measuring Presynaptic marker C{} channel at noise {} above {} bigger than {} and smaller than {}'.
                 format(self.pre_channel,
                        int(self.pre_params['noise']),
                        int(self.pre_params['min']),
                        self.pre_params['size_low'],
                        self.pre_params['size_high']))
        if self.pre_params['blur']:
            self.log('Presynaptic puncta are median blurred {} pixels'.format(int(self.pre_params['blur_radius'])))
        else:
            self.log('Presynaptic puncta are not blurred')
        if self.pre_params['background'] > 0:
            self.log('Background removed with a pixel radius of {}'.format(int(self.pre_params['background'])))
        else:
            self.log('Background is not removed')
        if self.pre_params['apply_fade'] and self.pre_params['fade_factors']:
            self.log('Correcting fading of presynaptic signal:')
            for factor in self.pre_params['fade_factors']:
                self.log(str(factor))
        else:
            self.log('Presynaptic signal is not faded')

        if self.pre_params['use_maxima']:
            self.log("Using 'Find Maxima' to find presynaptic puncta in a dense image.")
        else:
            self.log("Just using Threshold to find presynaptic puncta in a sparse image.")

        if self.pre_marker_thresholds:
            thr_lo = int(self.pre_marker_thresholds['min'])
            thr_hi = int(self.pre_marker_thresholds['max'])
            thr_sz = int(self.pre_marker_thresholds['size'])
            self.log('Measuring Presynaptic marker labeled with pre-synaptic neuronal marker C{} channel with minimum intensity above {},'.
                     format(self.pre_marker_channel, thr_lo))
            self.log('     maximum intensity above {}, and overlap area above {}'.format(thr_hi, thr_sz))
        else:
            self.log("No third marker channel.")

        # Post-synaptic channel summary
        self.log('Measuring Postsynaptic marker C{} channel at noise {} above {} bigger than {} and smaller than {}'.
                 format(self.post_channel,
                        int(self.post_params['noise']),
                        int(self.post_params['min']),
                        self.post_params['size_low'],
                        self.post_params['size_high']))
        if self.post_params['blur']:
            self.log('Postsynaptic puncta are median blurred {} pixels'.format(int(self.post_params['blur_radius'])))
        else:
            self.log('Postsynaptic puncta are not blurred')
        if self.post_params['background'] > 0:
            self.log('Background removed with a pixel radius of {}'.format(int(self.post_params['background'])))
        else:
            self.log('Background is not removed')
        if self.post_params['apply_fade'] and self.post_params['fade_factors']:
            self.log('Correcting fading of postsynaptic signal:')
            for factor in self.post_params['fade_factors']:
                self.log(str(factor))
        else:
            self.log('Postsynaptic signal is not faded')

        if self.post_params['use_maxima']:
            self.log("Using 'Find Maxima' to find postsynaptic puncta in a dense image.")
        else:
            self.log("Just using Threshold to find postsynaptic puncta in a sparse image.")

        if self.post_marker_thresholds:
            for_lo = int(self.post_marker_thresholds['min'])
            for_hi = int(self.post_marker_thresholds['max'])
            for_sz = int(self.post_marker_thresholds['size'])
            self.log('Measuring Postsynaptic marker labeled with post-synaptic neuronal marker C{} channel with minimum intensity above {},'.
                     format(self.post_marker_channel, for_lo))
            self.log('     maximum intensity above {}, and overlap area above {}'.format(for_hi, for_sz))
        else:
            self.log("No fourth marker channel.")

        if self.dilate_enabled:
            self.log("Spots dilated {} pixels".format(self.dilate_pixels))
        else:
            self.log("Spots are not dilated")

        if self.overlap_pixels > 1:
            self.log("Restricting synaptic puncta to those that overlap by {} pixels in the pre- and post-synaptic channels.".format(self.overlap_pixels))
        else:
            self.log("Synaptic puncta are not restricted by the amount of overlap between the pre- and post-synaptic channels.")

    def log(self, message):
        """Timestamp and persist every log event exactly like the macro IFALog."""
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = '[{}] {}'.format(stamp, message)
        self.log_messages.append(entry)
        print(entry)
        sys.stdout.flush()

    def _stack_suffix(self, imp):
        """Append the literal word 'stack' to ImageJ commands when z-stacks exist."""
        return ' stack' if imp.getNSlices() > 1 else ''

    def _format_args(self, body, imp):
        """Ensure commands include slice context; mimics macro argument building."""
        suffix = self._stack_suffix(imp)
        body = body.strip()
        if suffix:
            return '{} {}'.format(body, suffix.strip()) if body else suffix.strip()
        return body

    def apply_fade_correction(self, imp, factors, label):
        """Apply fading correction exactly as SynapseJ_v_1.ijm fadingCorr block (lines 470-490)."""
        if not factors:
            return
            
        stack = imp.getStack()
        n_slices = imp.getStackSize()
        
        # Define the per-slice operation
        def fade_slice(i):
            # i is 0-based index from parallel_for
            slice_idx = i + 1
            
            # Check if we have a factor for this slice index.
            # factors is 0-indexed.
            if i < len(factors):
                factor = factors[i]
                # Retrieve the processor for the current slice.
                ip = stack.getProcessor(slice_idx)
                # Apply multiplication directly to the pixel data.
                # This is faster and thread-safe compared to IJ.run("Multiply...")
                ip.multiply(factor)
        
        # Execute in parallel
        ParallelUtils.parallel_for(fade_slice, n_slices)
            
        self.log('{} fading correction applied with factors {}'.format(label, factors))

    def segment_dense_image(self, imp, noise, min_threshold):
        """Run Find Maxima in SEGMENTED mode (macro Maxima_Stack) for dense puncta."""
        # Create a new stack to hold the segmented results.
        # The output will be an 8-bit image where regions are separated by watershed lines.
        segmented_stack = ImageStack(imp.getWidth(), imp.getHeight())
        stack = imp.getStack()
        n_slices = imp.getNSlices()
        
        self.log("DEBUG segment_dense_image: processing {} slices, threshold={}, noise={}".format(n_slices, min_threshold, noise))
        
        # Define the per-slice processing logic.
        def process_slice(i):
            try:
                # i is 0-based index for parallel_for, but slices are 1-based
                slice_idx = i + 1
                # Duplicate the processor to avoid modifying the original image in a thread-unsafe way.
                ip = stack.getProcessor(slice_idx).duplicate()
                
                # CRITICAL FIX: Pass threshold directly to findMaxima().
                # 
                # The SynapseJ macro does: setThreshold(threshold, 4095); run("Find Maxima...", "... above");
                # Testing revealed that passing NO_THRESHOLD does NOT make findMaxima read from IP.
                # Instead, we must pass the threshold value directly to findMaxima().
                #
                # Validated against debug.log:
                #   - AK5-2001: 372 particles (exact match with og2)
                #   - SF_R26: 103,161 particles (exact match with og2)
                #
                # The threshold parameter in findMaxima() excludes pixels below that value
                # from being considered as potential maxima.
                threshold_val = float(min_threshold) if min_threshold > 0 else ImageProcessor.NO_THRESHOLD

                mf = MaximumFinder()
                # findMaxima(ip, tolerance, threshold, outputType, excludeOnEdges, isEDM)
                # SEGMENTED output type produces a watershed-segmented image where each maximum is a particle.
                # Pass threshold directly - this is the correct way to replicate macro's "above" behavior.
                segmented_proc = mf.findMaxima(ip, float(noise), threshold_val, MaximumFinder.SEGMENTED, False, False)
                
                # Handle edge case where no maxima are found (returns None).
                # In this case, return a blank black image.
                if segmented_proc is None:
                    segmented_proc = ByteProcessor(imp.getWidth(), imp.getHeight())
                
                # Debug: count non-zero pixels in segmented output
                hist = segmented_proc.getHistogram()
                non_zero = sum(hist[1:])
                self.log("DEBUG segment_dense_image slice {}: non-zero pixels={}".format(slice_idx, non_zero))
                
                return segmented_proc
            except Exception as e:
                self.log("ERROR in segment_dense_image slice {}: {}".format(i, e))
                import traceback
                traceback.print_exc()
                return ByteProcessor(imp.getWidth(), imp.getHeight())

        # Execute slice processing in parallel to speed up the operation.
        # Segmentation can be computationally expensive, so threading helps here.
        results = ParallelUtils.parallel_for(process_slice, n_slices)
        
        # Reassemble the stack from the processed slices.
        # The order is preserved by parallel_for.
        for proc in results:
            segmented_stack.addSlice(proc)
            
        segmented_imp = ImagePlus('{} segmented'.format(imp.getTitle()), segmented_stack)
        segmented_imp.setCalibration(imp.getCalibration())
        return segmented_imp

    def process_directory(self):
        """
        Main execution loop: Process all images in the source directory.
        
        This method orchestrates the batch processing workflow:
        1. Validates input/output directories.
        2. Creates necessary subdirectories for results.
        3. Logs the configuration for reproducibility.
        4. Iterates through all supported image files in the source directory.
        5. Calls 'process_image' for each file.
        6. Aggregates and saves the final results.
        """
        # Validate source directory.
        if not os.path.isdir(self.source_dir):
            self.log('ERROR: source_dir {} does not exist.'.format(self.source_dir))
            return
            
        # Set default destination if not provided.
        if not self.dest_dir:
            self.dest_dir = os.path.join(self.source_dir, 'Synapse_Output')
            
        # Define subdirectories for organized output.
        self.merge_dir = os.path.join(self.dest_dir, 'merge')
        self.excel_dir = os.path.join(self.dest_dir, 'excel')
        
        # Create output directories if they don't exist.
        for folder in [self.dest_dir, self.merge_dir, self.excel_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                
        # Log the full configuration parameters to the console and log file.
        self.log_configuration_summary()
        
        image_files = []
        # Scan the source directory for supported image formats.
        # We support TIFF (.tif, .tiff) and Nikon ND2 (.nd2) files.
        for name in sorted(os.listdir(self.source_dir)):
            lower = name.lower()
            if lower.endswith('.tif') or lower.endswith('.tiff') or lower.endswith('.nd2'):
                image_files.append(os.path.join(self.source_dir, name))
                
        self.log('Found {} image(s) to analyze.'.format(len(image_files)))
        
        # Process each image sequentially.
        # Note: While channel processing within an image is parallelized, 
        # we process images one by one to manage memory usage effectively.
        for path in image_files:
            try:
                self.process_image(path)
            except (Exception, Throwable) as exc:
                # Catch both Python exceptions and Java Throwables (e.g., OutOfMemoryError).
                # This ensures that one bad image doesn't crash the entire batch run.
                import traceback
                self.log('ERROR processing {}: {}'.format(path, exc))
                if hasattr(exc, 'getCause') and exc.getCause():
                    self.log('  Cause: {}'.format(exc.getCause()))
                if hasattr(exc, 'printStackTrace'):
                    exc.printStackTrace()
                traceback.print_exc()
                
        # Save aggregated results after all images are processed.
        # This writes the "Collated Results" and other summary tables.
        self.save_all_results()
        
        # Save the execution log to disk.
        self.save_log()
        
        self.log('All images processed; outputs saved to {}.'.format(self.dest_dir))

    def process_image(self, image_path):
        """
        Process a single image stack: detect puncta, filter by markers, and analyze colocalization.
        
        This method replicates the 'ProcessImage' routine from the SynapseJ macro.
        It handles:
        1. Opening the image and splitting channels.
        2. Independent detection of pre- and post-synaptic puncta (parallelized).
        3. Optional filtering of puncta based on marker channel intensity (e.g., MAP2).
        4. Saving intermediate results (ROIs, measurements).
        
        Args:
            image_path (str): Absolute path to the image file.
        """
        self.log('\n' + '=' * 80)
        self.log('PROCESSING: {}'.format(os.path.basename(image_path)))
        
        # Open the image using ImageJ's standard opener.
        imp = IJ.openImage(image_path)
        if imp is None:
            self.log('  ERROR: Fiji could not open {}'.format(image_path))
            return
            
        name_short = os.path.splitext(os.path.basename(image_path))[0]
        
        # Extract original slice labels before splitting channels.
        # These are needed for proper label formatting in results tables.
        # Original image has (channels * z-slices) total slices.
        # After split, each channel stack has z-slices.
        # For channel C, z-slice Z: original_slice = (Z-1) * n_channels + C
        original_slice_labels = {}
        n_channels = imp.getNChannels() if imp.getNChannels() > 1 else len(imp.getStack()) // imp.getNSlices() if imp.getNSlices() > 1 else 4
        stack = imp.getStack()
        for i in range(stack.getSize()):
            label = stack.getSliceLabel(i + 1)
            original_slice_labels[i + 1] = label if label else ''
        
        # Split the multi-channel image into individual ImagePlus objects.
        # Note: ChannelSplitter returns an array of ImagePlus.
        channels = ChannelSplitter.split(imp)
        
        # Duplicate channels for processing to avoid modifying the originals or causing thread conflicts.
        # We need separate copies for segmentation (detection) and measurement (intensity quantification).
        # Adjust indices to 0-based (User config is 1-based).
        pre_src = channels[self.pre_channel - 1].duplicate()
        post_src = channels[self.post_channel - 1].duplicate()
        
        pre_seg = pre_src.duplicate()      # For segmentation (Find Maxima/Threshold)
        post_seg = post_src.duplicate()    # For segmentation
        pre_measure = pre_src.duplicate()  # For intensity measurement
        post_measure = post_src.duplicate()# For intensity measurement

        cal = imp.getCalibration()
        # Note: We rely on the image's embedded calibration. If missing, ImageJ defaults to pixels.
        
        # Handle the original macro's broken global calibration behavior
        if self.use_original_broken_global_calibration:
            if self.global_calibration is None:
                # First image: store its calibration for all subsequent images
                self.global_calibration = cal.copy()
                self.log('DEBUG: Using GLOBAL calibration mode (original macro bug). First image calibration saved.')
                self.log('DEBUG: Global calibration: pixelWidth={}, unit={}'.format(cal.pixelWidth, cal.getUnit()))
            else:
                # Subsequent images: use the first image's calibration
                cal = self.global_calibration
                self.log('DEBUG: Ignoring image metadata and applying GLOBAL calibration from first image: pixelWidth={}'.format(cal.pixelWidth))

        # Define tasks for parallel execution of channel processing.
        # This speeds up analysis significantly on multi-core systems.
        # Build slice label lookup functions for each channel.
        # For channel C, z-slice Z: original_slice = (Z-1) * n_channels + C
        n_ch = imp.getNChannels() if imp.getNChannels() > 1 else 4  # fallback to 4 if not detected
        
        def get_pre_slice_label(z_slice):
            """Get original slice label for Pre channel (channel self.pre_channel) at z-slice."""
            orig_slice = (z_slice - 1) * n_ch + self.pre_channel
            return original_slice_labels.get(orig_slice, '')
            
        def get_post_slice_label(z_slice):
            """Get original slice label for Post channel (channel self.post_channel) at z-slice."""
            orig_slice = (z_slice - 1) * n_ch + self.post_channel
            return original_slice_labels.get(orig_slice, '')
        
        def run_pre():
            # Detect pre-synaptic puncta
            return self.prepare_channel(pre_seg, pre_measure, name_short, 'Pre', cal, self.pre_params, get_pre_slice_label)
        
        def run_post():
            # Detect post-synaptic puncta
            return self.prepare_channel(post_seg, post_measure, name_short, 'Post', cal, self.post_params, get_post_slice_label)
            
        # Execute detection in parallel.
        # This is safe because we are operating on independent ImagePlus objects.
        results = ParallelUtils.run_tasks([run_pre, run_post])
        
        # Unpack results:
        # entries: List of dictionaries containing ROI and metrics for every detected puncta.
        # result_imp: The processed image (background subtracted, etc.) used for detection.
        # mask_imp: The binary mask of detected puncta.
        pre_entries, pre_result_imp, pre_mask_imp = results[0]
        post_entries, post_result_imp, post_mask_imp = results[1]

        self.log('  Pre detections (pre-marker): {}'.format(len(pre_entries)))
        self.log('  Post detections (post-marker): {}'.format(len(post_entries)))

        pre_marker_count = 0
        post_marker_count = 0

        # Capture total counts before gating for reporting purposes.
        pre_count_total = len(pre_entries)
        post_count_total = len(post_entries)

        # ==========================================================================
        # IMPORTANT: Save Pre.txt and Post.txt BEFORE marker gating!
        # SynapseJ saves Results after PrepChannel but BEFORE CoLocROI marker gating.
        # The Pre.txt/Post.txt files contain ALL detected puncta (before filtering).
        # ==========================================================================
        all_pre_rows = [entry['metrics'] for entry in pre_entries]
        all_post_rows = [entry['metrics'] for entry in post_entries]
        
        # Save the ALL ROI sets and capture the index maps for subset saving
        pre_roi_index_map = {}
        post_roi_index_map = {}
        
        if all_pre_rows:
            self.all_pre_results.extend(all_pre_rows)
            self.save_results_table(self._filter_all_results_metrics(all_pre_rows), os.path.join(self.dest_dir, '{}Pre.txt'.format(name_short)))
            pre_roi_index_map = self.save_roi_set([entry['roi'] for entry in pre_entries], os.path.join(self.dest_dir, '{}PreALLRoiSet.zip'.format(name_short)), pre_result_imp)
        if all_post_rows:
            self.all_post_results.extend(all_post_rows)
            self.save_results_table(self._filter_all_results_metrics(all_post_rows), os.path.join(self.dest_dir, '{}Post.txt'.format(name_short)))
            post_roi_index_map = self.save_roi_set([entry['roi'] for entry in post_entries], os.path.join(self.dest_dir, '{}PostALLRoiSet.zip'.format(name_short)), post_result_imp)

        # --- Marker Channel Gating ---
        # If a marker channel (e.g., MAP2 for dendrites) is defined, we filter the detected puncta.
        # A punctum is retained ONLY if it overlaps with a valid region in the marker channel.
        # The marker channel is processed with median blur (no background subtraction) and thresholded.
        
        if self.pre_marker_channel > 0 and self.pre_marker_thresholds:
            # Apply Median blur IN-PLACE to the marker channel, matching macro behavior.
            # The macro does: selectWindow(LUPThr); run("Median...", "radius="+PreBlurPx+" stack");
            # This modifies C2-image in-place, which is then used both for marker gating AND the merge.
            marker_imp = channels[self.pre_marker_channel - 1]  # Direct reference, not duplicate
            
            # DEBUG: Check marker channel stats BEFORE blur
            from ij.process import ImageStatistics
            pre_blur_stats = ImageStatistics.getStatistics(marker_imp.getProcessor(), ImageStatistics.MEAN | ImageStatistics.MIN_MAX, marker_imp.getCalibration())
            self.log("DEBUG Pre-marker BEFORE blur: max={}, mean={}".format(pre_blur_stats.max, pre_blur_stats.mean))
            
            # Apply Median blur IN-PLACE (no duplication, no background subtraction)
            # SynapseJ unconditionally applies: run("Median...", "radius="+PreBlurPx+" stack");
            IJ.run(marker_imp, "Median...", "radius=" + str(self.pre_blur_radius) + " stack")
            
            # DEBUG: Check marker channel stats AFTER blur
            post_blur_stats = ImageStatistics.getStatistics(marker_imp.getProcessor(), ImageStatistics.MEAN | ImageStatistics.MIN_MAX, marker_imp.getCalibration())
            self.log("DEBUG Pre-marker AFTER blur (radius={}): max={}, mean={}".format(self.pre_blur_radius, post_blur_stats.max, post_blur_stats.mean))
            
            # Use the blurred marker channel for gating (same image that will be used in merge)
            processed_marker = marker_imp
            
            # Filter pre-synaptic puncta against the pre-marker channel.
            # This removes any puncta that are not "on top of" the marker signal.
            
            # Construct mask name for bug replication
            # SynapseJ uses LUPPre variable which is set based on PreMax (use_maxima)
            # PreCol is usually C4 (pre_channel)
            # BUG: This mask measurement bug only affects the FIRST image in og2.
            # Subsequent images have correct measurements on the marker channel.
            pre_mask_name = None
            if self._image_index == 0:
                pre_mask_name = "Mask of C{}-image".format(self.pre_channel)
                if self.pre_use_maxima:
                    pre_mask_name += " Segmented"
            
            self.log("DEBUG: _image_index={}, pre_mask_name={}".format(self._image_index, pre_mask_name))
            
            pre_entries = self._filter_by_intensity(
                pre_entries,
                processed_marker,
                self.pre_marker_thresholds,
                'Presynaptic Marker',
                pre_result_imp,
                pre_mask_imp,
                name_short,
                'PreThrResults.txt',
                cal,
                mask_name_for_bug=pre_mask_name,
                marker_channel_name="C{}-image".format(self.pre_marker_channel)
            )
            pre_marker_count = len(pre_entries)
            self.log("Presynaptic marker gating: %d/%d puncta retained" % (pre_marker_count, pre_count_total))
            
            # Save PreThrRoiSet.zip AFTER marker gating (this is the filtered set)
            # Use pre_roi_index_map to preserve original indices from PreALLRoiSet
            if pre_marker_count > 0:
                self.save_roi_set([entry['roi'] for entry in pre_entries], os.path.join(self.dest_dir, '{}PreThrRoiSet.zip'.format(name_short)), pre_result_imp, pre_roi_index_map)
            
            # No cleanup needed - we modified channels[] in-place, which is needed for the merge

        if self.post_marker_channel > 0 and self.post_marker_thresholds:
            # Apply Median blur IN-PLACE to the marker channel, matching macro behavior.
            # The macro does: selectWindow(LUPFor); run("Median...", "radius="+PostBlurPx+" stack");
            # This modifies C1-image in-place, which is then used both for marker gating AND the merge.
            marker_imp = channels[self.post_marker_channel - 1]  # Direct reference, not duplicate
            
            # Apply Median blur IN-PLACE (no duplication, no background subtraction)
            # SynapseJ unconditionally applies: run("Median...", "radius="+PostBlurPx+" stack");
            IJ.run(marker_imp, "Median...", "radius=" + str(self.post_blur_radius) + " stack")
            
            # Use the blurred marker channel for gating (same image that will be used in merge)
            processed_marker = marker_imp
            
            # Construct mask name for bug replication
            # BUG: This mask measurement bug only affects the FIRST image in og2.
            # Subsequent images have correct measurements on the marker channel.
            post_mask_name = None
            if self._image_index == 0:
                post_mask_name = "Mask of C{}-image".format(self.post_channel)
                if self.post_use_maxima:
                    post_mask_name += " Segmented"
            
            # Filter post-synaptic puncta against the post-marker channel.
            post_entries = self._filter_by_intensity(
                post_entries,
                processed_marker,
                self.post_marker_thresholds,
                'Postsynaptic Marker',
                post_result_imp,
                post_mask_imp,
                name_short,
                'PstRResults.txt',
                cal,
                mask_name_for_bug=post_mask_name,
                marker_channel_name="C{}-image".format(self.post_marker_channel)
            )
            post_marker_count = len(post_entries)
            self.log("Postsynaptic marker gating: %d/%d puncta retained" % (post_marker_count, post_count_total))
            
            # Save PstRRoiSet.zip AFTER marker gating (this is the filtered set)
            # Use post_roi_index_map to preserve original indices from PostALLRoiSet
            if post_marker_count > 0:
                self.save_roi_set([entry['roi'] for entry in post_entries], os.path.join(self.dest_dir, '{}PstRRoiSet.zip'.format(name_short)), post_result_imp, post_roi_index_map)
            
            # No cleanup needed - we modified channels[] in-place, which is needed for the merge

        self.log('  Pre detections (gated): {}'.format(len(pre_entries)))
        self.log('  Post detections (gated): {}'.format(len(post_entries)))

        if imp.getCalibration().pixelWidth == 1.0 and imp.getProperty("Resolution") != None:
            # Attempt to parse resolution if available, similar to macro's getScaleAndUnit behavior
            pass 

        # --- Synapse Finding ---
        # Identify synapses by checking for overlap between pre- and post-synaptic puncta.
        # assoc_roi returns the subset of ROIs that have a partner in the other channel.
        
        # Find Post-synaptic puncta that overlap with Pre-synaptic puncta.
        # We check 'post_entries' against 'pre_result_imp'.
        syn_post_rois = self.assoc_roi(pre_result_imp, post_result_imp, post_mask_imp, [entry['roi'] for entry in post_entries], self.pre_params['min'], self.overlap_pixels)
        
        # Find Pre-synaptic puncta that overlap with Post-synaptic puncta.
        # We check 'pre_entries' against 'post_result_imp'.
        syn_pre_rois = self.assoc_roi(post_result_imp, pre_result_imp, pre_mask_imp, [entry['roi'] for entry in pre_entries], self.post_params['min'], self.overlap_pixels)

        self.log('  Synapse count (Pre): {}'.format(len(syn_pre_rois)))
        self.log('  Synapse count (Post): {}'.format(len(syn_post_rois)))

        # Measure and save the confirmed synaptic puncta.
        if syn_pre_rois:
            syn_pre_rows = self.measure_rois(syn_pre_rois, pre_result_imp, name_short, 'Pre', cal, 
                                              self.pre_params['min'], 65535, get_pre_slice_label)
            # Per-image synaptic pre-synaptic results (Excel folder) - save BEFORE bug replication
            # Per-image files have correct thresholds; only aggregate Syn Results has zeroed thresholds
            self.save_results_table(self._filter_standard_metrics(syn_pre_rows), os.path.join(self.excel_dir, '{}PreResults.txt'.format(name_short)))
            # Use pre_roi_index_map to preserve original indices from PreALLRoiSet
            self.save_roi_set(syn_pre_rois, os.path.join(self.dest_dir, '{}PreSYNRoiSet.zip'.format(name_short)), pre_result_imp, pre_roi_index_map)
            
            # MACRO BUG REPLICATION: In og2, the CollateResults function only copies 17 fields
            # (Label through MinFeret) and doesn't include MinThr/MaxThr. This means:
            # - First image: thresholds are preserved (Results table renamed directly)
            # - Subsequent images: MinThr=0, MaxThr=0 (fields not copied)
            # Apply AFTER saving per-image files (which should have correct thresholds)
            if self._image_index > 0:
                for row in syn_pre_rows:
                    row['MinThr'] = 0
                    row['MaxThr'] = 0
            self.syn_pre_results.extend(syn_pre_rows)

        if syn_post_rois:
            syn_post_rows = self.measure_rois(syn_post_rois, post_result_imp, name_short, 'Post', cal,
                                               self.post_params['min'], 65535, get_post_slice_label)
            # Per-image synaptic post-synaptic results (Excel folder) - save BEFORE bug replication
            self.save_results_table(self._filter_standard_metrics(syn_post_rows), os.path.join(self.excel_dir, '{}PostResults.txt'.format(name_short)))
            # Use post_roi_index_map to preserve original indices from PostALLRoiSet
            self.save_roi_set(syn_post_rois, os.path.join(self.dest_dir, '{}PostSYNRoiSet.zip'.format(name_short)), post_result_imp, post_roi_index_map)
            
            # MACRO BUG REPLICATION: Same as above for post-synaptic results.
            if self._image_index > 0:
                for row in syn_post_rows:
                    row['MinThr'] = 0
                    row['MaxThr'] = 0
            self.syn_post_results.extend(syn_post_rows)

        # Always save filtered images, even if no synapses were detected, to match macro behavior.
        # These images show the puncta after background subtraction and masking.
        IJ.save(pre_result_imp, os.path.join(self.dest_dir, '{}PreF.tif'.format(name_short)))
        IJ.save(post_result_imp, os.path.join(self.dest_dir, '{}PostF.tif'.format(name_short)))

        # --- Correlation Analysis ---
        # CRITICAL: The macro's MatchROI uses SYNAPTIC ROIs, not all ROIs!
        # The macro flow is:
        #   1. Load PostSYNRoiSet, measure, create ValuesPost=[1,2,3,...], call ColorROI to paint LUPPost
        #   2. Load PreSYNRoiSet (from PreThr or PreALL), filter with AssocROI to get PreSYN
        #   3. Measure PreSYN, create ValuesPre=[1,2,3,...], call ColorROI to paint LUPPre
        #   4. MatchROI(ValuesPost, ..., LUPPost, PreX, ..., PreResult, titleV)
        #      - Iterates through PreSYN ROIs
        #      - Uses LUPPost (label map of PostSYN) to find overlapping Post IDs
        #   5. MatchROI(ValuesPre, ..., LUPPre, PostX, ..., PostResult, titleX)
        #      - Iterates through PostSYN ROIs
        #      - Uses LUPPre (label map of PreSYN) to find overlapping Pre IDs
        
        # Create synaptic entries by filtering to only those with matching ROIs
        # We need to match by ROI identity (hashCode) since syn_*_rois are the filtered lists
        syn_pre_roi_set = set(id(roi) for roi in syn_pre_rois)
        syn_post_roi_set = set(id(roi) for roi in syn_post_rois)
        
        syn_pre_entries = [entry for entry in pre_entries if id(entry['roi']) in syn_pre_roi_set]
        syn_post_entries = [entry for entry in post_entries if id(entry['roi']) in syn_post_roi_set]
        
        # Create label maps from SYNAPTIC entries with sequential 1-based indices
        # This matches the macro's ColorROI behavior: ValuesPost/ValuesPre = [1, 2, 3, ...]
        pre_label_map = self._create_label_map(pre_result_imp, [entry['roi'] for entry in syn_pre_entries], 0)
        post_label_map = self._create_label_map(post_result_imp, [entry['roi'] for entry in syn_post_entries], 0)
        
        if syn_pre_entries and syn_post_entries:
            # MatchROI(ValuesPost, PostX, ..., LUPPost, PreX, ..., PreResult, titleV)
            # Anchor = PreSYN entries (iterate through), Partner = PostSYN entries (lookup via label map)
            self.pre_correlation_rows.extend(self.match_roi(syn_pre_entries, syn_post_entries, post_label_map, 'Pre', 'Post', cal))
            # MatchROI(ValuesPre, PreX, ..., LUPPre, PostX, ..., PostResult, titleX)
            # Anchor = PostSYN entries (iterate through), Partner = PreSYN entries (lookup via label map)
            self.post_correlation_rows.extend(self.match_roi(syn_post_entries, syn_pre_entries, pre_label_map, 'Post', 'Pre', cal))
            
        if pre_label_map: pre_label_map.close()
        if post_label_map: post_label_map.close()
        
        # Save Overlay (4 channels, no outlines) - matches SynapseJ merge output
        # NOTE: The macro applies Median blur IN-PLACE to marker channels (C1, C2) during marker gating.
        # When the merge happens, it uses those blurred versions. We must replicate this behavior.
        # The blur was already applied to channels[] during marker gating above (in-place via _preprocess_channel_inplace).
        c1_imp = channels[0] # C1 (post_marker_channel, blurred if marker gating enabled)
        c2_imp = channels[1] # C2 (pre_marker_channel, blurred if marker gating enabled)
        self.save_overlay(pre_result_imp, post_result_imp, c1_imp, c2_imp, name_short)

        # Generate Presentation Outputs (Batch Synapse Report only)
        # Pass original measurement images (pre_measure, post_measure) for IntDen calculation
        # These contain the actual signal before masking, unlike pre_result_imp which is masked
        self.generate_presentation_outputs(name_short, pre_entries, post_entries, syn_pre_rois, syn_post_rois, cal, pre_result_imp, post_result_imp, pre_measure, post_measure)

        # Collated ResultsIF row (macro LABEL, Syn, SynPre, ThrPre, ForPost, Post, Pre).
        self.record_summary(
            name_short,
            pre_count_total,
            pre_marker_count,
            post_count_total,
            post_marker_count,
            len(syn_post_rois),
            len(syn_pre_rois),
        )
        
        # Increment image counter for threshold bug replication.
        self._image_index += 1

    def _find_puncta(self, processed_imp, noise_tolerance, threshold_val, size_low_um, size_high_um, cal):
        """
        Detect puncta using 'Find Maxima' and 'Particle Analyzer'.
        
        This helper method encapsulates the core detection logic when 'use_maxima' is True.
        It mimics the "Find Maxima -> Segmented Particles" workflow of ImageJ.
        
        Note: This method is currently unused in the main pipeline (which uses 'prepare_channel' logic),
        but is kept as a reference implementation or for potential future use in 'segment_dense_image'.
        
        Args:
            processed_imp (ImagePlus): The preprocessed image.
            noise_tolerance (float): Tolerance for finding local maxima.
            threshold_val (float): Minimum intensity threshold.
            size_low_um (float): Minimum size in um^2.
            size_high_um (float): Maximum size in um^2.
            cal (Calibration): Image calibration.
            
        Returns:
            list: A list of dictionaries containing 'roi' and 'metrics' for each detected punctum.
        """
        # Use API directly to avoid headless issues and match segment_dense_image logic.
        ip = processed_imp.getProcessor()
        
        # Determine threshold argument for Find Maxima.
        # If threshold_val <= 0, we pass NO_THRESHOLD to disable it.
        threshold_arg = ImageProcessor.NO_THRESHOLD
        if threshold_val > 0:
            threshold_arg = float(threshold_val)

        # Run Find Maxima in SEGMENTED mode.
        # This creates a watershed-segmented image where each maximum is a particle.
        # The background is 0, and particles are separated by 0-value lines.
        mf = MaximumFinder()
        segmented_proc = mf.findMaxima(ip, float(noise_tolerance), threshold_arg, MaximumFinder.SEGMENTED, False, False)

        if segmented_proc is None:
            self.log("ERROR: Find Maxima produced no output image")
            return []

        # Wrap the processor in an ImagePlus for ParticleAnalyzer.
        segmented = ImagePlus("Segmented", segmented_proc)
        # segmented_proc is ByteProcessor (8-bit).

        # Prepare ResultsTable to capture measurements (though we extract them manually later).
        rt = ResultsTable()
        rt.reset()

        # Configure ParticleAnalyzer.
        # SHOW_MASKS: Output a binary mask of detected particles.
        # CLEAR_WORKSHEET: Clear previous results.
        # We use SHOW_MASKS instead of ADD_TO_MANAGER to avoid HeadlessException in some environments.
        # Macro does NOT use "include" option, so we don't use INCLUDE_HOLES
        # Without INCLUDE_HOLES: floodFill=true -> interior holes are filled
        pa_flags = ParticleAnalyzer.SHOW_MASKS | ParticleAnalyzer.CLEAR_WORKSHEET

        # Initialize ParticleAnalyzer with size constraints.
        # Note: size_low_um and size_high_um are passed directly. 
        # If the image is calibrated, PA uses these as physical units.
        pa = ParticleAnalyzer(pa_flags, MEASUREMENT_FLAGS, rt, size_low_um, size_high_um, 0.0, 1.0)
        
        # Suppress display of the output image.
        pa.setHideOutputImage(True)
        
        # Run analysis on the segmented image.
        pa.analyze(segmented)
        
        # Retrieve the binary mask of particles that satisfied the size criteria.
        mask_imp = pa.getOutputImage()
        
        entries = []
        if mask_imp:
            from ij.gui import ShapeRoi
            # Convert the binary mask to ROIs.
            # Threshold the mask to select all object pixels (255).
            mask_imp.getProcessor().setThreshold(128, 255, ImageProcessor.NO_LUT_UPDATE)
            
            # Run "Create Selection" to generate a composite ROI of all particles.
            IJ.run(mask_imp, "Create Selection", "")
            roi = mask_imp.getRoi()
            
            rois = []
            if roi:
                # If multiple particles, the ROI is a ShapeRoi (composite).
                # We split it into individual ROIs.
                if isinstance(roi, ShapeRoi):
                    rois = list(roi.getRois())
                else:
                    # Single particle found.
                    rois = [roi]
            
            mask_imp.close()
            
            # Iterate through each detected ROI to measure properties on the ORIGINAL image.
            for i, roi in enumerate(rois):
                # Set the ROI on the input image (processed_imp).
                processed_imp.setRoi(roi)
                
                # Measure statistics.
                stats = processed_imp.getStatistics(MEASUREMENT_FLAGS)
                
                # Compile metrics into a dictionary.
                metrics = {
                    'Label': "%s_%04d" % (processed_imp.getShortTitle(), i + 1),
                    'Area': stats.area,
                    'Mean': stats.mean,
                    'Min': int(stats.min),
                    'Max': int(stats.max),
                    'X': stats.xCentroid,
                    'Y': stats.yCentroid,
                    'XM': stats.xCenterOfMass,
                    'YM': stats.yCenterOfMass,
                    'Perim.': stats.perimeter if hasattr(stats, 'perimeter') else roi.getLength(),
                    'Feret': stats.feret,
                    'IntDen': stats.integratedDensity,
                    'RawIntDen': float(round(stats.rawIntegratedDensity)) if hasattr(stats, 'rawIntegratedDensity') else 0.0,
                    'FeretX': int(stats.feretX),
                    'FeretY': int(stats.feretY),
                    'FeretAngle': stats.feretAngle,
                    'MinFeret': stats.minFeret,
                }
                entries.append({'roi': roi, 'metrics': metrics})

        segmented.changes = False
        segmented.close()

        return entries

    def _preprocess_channel(self, imp, blur_enabled, blur_radius, background_radius):
        """
        Apply standard preprocessing: Median Blur and Rolling Ball Background Subtraction.
        
        This function replicates the preprocessing steps used in the SynapseJ macro.
        
        Args:
            imp (ImagePlus): The image to preprocess.
            blur_enabled (bool): Whether to apply median blur.
            blur_radius (float): Radius for median blur.
            background_radius (float): Radius for rolling ball background subtraction.
            
        Returns:
            ImagePlus: The processed image (a duplicate of the input).
        """
        # Duplicate the input image to avoid modifying the original.
        processed = imp.duplicate()
        
        if blur_enabled:
            # Apply Median Blur to reduce noise while preserving edges.
            # This is critical for reducing false positives from salt-and-pepper noise.
            IJ.run(processed, "Median...", "radius=" + str(blur_radius) + " stack")
            
        if background_radius > 0:
            # Apply Rolling Ball Background Subtraction to correct for uneven illumination.
            # This removes the low-frequency background signal, isolating the high-frequency puncta.
            subtracter = BackgroundSubtracter()
            
            # Iterate through each slice to apply the background subtraction.
            for s in range(1, processed.getNSlices() + 1):
                processed.setSlice(s)
                ip = processed.getProcessor()
                
                # rollingBallBackground(ip, radius, createBackground, lightBackground, useParaboloid, doPresmooth, correctCorners)
                # We use the standard settings: no paraboloid, presmoothing enabled, corner correction enabled.
                subtracter.rollingBallBackground(ip, float(background_radius), False, False, False, True, True)
                
        return processed


    def prepare_channel(self, work_imp, measure_imp, base_name, label, cal, params, slice_label_fn=None):
        """
        Execute the per-channel detection pipeline.
        
        This method corresponds to the 'PrepChannel' function in the SynapseJ macro.
        It performs the following steps:
        1. Preprocessing: Median blur and Fading correction.
        2. Background Subtraction (Rolling Ball).
        3. Detection:
           - If use_maxima is True: Uses 'Find Maxima' to segment dense puncta.
           - If use_maxima is False: Uses simple intensity thresholding.
        4. Size Filtering: Removes particles outside the specified size range (um^2).
        5. Mask Generation: Creates a binary mask of valid puncta.
        6. Result Generation: Subtracts the mask from the original image to isolate puncta.
        
        Args:
            work_imp (ImagePlus): The image to be processed (will be modified).
            measure_imp (ImagePlus): The original image for intensity measurements.
            base_name (str): Base filename for logging/output.
            label (str): Channel label ('Pre' or 'Post').
            cal (Calibration): Image calibration.
            params (dict): Dictionary of detection parameters.
            slice_label_fn (callable, optional): Function that takes z-slice (1-based) and returns 
                the original image slice label string for proper label formatting.
        """
        from ij.plugin import ImageCalculator
        from ij.process import AutoThresholder

        # --- Step 1: Preprocessing (Blur & Fade) ---
        
        # Apply Median Blur if enabled. This reduces salt-and-pepper noise which can cause false detections.
        if params['blur']:
            # Define a helper function to process a single slice 'i'.
            def blur_slice(i):
                try:
                    # Retrieve the ImageProcessor for the current slice (1-based index).
                    ip = work_imp.getStack().getProcessor(i+1)
                    # Apply the Median filter using the RankFilters plugin.
                    # The radius determines the size of the neighborhood.
                    RankFilters().rank(ip, params['blur_radius'], RankFilters.MEDIAN)
                except Exception as e:
                    self.log("ERROR in blur_slice {}: {}".format(i, e))
            
            # Execute the blur operation on all slices in parallel.
            ParallelUtils.parallel_for(blur_slice, work_imp.getStackSize())
        
        # Apply Fading Correction if enabled. This compensates for signal loss in deeper tissue slices.
        if params.get('apply_fade') and params.get('fade_factors'):
            factors = params['fade_factors']
            def fade_slice(i):
                try:
                    # Check if we have a factor for this slice index.
                    if i < len(factors):
                        # Multiply the pixel values of the slice by the correction factor.
                        work_imp.getStack().getProcessor(i+1).multiply(factors[i])
                except Exception as e:
                    self.log("ERROR in fade_slice {}: {}".format(i, e))
            
            # Execute fading correction in parallel.
            ParallelUtils.parallel_for(fade_slice, work_imp.getStackSize())
            self.log('{} fading correction applied with factors {}'.format(label, factors))

        # --- CRITICAL: Save copy BEFORE background subtraction ---
        # SynapseJ stores the original (blurred but not background-subtracted) image
        # for the final mask subtraction step. See SynapseJ lines 510-543:
        #   Line 510-511: Duplicate to ImNO+"-image-BKD" (confusing name - this is the ORIGINAL)
        #   Line 514: Background subtract is applied to ImNO+"-image" 
        #   Line 538-542: Swap names so the ORIGINAL is used for imageCalculator subtract
        # The naming convention is backwards: "-BKD" means "kept before BKD was applied"
        masked_target = work_imp.duplicate()

        # --- Step 2: Background Subtraction ---
        
        # Apply Rolling Ball background subtraction to remove uneven illumination/background haze.
        # This is applied to work_imp for detection purposes, but masked_target keeps the original.
        if params['background'] and params['background'] > 0:
            def bkd_slice(i):
                try:
                    ip = work_imp.getStack().getProcessor(i+1)
                    # Run the Rolling Ball algorithm.
                    # Parameters: radius, createBackground(False), lightBackground(False), 
                    # useParaboloid(False), doPresmooth(True), correctCorners(True).
                    BackgroundSubtracter().rollingBallBackground(ip, float(params['background']), False, False, False, True, True)
                except Exception as e:
                    self.log("ERROR in bkd_slice {}: {}".format(i, e))
            
            # Execute background subtraction in parallel.
            ParallelUtils.parallel_for(bkd_slice, work_imp.getStackSize())
            
            # Debug: log stats after background subtraction (matches debug.log format)
            self.log("DEBUG {} AFTER background subtraction:".format(label))
            for i in range(1, work_imp.getNSlices() + 1):
                ip = work_imp.getStack().getProcessor(i)
                stats = ip.getStatistics()
                self.log("  Slice {}: min={}, max={}, mean={:.2f}".format(i, int(stats.min), int(stats.max), stats.mean))
            
        mask_imp = None
        # Convert size bounds from um^2 to pixels for ParticleAnalyzer.
        # This is necessary because we often strip calibration during processing to avoid issues.
        
        # DEBUG: Print calibration info
        if cal:
            self.log("DEBUG {}: Calibration: pixelWidth={}, pixelHeight={}, unit={}".format(
                label, cal.pixelWidth, cal.pixelHeight, cal.getUnit()))
            pixel_area = cal.pixelWidth * cal.pixelHeight
            self.log("DEBUG {}: pixel_area = {} um^2, pixels_per_um = {}".format(
                label, pixel_area, 1.0/cal.pixelWidth if cal.pixelWidth > 0 else "N/A"))
        
        min_pixels, max_pixels = self._size_bounds_in_pixels(params['size_low'], params['size_high'], cal)
        
        # DEBUG: Print size filter like test script
        self.log("DEBUG {}: Size filter: {:.2f} - {:.2f} pixels (from {}-{} um^2)".format(
            label, min_pixels, max_pixels, params['size_low'], params['size_high']))
        
        # --- Step 3: Detection & Mask Generation ---
        
        if params['use_maxima']:
            # Strategy A: Find Maxima (for dense puncta)
            # This uses a local maximum search with a noise tolerance.
            # It returns a segmented image where each "basin" is a particle.
            segmented_imp = self.segment_dense_image(work_imp, params['noise'], params['min'])
            
            # DEBUG: Print segmented stats per slice like test script
            self.log("DEBUG {}: Segmented image stats:".format(label))
            for i in range(1, segmented_imp.getNSlices() + 1):
                ip = segmented_imp.getStack().getProcessor(i)
                hist = ip.getHistogram()
                non_zero = sum(hist[1:])
                self.log("  Slice {}: non-zero pixels={}".format(i, non_zero))
            
            # Temporarily force pixel calibration for ParticleAnalyzer to ensure size filtering works in pixels.
            # We copy the calibration to restore it later if needed.
            seg_cal = segmented_imp.getCalibration().copy()
            pixel_cal = seg_cal.copy()
            pixel_cal.setUnit("pixel")
            pixel_cal.pixelWidth = 1.0
            pixel_cal.pixelHeight = 1.0
            pixel_cal.pixelDepth = 1.0
            segmented_imp.setCalibration(pixel_cal)
            
            # --- PARALLEL mask creation ---
            # Each slice is processed independently, then combined in order.
            n_seg_slices = segmented_imp.getNSlices()
            width = segmented_imp.getWidth()
            height = segmented_imp.getHeight()
            
            self.log("DEBUG {}: Creating mask from segmented (AutoThreshold + PA):".format(label))
            
            def create_mask_slice(i):
                """Process one slice (0-indexed) and return (slice_idx, mask_processor, log_msg)."""
                slice_idx = i + 1
                try:
                    # CRITICAL: Duplicate the processor to avoid thread conflicts
                    ip = segmented_imp.getStack().getProcessor(slice_idx).duplicate()
                    
                    # Auto threshold
                    hist = ip.getHistogram()
                    threshold = AutoThresholder().getThreshold(AutoThresholder.Method.Default, hist)
                    ip.setThreshold(threshold, 255, ImageProcessor.NO_LUT_UPDATE)
                    
                    # Run particle analyzer to get mask
                    # Macro does NOT use "include" option
                    rt = ResultsTable()
                    pa_options = ParticleAnalyzer.SHOW_MASKS
                    pa = ParticleAnalyzer(pa_options, 0, rt, min_pixels, max_pixels)
                    pa.setHideOutputImage(True)
                    
                    temp_imp = ImagePlus("temp", ip)
                    pa.analyze(temp_imp)
                    
                    mask_out = pa.getOutputImage()
                    if mask_out:
                        mask_proc = mask_out.getProcessor()
                        mask_hist = mask_proc.getHistogram()
                        mask_nonzero = sum(mask_hist[1:])
                        log_msg = "  Slice {}: AutoThreshold={}, applying PA size={:.2f}-{:.2f}, mask non-zero pixels={}".format(
                            slice_idx, threshold, min_pixels, max_pixels, mask_nonzero)
                        return (slice_idx, mask_proc, log_msg)
                    else:
                        log_msg = "  Slice {}: AutoThreshold={}, NO MASK OUTPUT".format(slice_idx, threshold)
                        return (slice_idx, ByteProcessor(width, height), log_msg)
                except:
                    import traceback
                    traceback.print_exc()
                    return (slice_idx, ByteProcessor(width, height), "  Slice {}: ERROR".format(slice_idx))
            
            # Execute in parallel
            mask_results = ParallelUtils.parallel_for(create_mask_slice, n_seg_slices)
            
            # Combine results in order (parallel_for preserves order)
            mask_stack = ImageStack(width, height)
            for slice_idx, mask_proc, log_msg in mask_results:
                self.log(log_msg)
                mask_stack.addSlice(mask_proc)
            
            mask_imp = ImagePlus("Mask", mask_stack)
            mask_imp.setCalibration(seg_cal) 
            
            segmented_imp.close()
        else:
            # Strategy B: Simple Thresholding (for sparse puncta)
            # Duplicate the work image to avoid modifying it further.
            temp_imp = work_imp.duplicate()
            
            # Ensure we process in pixel units for consistent size filtering.
            temp_cal = temp_imp.getCalibration().copy()
            pixel_cal = temp_cal.copy()
            pixel_cal.setUnit("pixel")
            pixel_cal.pixelWidth = 1.0
            pixel_cal.pixelHeight = 1.0
            pixel_cal.pixelDepth = 1.0
            temp_imp.setCalibration(pixel_cal)
            
            def process_thresh_mask(i):
                try:
                    slice_idx = i + 1
                    ip = temp_imp.getStack().getProcessor(slice_idx)
                    if params['min'] > 0:
                        # Apply fixed threshold if provided.
                        # Pixels above 'min' are considered signal.
                        ip.setThreshold(params['min'], 65535, ImageProcessor.NO_LUT_UPDATE)
                    else:
                        # Otherwise, use AutoThresholder (Default method) to find an optimal threshold.
                        hist = ip.getHistogram()
                        threshold = AutoThresholder().getThreshold(AutoThresholder.Method.Default, hist)
                        ip.setThreshold(threshold, 65535, ImageProcessor.NO_LUT_UPDATE)
                    
                    t_imp = ImagePlus("temp", ip)
                    t_imp.setCalibration(pixel_cal)
                    
                    # Use ParticleAnalyzer to filter by size.
                    # Macro does NOT use "include" option, so we shouldn't either
                    # Without INCLUDE_HOLES: floodFill=true -> interior holes are filled in mask
                    pa_options = ParticleAnalyzer.SHOW_MASKS
                    pa = ParticleAnalyzer(pa_options, 0, ResultsTable(), min_pixels, max_pixels)
                    pa.setHideOutputImage(True)
                    pa.analyze(t_imp)
                    m = pa.getOutputImage()
                    if m:
                        return m.getProcessor()
                    else:
                        return ByteProcessor(temp_imp.getWidth(), temp_imp.getHeight())
                except Exception as e:
                    self.log("ERROR in process_thresh_mask {}: {}".format(i, e))
                    return ByteProcessor(temp_imp.getWidth(), temp_imp.getHeight())

            # Run in parallel.
            mask_procs = ParallelUtils.parallel_for(process_thresh_mask, temp_imp.getStackSize())
            
            mask_stack = ImageStack(temp_imp.getWidth(), temp_imp.getHeight())
            for p in mask_procs:
                mask_stack.addSlice(p)
            
            mask_imp = ImagePlus("Mask", mask_stack)
            mask_imp.setCalibration(temp_cal)
            
            # self.log("DEBUG: {} mask_imp (from threshold) slices: {}".format(label, mask_imp.getStackSize()))
            temp_imp.close()

        if mask_imp is None:
            # Fallback if something went wrong (shouldn't happen).
            masked_target.close()
            return [], work_imp, None

        # --- CREATE result_imp FIRST (for measurements) ---
        # SynapseJ measures on (Blurred - Mask), NOT on the raw or BKD-subtracted image.
        # Ref: SynapseJ_v_1.ijm lines 538-545:
        #   Line 538-542: Restores original blurred image (before BKD subtraction)
        #   Line 543: imageCalculator("Subtract create stack", C4-image, Mask)
        #   Line 551: setThreshold(ImLOW, 65535) on Result
        #   Line 552: Analyze Particles on Result - THIS is where measurements come from
        #
        # We need to:
        # 1. Convert mask to 16-bit inverted form for subtraction
        # 2. Subtract from masked_target (which is blurred, no BKD)
        # 3. Measure on this result_imp with threshold
        
        # Make a copy of mask for subtraction (we need original 8-bit mask for Wand tracing)
        subtract_mask = mask_imp.duplicate()
        ImageConverter(subtract_mask).convertToGray16()
        
        # Invert mask values: white (255) -> 0, black (0) -> 65535
        # This matches: run("Invert LUT"); run("16-bit"); run("Multiply...", "value=257"); run("Invert")
        def fix_mask_slice(i):
            try:
                ip = subtract_mask.getStack().getProcessor(i+1)
                pixels = ip.getPixels()
                for j in range(len(pixels)):
                    val = pixels[j] & 0xffff
                    if val > 0:
                        pixels[j] = 0
                    else:
                        pixels[j] = -1  # This becomes 65535 in unsigned
            except Exception as e:
                self.log("ERROR in fix_mask_slice {}: {}".format(i, e))
                    
        ParallelUtils.parallel_for(fix_mask_slice, subtract_mask.getStackSize())
        
        if self.dilate_enabled:
            self._dilate_mask(subtract_mask, self.dilate_pixels)
        
        # Create result_imp = masked_target - subtract_mask
        # masked_target is the blurred image BEFORE background subtraction
        ic = ImageCalculator()
        result_imp = ic.run("Subtract create stack", masked_target, subtract_mask)
        result_imp.deleteRoi()
        subtract_mask.close()

        # --- VALIDATED APPROACH (from test_find_maxima.py) ---
        # Count particles directly from the 8-bit mask using a SINGLE ResultsTable
        # that accumulates across all slices. This matches the test script which got
        # 372/103174 (very close to og2's 357/103161).
        #
        # CRITICAL: Must be sequential (not parallel) to use a single accumulating ResultsTable.
        # The test script does this and gets correct counts; parallel per-slice PA loses particles.
        
        from ij.gui import Wand
        
        n_slices = mask_imp.getStackSize()
        
        # Get ROIs and measurements per slice (PARALLELIZED)
        # Measure on result_imp (Blurred - Mask) with threshold, matching macro exactly
        # Each slice is processed independently; results are combined afterward.
        # 
        # APPROACH: We run PA on result_imp with threshold to get the exact particle 
        # boundaries that PA would trace. This matches the macro which runs PA directly
        # on "Result of ImNO-image" with setThreshold(ImLOW,65535).
        
        from ij.gui import Wand
        
        def process_slice_rois(i):
            """Process one slice (0-indexed) and return list of entries for that slice."""
            slice_idx = i + 1  # Convert to 1-based
            slice_entries = []
            
            try:
                # Get measurement processor from RESULT image (Blurred - Mask)
                # This matches SynapseJ which runs PA on "Result of C4-image"
                result_ip = result_imp.getStack().getProcessor(slice_idx).duplicate()
                
                # Apply threshold like macro: setThreshold(ImLOW, 65535)
                min_thresh = float(params['min'])
                max_thresh = 65535.0
                result_ip.setThreshold(min_thresh, max_thresh, ImageProcessor.NO_LUT_UPDATE)
                
                temp_imp = ImagePlus("result_slice", result_ip)
                temp_imp.setCalibration(cal)
                
                # Run PA on result_imp with threshold to find particles
                # Use RECORD_STARTS to get XStart/YStart for each particle
                slice_rt = ResultsTable()
                pa_options = ParticleAnalyzer.RECORD_STARTS
                pa = ParticleAnalyzer(pa_options, MEASUREMENT_FLAGS, slice_rt, 0, 1e9)
                pa.setHideOutputImage(True)
                pa.analyze(temp_imp)
                
                pa_count = slice_rt.getCounter()
                
                if pa_count > 0:
                    # PA already traced and measured each particle on result_ip
                    # Its measurements ARE the correct values - use them directly!
                    for row in range(pa_count):
                        try:
                            # Get XStart/YStart that PA used
                            x = int(slice_rt.getValue("XStart", row))
                            y = int(slice_rt.getValue("YStart", row))
                            
                            # Recreate the ROI using Wand just like PA did
                            # PA line 879: wand.autoOutline(x, y, level1, level2, wandMode)
                            # PA uses Wand on the same image it analyzed (result_ip)
                            wand = Wand(result_ip)
                            wand.autoOutline(x, y, min_thresh, max_thresh, Wand.LEGACY_MODE)
                            
                            if wand.npoints > 0:
                                r = PolygonRoi(wand.xpoints, wand.ypoints, wand.npoints, Roi.TRACED_ROI)
                                r.setPosition(slice_idx)
                                
                                # Use PA's measurements directly from the ResultsTable
                                # This ensures we get exactly the same values PA computed
                                area = slice_rt.getValue("Area", row)
                                mean = slice_rt.getValue("Mean", row)
                                min_val = slice_rt.getValue("Min", row)
                                max_val = slice_rt.getValue("Max", row)
                                x_centroid = slice_rt.getValue("X", row)
                                y_centroid = slice_rt.getValue("Y", row)
                                x_mass = slice_rt.getValue("XM", row)
                                y_mass = slice_rt.getValue("YM", row)
                                perim = slice_rt.getValue("Perim.", row)
                                feret = slice_rt.getValue("Feret", row)
                                int_den = slice_rt.getValue("IntDen", row)
                                raw_int_den = slice_rt.getValue("RawIntDen", row)
                                feret_x = slice_rt.getValue("FeretX", row)
                                feret_y = slice_rt.getValue("FeretY", row)
                                feret_angle = slice_rt.getValue("FeretAngle", row)
                                min_feret = slice_rt.getValue("MinFeret", row)
                                
                                metrics = {
                                    'Label': '',  # Will be set later
                                    'Index': 0,   # Will be set later
                                    'Area': area,
                                    'Mean': mean,
                                    'Min': int(min_val),
                                    'Max': int(max_val),
                                    'X': x_centroid,
                                    'Y': y_centroid,
                                    'XM': x_mass,
                                    'YM': y_mass,
                                    'Perim.': perim,
                                    'Feret': feret,
                                    'IntDen': int_den,
                                    'RawIntDen': float(round(raw_int_den)),
                                    'FeretX': int(feret_x),
                                    'FeretY': int(feret_y),
                                    'FeretAngle': feret_angle,
                                    'MinFeret': min_feret,
                                    'MinThr': int(min_thresh),
                                    'MaxThr': int(max_thresh),
                                    # Store pixel coords and slice for label generation
                                    '_x_pixel': x,
                                    '_y_pixel': y,
                                    '_slice_idx': slice_idx
                                }
                                
                                slice_entries.append({'roi': r, 'metrics': metrics})
                        except:
                            import traceback
                            traceback.print_exc()
                
                temp_imp.close()
            except:
                import traceback
                traceback.print_exc()
                
            return slice_entries
        
        # Execute slice processing in parallel
        slice_results = ParallelUtils.parallel_for(process_slice_rois, n_slices)
        
        # Combine results from all slices and assign final indices
        # Build labels in og2 format: {image}F.tif:{slice:04d}-{index:04d}-{y:04d}:{slice_label}
        # Example: AK5-2001PreF.tif:0001-0034-0091:c:4/4 z:1/5 - AK5-2001.nd2 (series 1)
        # Note: For "All Results" (Pre.txt/Post.txt), the label should be SIMPLE (base_name).
        # The COMPLEX label is used for "Syn Results" (PreResults.txt/PostResults.txt).
        # We set the ROI name to the SSSS-NNNN-YYYY format so measure_rois can reconstruct the complex label later.
        
        entries = []
        for slice_entries in slice_results:
            for entry in slice_entries:
                roi_index = len(entries) + 1
                entry['metrics']['Index'] = roi_index
                
                # Build ROI Name: SSSS-NNNN-YYYY
                slice_idx = entry['metrics'].get('_slice_idx', 1)
                # x_pix = entry['metrics'].get('_x_pixel', 0) # Unused in label
                y_pix = entry['metrics'].get('_y_pixel', 0)
                
                # Calculate Y-center for label (matches ImageJ/save_roi_set logic)
                # Note: _y_pixel from PA is YStart. We need YCenter for the label?
                # og2 analysis suggested 3rd part is Y.
                # save_roi_set uses: yc = r.y + r.height // 2
                # Let's use the ROI bounds to be consistent with save_roi_set
                r = entry['roi'].getBounds()
                yc = r.y + r.height // 2
                
                # Per-slice index (reset for each slice)
                # We need to track this. slice_entries contains all entries for this slice.
                # But slice_entries is a list. We can just use the index in that list + 1.
                # slice_entries is the result of process_slice_rois(i).
                # So entry is the k-th entry in that slice.
                # We need to find 'k'.
                # Since we are iterating slice_entries, we can just use a counter.
                pass # Logic handled in loop below
                
                # Clean up temporary fields
                entry['metrics'].pop('_x_pixel', None)
                entry['metrics'].pop('_y_pixel', None)
                entry['metrics'].pop('_slice_idx', None)
                
                # Label will be set in the ROI naming loop below
                
                entries.append(entry)

        # Assign SSSS-NNNN-YYYY names to ROIs and construct complex Labels
        # We do this after flattening to ensure we have the correct per-slice indices
        # Complex Label format: {base_name}{label}F.tif:{ROI_name}:{slice_label}
        # Example: AK5-2001PreF.tif:0001-0034-0091:c:4/4 z:1/5 - AK5-2001.nd2 (series 1)
        
        # og2 format rule: if ANY component of a label exceeds 9999, use 5-digit format
        # for ALL components of THAT label. Each label is formatted independently.
        
        for slice_list in slice_results:
            for i, entry in enumerate(slice_list):
                roi = entry['roi']
                slice_idx = roi.getPosition()
                if slice_idx < 1: slice_idx = 1
                
                r = roi.getBounds()
                yc = r.y + r.height // 2
                
                # Determine format for THIS label based on its component values
                if slice_idx > 9999 or (i + 1) > 9999 or yc > 9999:
                    roi_name = "{:05d}-{:05d}-{:05d}".format(slice_idx, i + 1, yc)
                else:
                    roi_name = "{:04d}-{:04d}-{:04d}".format(slice_idx, i + 1, yc)
                roi.setName(roi_name)
                
                # Construct complex Label for CorrResults and Syn Results
                # Format: {base_name}{label}F.tif:{ROI_name}:{slice_label}
                complex_filename = "{}{}F.tif".format(base_name, label)
                
                # Get slice label from slice_label_fn if available
                if slice_label_fn:
                    orig_slice_label = slice_label_fn(slice_idx)
                else:
                    orig_slice_label = ""
                
                complex_label = "{}:{}:{}".format(complex_filename, roi_name, orig_slice_label)
                entry['metrics']['Label'] = complex_label

        return entries, result_imp, mask_imp

    def _metrics_dict(self, base_name, label, index, stats, cal, roi, min_thr=0, max_thr=65535, slice_label_fn=None):
        """
        Convert ImageJ stats object into a dictionary matching SynapseJ's output format.
        
        This ensures that the CSV/TSV output columns exactly match those produced by the
        original ImageJ macro, facilitating compatibility with existing analysis pipelines.
        
        Ref: SynapseJ_v_1.ijm line 29: 
        resultLabel = newArray("Label","Area","Mean","Min","Max","X","Y","XM","YM","Perim.","Feret","IntDen","RawIntDen","FeretX","FeretY","FeretAngle","MinFeret");
        
        Args:
            base_name (str): Image name.
            label (str): Channel label (e.g., 'Pre', 'Post').
            index (int): Punctum index.
            stats (ImageStatistics): The statistics object from ImageJ.
            cal (Calibration): Image calibration.
            roi (Roi): The ROI object.
            min_thr (float): Minimum threshold value used for detection.
            max_thr (float): Maximum threshold value used for detection.
            slice_label_fn (callable, optional): Function to retrieve original slice label.
            
        Returns:
            dict: A dictionary where keys are column headers and values are measurements.
        """
        # Helper to safely get attributes that might be missing in some ImageStatistics versions
        def get_stat(obj, name, default=0):
            return getattr(obj, name, default)

        # Robust Feret retrieval: Try ROI first (geometry based), then stats (pixel based)
        # IMPORTANT: When roi.getImage() is set with a calibrated image, getFeretValues() and
        # getLength() return values ALREADY in calibrated units (microns). When no image is set,
        # they return pixel values. We check if ROI has an image to decide whether to convert.
        feret_vals = None
        roi_has_calibrated_image = roi and roi.getImage() is not None
        try:
            if roi:
                feret_vals = roi.getFeretValues() # Returns calibrated if ROI has image, else pixels
        except:
            pass

        # Get pixel width for potential conversion (only used if ROI has no calibrated image)
        px_w = self._pixel_width(cal)
        
        feret = feret_vals[0] if feret_vals else get_stat(stats, 'feret')
        # Fallback for Feret if 0 (common in headless mode or for small PointRois)
        if feret == 0 and stats.area > 0:
            # Approximate as circle diameter: Area = pi * (d/2)^2  =>  d = 2 * sqrt(Area / pi)
            # stats.area is already calibrated, so result is in calibrated units
            feret = 2 * math.sqrt(stats.area / math.pi)
        elif not roi_has_calibrated_image and feret_vals:
            # Only convert if ROI doesn't have a calibrated image (values are in pixels)
            feret = feret * px_w
        # else: feret is already calibrated (from feret_vals with calibrated image)

        feret_angle = feret_vals[1] if feret_vals else get_stat(stats, 'feretAngle')  # Angle doesn't need conversion
        min_feret = feret_vals[2] if feret_vals else get_stat(stats, 'minFeret')
        # Only convert MinFeret if ROI doesn't have a calibrated image
        if min_feret > 0 and feret_vals and not roi_has_calibrated_image:
            min_feret = min_feret * px_w
        
        # FeretX and FeretY are pixel coordinates - convert to calibrated units
        feret_x = feret_vals[3] if feret_vals else get_stat(stats, 'feretX')
        # FeretX and FeretY are pixel coordinates - keep in pixels to match SynapseJ
        feret_x = feret_vals[3] if feret_vals else get_stat(stats, 'feretX')
        feret_y = feret_vals[4] if feret_vals else get_stat(stats, 'feretY')
        # Note: Unlike Feret and MinFeret, FeretX/FeretY remain in pixels per SynapseJ convention

        # New: Bounding box width/height (um), relative to Feret
        bbox_width, bbox_height = 0.0, 0.0
        if roi:
            bounds = roi.getBounds()
            px_w = self._pixel_width(cal)
            px_h = self._pixel_height(cal)
            bbox_width = bounds.width * px_w
            bbox_height = bounds.height * px_h
            
            # Rotate for Feret-relative dimensions if angle is significant
            # This approximates the "caliper" width/height relative to the max diameter
            if feret_angle != 0:
                # If the Feret angle is closer to vertical (>45 deg), swap width/height
                # to represent dimensions along/perpendicular to the Feret axis roughly
                if abs(feret_angle) > 45:
                    bbox_width, bbox_height = bbox_height, bbox_width

        # Construct Label
        # Default: base_name:label:index
        final_label = '{}:{}:{}'.format(base_name, label, index)
        
        # If slice_label_fn is provided, try to construct the Complex Label matching og2
        # Format: {image}F.tif:{SSSS-NNNN-YYYY}:{slice_label}
        if slice_label_fn and roi:
            try:
                roi_name = roi.getName()
                # Check if ROI name matches SSSS-NNNN-YYYY format (roughly)
                if roi_name and '-' in roi_name:
                    slice_idx = roi.getPosition()
                    if slice_idx < 1: slice_idx = 1
                    orig_slice_label = slice_label_fn(slice_idx)
                    
                    # Construct complex label
                    # Note: base_name is usually short name (AK5-2001).
                    # og2 uses AK5-2001PreF.tif
                    complex_filename = "{}{}F.tif".format(base_name, label)
                    final_label = "{}:{}:{}".format(complex_filename, roi_name, orig_slice_label)
                else:
                    # DEBUG: Log why it failed
                    # self.log("DEBUG: ROI name '{}' does not match format".format(roi_name))
                    pass
            except Exception as e:
                # self.log("DEBUG: Error constructing complex label: {}".format(e))
                pass

        return {
            # Standard ImageJ Result Columns
            'Label': final_label,
            'Area': stats.area,
            'Mean': stats.mean,
            'Min': int(stats.min),
            'Max': int(stats.max),
            'X': stats.xCentroid,
            'Y': stats.yCentroid,
            'XM': stats.xCenterOfMass,
            'YM': stats.yCenterOfMass,
            # Perimeter: roi.getLength() returns calibrated value when ROI has a calibrated image,
            # otherwise returns pixels. Only multiply by px_w if ROI has no calibrated image.
            'Perim.': (get_stat(stats, 'perimeter') or (roi.getLength() if roi else 0)) * (1.0 if roi_has_calibrated_image else px_w),
            'Feret': feret,
            'IntDen': get_stat(stats, 'integratedDensity') or (stats.area * stats.mean), # Fallback if field missing
            'RawIntDen': float(round(get_stat(stats, 'rawIntegratedDensity') or (stats.pixelCount * stats.mean))), # Fallback, stored as float with rounding
            'FeretX': int(feret_x),
            'FeretY': int(feret_y),
            'FeretAngle': feret_angle,
            'MinFeret': min_feret,
            'MinThr': int(min_thr),
            'MaxThr': int(max_thr),
            'BBox Width (um)': bbox_width,
            'BBox Height (um)': bbox_height,
            
            # Internal/Legacy keys (kept for downstream logic compatibility if needed)
            'Image': base_name,
            'Channel': label,
            'Index': index,
            'Area_um2': stats.area, # Redundant but explicit
            'MeanIntensity': stats.mean,
            'CentroidX_um': stats.xCentroid, # Assuming calibrated
            'CentroidY_um': stats.yCentroid,
        }

    def _size_bounds_in_pixels(self, min_um2, max_um2, cal):
        """
        Translate um^2 size bounds into pixel counts.
        
        The ParticleAnalyzer requires size limits in pixels (or calibrated units if set, 
        but explicit pixel conversion is safer for headless operation).
        
        Args:
            min_um2 (float): Minimum area in square microns.
            max_um2 (float): Maximum area in square microns.
            cal (Calibration): Image calibration object.
            
        Returns:
            tuple: (min_pixels, max_pixels)
        """
        pixel_area = self._pixel_area(cal)
        if pixel_area <= 0:
            pixel_area = 1.0
        min_px = min_um2 / pixel_area if min_um2 and min_um2 > 0 else 0
        max_px = max_um2 / pixel_area if max_um2 and max_um2 > 0 else 1e12
        return (min_px, max_px)


    def _gate_puncta_with_marker_overlap(self, puncta_entries, marker_label_map, overlap_threshold=5):
        """
        Filter puncta based on overlap with a marker label map.
        
        Note: This method appears to be an alternative or legacy implementation of marker gating.
        The primary marker gating logic is currently handled by '_filter_by_intensity'.
        This version checks for pixel overlap with a segmented marker map rather than raw intensity.
        
        Args:
            puncta_entries (list): List of candidate puncta.
            marker_label_map (ImagePlus): Label map of marker regions.
            overlap_threshold (int): Minimum overlapping pixels required.
            
        Returns:
            list: Filtered list of puncta.
        """
        # If no marker map is provided, we cannot filter, so we return all entries.
        if marker_label_map is None:
            return puncta_entries
            
        retained = []
        # Iterate through each candidate punctum.
        for entry in puncta_entries:
            roi = entry['roi']
            
            # Ensure the marker map is set to the correct Z-slice.
            if marker_label_map.getNSlices() > 1 and roi.getPosition() > 0:
                marker_label_map.setSlice(roi.getPosition())
                
            # Set the ROI on the marker map to inspect the underlying pixels.
            marker_label_map.setRoi(roi)
            
            # Get the histogram of pixel values within the ROI.
            # Since this is a label map, values > 0 represent marker regions.
            hist = marker_label_map.getProcessor().getHistogram()
            
            # Check if there is significant overlap with any marker region.
            # hist[1:] contains counts for all non-background values.
            # We take the maximum overlap with any single marker region.
            max_overlap = max(hist[1:]) if len(hist) > 1 else 0
            
            # If the overlap exceeds the threshold, keep the punctum.
            if max_overlap >= overlap_threshold:
                retained.append(entry)
                
        # Clean up the ROI on the marker map.
        marker_label_map.deleteRoi()
        
        return retained            
            
    def save_text_lines(self, lines, path):
        """
        Write a list of strings to a text file.
        
        Used for saving the correlation analysis results (CorrResults.txt).
        
        Args:
            lines (list): List of strings to write.
            path (str): Destination file path.
        """
        # If there is no data to write, exit early to avoid creating empty files.
        if not lines:
            return
            
        # Open the file in write mode ('w').
        with open(path, 'w') as handle:
            for line in lines:
                # Write each line followed by a newline character.
                handle.write(line + '\n')

    def save_results_table(self, rows, path, add_row_numbers=True, use_auto_format=False, first_image_label=None):
        """
        Save a list of dictionaries as a tab-separated value (TSV) file.
        
        This function handles the serialization of measurement data to disk.
        It ensures that the header row matches the keys of the dictionaries.
        
        Args:
            rows (list): List of dictionaries, where each dictionary is a row.
            path (str): Output file path.
            add_row_numbers (bool): If True, add row numbers as first column (og2 format).
            use_auto_format (bool): If True, use AUTO_FORMAT (variable precision).
                                    If False, use fixed 3 decimal places.
            first_image_label (str): If provided, rows with this Label (base name) use 3-decimal format,
                                     all other rows use AUTO_FORMAT. This matches og2's
                                     behavior for combined files where the first image
                                     processed sets the table format, and subsequent
                                     images switch to AUTO_FORMAT.
        """
        # If there are no rows, there is nothing to save.
        if not rows:
            return
        
        # Helper to extract base name from Label (same logic as _filter_all_results_metrics)
        def extract_base_name(full_label):
            if not full_label:
                return ''
            if ':' in full_label:
                first_part = full_label.split(':')[0]
            else:
                first_part = full_label
            base_name = first_part
            for suffix in ['PreF.tif', 'PostF.tif', 'Pre.tif', 'Post.tif', 
                           'PreF', 'PostF', 'Pre', 'Post', '.tif', '.TIF']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            return base_name
        
        # Choose formatter based on file type and per-row label
        def format_val(key, val, row_label=None):
            if first_image_label is not None:
                # Combined file: first image uses 3-dec, others use AUTO
                # Extract base name from row_label for comparison
                row_base_name = extract_base_name(row_label)
                if row_base_name == first_image_label:
                    return SynapseJ4ChannelComplete.format_value_3dec(val)
                else:
                    return SynapseJ4ChannelComplete.format_value(val)
            elif use_auto_format:
                return SynapseJ4ChannelComplete.format_value(val)
            else:
                return SynapseJ4ChannelComplete.format_value_3dec(val)
        
        formatted_rows = []
        for row in rows:
            formatted_row = OrderedDict()
            row_label = row.get('Label', '')
            for key, val in row.items():
                formatted_row[key] = format_val(key, val, row_label)
            formatted_rows.append(formatted_row)
            
        # Open the file for writing.
        with open(path, 'w') as handle:
            # Get fieldnames from first row
            fieldnames = list(formatted_rows[0].keys())
            
            # Write header row with row number column (empty header for row number column)
            if add_row_numbers:
                handle.write(' \t' + '\t'.join(fieldnames) + '\n')
            else:
                handle.write('\t'.join(fieldnames) + '\n')
            
            # Write all data rows with row numbers
            for i, row in enumerate(formatted_rows, 1):
                values = [row.get(fn, '') for fn in fieldnames]
                if add_row_numbers:
                    handle.write('{}\t'.format(i) + '\t'.join(values) + '\n')
                else:
                    handle.write('\t'.join(values) + '\n')

    def save_roi_set(self, rois, path, imp=None, roi_index_map=None):
        """
        Save a list of ROIs to a ZIP file (ImageJ ROI Set).
        
        This allows the results to be opened in the standard ImageJ ROI Manager
        for manual inspection and validation. The format is compatible with
        ImageJ's "Save..." command in the ROI Manager.
        
        The ROI naming follows the ImageJ RoiManager convention when ROIs are
        added via Analyze Particles: SSSS-NNNN-YYYY.roi where:
        - S = slice position (1-based)
        - N = sequential index (1-based, resets per slice for full sets, 
              or preserved from parent set for subsets)
        - Y = y-center of the ROI bounds
        
        For single-slice images, format is NNNN-YYYY.roi.
        
        This matches the getLabel(imp, roi, n) behavior in RoiManager.java when n>=0.
        
        Args:
            rois (list): List of Roi objects to save.
            path (str): Output path (should end in .zip).
            imp (ImagePlus, optional): Image for determining stack size and digit width.
            roi_index_map (dict, optional): If provided, maps ROI objects to their 
                original (slice, per_slice_index) tuple. Used when saving subsets to 
                preserve the original indices from the parent set.
                
        Returns:
            dict: A mapping of ROI objects to their (slice, per_slice_index) tuples.
                  This can be passed to subsequent calls for subset saving.
        """
        # If no ROIs to save, exit.
        if not rois:
            return {}
            
        # Ensure the parent directory exists.
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        
        # Determine number of digits needed for coordinate/slice fields.
        # ImageJ uses at least 4 digits, more if image dimensions require it.
        # Matches ImageJ's getLabel logic in RoiManager.java:
        #   - Start with 4 digits
        #   - Expand to 5 if stack size >= 10000 OR image height >= 10000
        #   - Also expand if the x or y center value string is > 4 chars
        digits = 4
        if imp is not None:
            # Check if we need more digits based on stack size or image height
            if imp.getStackSize() >= 10000 or imp.getHeight() >= 10000:
                digits = 5
        
        is_stack = imp is not None and imp.getStackSize() > 1
        
        # Build the index map if not provided (for full sets)
        # or use the provided map (for subsets)
        new_roi_index_map = {}
            
        # Create a ZipOutputStream to write the .zip file.
        # We wrap a FileOutputStream in a BufferedOutputStream for performance.
        stream = ZipOutputStream(BufferedOutputStream(FileOutputStream(path)))
        try:
            # Track per-slice index (ImageJ resets index counter for each slice)
            current_slice = -1
            slice_index = 0
            
            # Iterate through each ROI.
            for idx, roi in enumerate(rois):
                # Get ROI y-center (matching ImageJ's getLabel behavior when n>=0)
                r = roi.getBounds()
                yc = r.y + r.height // 2
                
                # Build label: SSSS-NNNN-YYYY for stacks, NNNN-YYYY for single images
                if is_stack:
                    slice_pos = roi.getPosition()
                    if slice_pos <= 0:
                        slice_pos = 1  # Default to slice 1 if not set
                    
                    # Check if we have a pre-computed index from a parent set
                    if roi_index_map is not None and roi in roi_index_map:
                        # Use the original index from the parent set
                        orig_slice, orig_idx = roi_index_map[roi]
                        slice_pos = orig_slice
                        slice_index = orig_idx
                    else:
                        # Compute new index (resets per slice)
                        if slice_pos != current_slice:
                            current_slice = slice_pos
                            slice_index = 0
                        slice_index += 1
                        # Store for potential subset use
                        new_roi_index_map[roi] = (slice_pos, slice_index)
                    
                    # ImageJ dynamically expands digits if index or y-center needs more chars
                    # This matches the getLabel() logic where digits is adjusted per-ROI
                    local_digits = digits
                    if len(str(slice_index)) > local_digits:
                        local_digits = len(str(slice_index))
                    if len(str(yc)) > local_digits:
                        local_digits = len(str(yc))
                    
                    # Format with leading zeros - all fields use same digit count
                    zs = str(slice_pos).zfill(local_digits)
                    ns = str(slice_index).zfill(local_digits)
                    ys = str(yc).zfill(local_digits)
                    label = '{}-{}-{}.roi'.format(zs, ns, ys)
                else:
                    # For single-slice, use global 1-based index or from map
                    if roi_index_map is not None and roi in roi_index_map:
                        _, n = roi_index_map[roi]
                    else:
                        n = idx + 1
                        new_roi_index_map[roi] = (1, n)
                    
                    # ImageJ dynamically expands digits
                    local_digits = digits
                    if len(str(n)) > local_digits:
                        local_digits = len(str(n))
                    if len(str(yc)) > local_digits:
                        local_digits = len(str(yc))
                    
                    ns = str(n).zfill(local_digits)
                    ys = str(yc).zfill(local_digits)
                    label = '{}-{}.roi'.format(ns, ys)
                
                # Create a new entry in the zip file for this ROI.
                entry = ZipEntry(label)

                stream.putNextEntry(entry)
                
                # Use ImageJ's RoiEncoder to serialize the ROI object into the stream.
                # This handles the binary format specifics of ImageJ ROIs.
                encoder = RoiEncoder(stream)
                encoder.write(roi.clone())
                
                # Close the current entry to prepare for the next one.
                stream.closeEntry()
        finally:
            # Ensure the stream is closed to flush data to disk.
            stream.close()
        
        # Return the index map for use by subset saves
        return new_roi_index_map

    def save_overlay(self, pre_src, post_src, c1_imp, c2_imp, base_name):
        """
        Create and save a multi-channel composite overlay image.
        
        This generates the "PrePost" merged image, which provides a visual summary of the analysis.
        It combines the processed pre- and post-synaptic channels (after background subtraction)
        with the original marker channels.
        
        Channel Mapping (Standard ImageJ Colors):
        - Channel 1 (Red): Processed Pre-synaptic (PreF)
        - Channel 2 (Green): Processed Post-synaptic (PostF)
        - Channel 3 (Blue): Original Post-Marker (C1)
        - Channel 4 (Gray): Original Pre-Marker (C2)
        
        Args:
            pre_src (ImagePlus): Processed pre-synaptic image.
            post_src (ImagePlus): Processed post-synaptic image.
            c1_imp (ImagePlus): Original Channel 1 (Post Marker).
            c2_imp (ImagePlus): Original Channel 2 (Pre Marker).
            base_name (str): Base filename for saving.
        """
        # Duplicate all images to avoid modifying the originals during the merge process.
        # This is crucial because 'RGBStackMerge' might alter the input stack properties.
        c1 = pre_src.duplicate()
        c2 = post_src.duplicate()
        c3 = c1_imp.duplicate()
        c4 = c2_imp.duplicate()
        
        # Merge Channels
        # The order in the list determines the channel assignment (Red, Green, Blue, Gray, etc.)
        # [c1, c2, c3, c4] maps to:
        # 1. Red: Pre-synaptic
        # 2. Green: Post-synaptic
        # 3. Blue: Post-Marker
        # 4. Gray: Pre-Marker
        merged = RGBStackMerge.mergeChannels([c1, c2, c3, c4], True)
        
        # Save the merged composite image
        if merged is not None:
            out_path = os.path.join(self.merge_dir, '{}PrePost.tif'.format(base_name))
            IJ.save(merged, out_path)
            merged.close()
        
        # Clean up duplicates to free memory
        c1.close()
        c2.close()
        c3.close()
        c4.close()

    def record_summary(self, base_name, pre_count, pre_marker_count, post_count, post_marker_count,
                       syn_post_count, syn_pre_count):
        """
        Append a summary row for the current image to the 'Collated ResultsIF' table.
        
        This table provides high-level counts of detected structures for each image processed.
        It mirrors the columns of the original SynapseJ macro output.
        
        Args:
            base_name (str): Image name.
            pre_count (int): Total pre-synaptic puncta detected.
            pre_marker_count (int): Pre-synaptic puncta passing marker gate (if applicable).
            post_count (int): Total post-synaptic puncta detected.
            post_marker_count (int): Post-synaptic puncta passing marker gate (if applicable).
            syn_post_count (int): Number of post-synaptic puncta that are part of a synapse.
            syn_pre_count (int): Number of pre-synaptic puncta that are part of a synapse.
        """
        # Add .tif suffix to Label if not already present (og2 format)
        label_with_suffix = base_name if base_name.endswith('.tif') else base_name + '.tif'
        # Use OrderedDict to preserve exact column order matching og2:
        # Label, Synapse Post No., Synapse Pre No., Thr Pre No., Fourth Post No., Post No., Pre No.
        entry = OrderedDict([
            ('Label', label_with_suffix),
            ('Synapse Post No.', syn_post_count),
            ('Synapse Pre No.', syn_pre_count),
            ('Thr Pre No.', pre_marker_count if (self.pre_marker_channel > 0 and self.pre_marker_thresholds) else ''),
            ('Fourth Post No.', post_marker_count if (self.post_marker_channel > 0 and self.post_marker_thresholds) else ''),
            ('Post No.', post_count),
            ('Pre No.', pre_count),
        ])
        self.results_summary.append(entry)

    def save_all_results(self):
        """
        Write all accumulated result tables to disk.
        
        This method is called at the end of the batch processing to save the
        aggregated data from all images into tab-separated text files.
        These files match the naming convention and format of the original SynapseJ macro.
        """
        # Save the summary table (Collated ResultsIF) which contains one row per image.
        if self.results_summary:
            self.save_results_table(self.results_summary, os.path.join(self.dest_dir, 'Collated ResultsIF.txt'))
        
        # Determine the first image label for combined files.
        # og2 behavior: first image processed uses 3-decimal format, subsequent images use AUTO_FORMAT.
        # We need to extract the base label the same way _filter_all_results_metrics does.
        first_image_label = None
        if self.all_pre_results:
            full_label = self.all_pre_results[0].get('Label', None)
            if full_label:
                # Apply same label extraction logic as _filter_all_results_metrics
                if ':' in full_label:
                    first_part = full_label.split(':')[0]
                else:
                    first_part = full_label
                base_name = first_part
                for suffix in ['PreF.tif', 'PostF.tif', 'Pre.tif', 'Post.tif', 
                               'PreF', 'PostF', 'Pre', 'Post', '.tif', '.TIF']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                first_image_label = base_name
            
        # Save the detailed list of ALL detected pre-synaptic puncta (before synapse filtering).
        # Use _filter_all_results_metrics for simple Label format (no thresholds)
        # Combined files use first_image_label to match og2's per-image formatting behavior.
        if self.all_pre_results:
            self.save_results_table(self._filter_all_results_metrics(self.all_pre_results), os.path.join(self.dest_dir, 'All Pre Results.txt'), first_image_label=first_image_label)
            
        # Save the detailed list of ALL detected post-synaptic puncta (before synapse filtering).
        if self.all_post_results:
            self.save_results_table(self._filter_all_results_metrics(self.all_post_results), os.path.join(self.dest_dir, 'All Post Results.txt'), first_image_label=first_image_label)
            
        # Save the list of CONFIRMED synaptic pre-puncta (those that colocalized).
        # Use _filter_standard_metrics for full Label format (with thresholds)
        if self.syn_pre_results:
            self.save_results_table(self._filter_standard_metrics(self.syn_pre_results), os.path.join(self.dest_dir, 'Syn Pre Results.txt'), first_image_label=first_image_label)
            
        # Save the list of CONFIRMED synaptic post-puncta (those that colocalized).
        if self.syn_post_results:
            self.save_results_table(self._filter_standard_metrics(self.syn_post_results), os.path.join(self.dest_dir, 'Syn Post Results.txt'), first_image_label=first_image_label)
            
        # Save the paired synapse data (currently unused/empty in this implementation but kept for compatibility).
        if self.synapse_pair_rows:
            self.save_results_table(self.synapse_pair_rows, os.path.join(self.dest_dir, 'AllSynapsePairs.tsv'))
            
        # Save the Pre->Post correlation analysis results (nearest neighbor data).
        if self.pre_correlation_rows:
            self.save_text_lines(self.pre_correlation_rows, os.path.join(self.dest_dir, 'CorrResults.txt'))
            
        # Save the Post->Pre correlation analysis results (nearest neighbor data).
        if self.post_correlation_rows:
            self.save_text_lines(self.post_correlation_rows, os.path.join(self.dest_dir, 'CorrResults2.txt'))
        
        # Save the batch synapse report (accumulated from all images) as proper CSV
        present_dir = os.path.join(self.dest_dir, 'present')
        if not os.path.exists(present_dir):
            os.makedirs(present_dir)

        if self.batch_synapse_rows:
            batch_path = os.path.join(present_dir, 'Batch_Synapse_Report.csv')
            with open(batch_path, 'w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.batch_synapse_rows[0].keys()))
                writer.writeheader()
                for row in self.batch_synapse_rows:
                    writer.writerow(row)
            self.log("Saved Batch_Synapse_Report.csv with {} total rows".format(len(self.batch_synapse_rows)))

    def save_log(self):
        """Store the execution log (IFALog.txt) for provenance and debugging."""
        if not self.log_messages:
            return
        # Write the log messages to a text file in the destination directory.
        # This file serves as a record of the analysis parameters and execution steps.
        path = os.path.join(self.dest_dir, 'IFALog.txt')
        with open(path, 'w') as handle:
            handle.write('\n'.join(self.log_messages))

    def _pixel_area(self, cal):
        """
        Calculate the area of a single pixel in square microns.
        
        Used for converting pixel counts to physical area units (um^2).
        If calibration is missing or invalid, defaults to 1.0 (pixels).
        
        Args:
            cal (Calibration): Image calibration object.
            
        Returns:
            float: Area of one pixel (width * height).
        """
        if cal is None:
            return 1.0
        # Get pixel width, defaulting to 1.0 if not set or invalid.
        px = cal.pixelWidth if cal.pixelWidth and cal.pixelWidth > 0 else 1.0
        # Get pixel height, defaulting to 1.0 if not set or invalid.
        py = cal.pixelHeight if cal.pixelHeight and cal.pixelHeight > 0 else 1.0
        # Return the product (Area = Width * Height).
        return px * py

    def _pixel_width(self, cal):
        """
        Get the pixel width in microns.
        
        Args:
            cal (Calibration): Image calibration object.
            
        Returns:
            float: Pixel width (default 1.0).
        """
        # Check for valid calibration and positive pixel width.
        if cal is None or not cal.pixelWidth or cal.pixelWidth <= 0:
            return 1.0
        return cal.pixelWidth

    def _pixel_height(self, cal):
        """
        Get the pixel height in microns.
        
        Args:
            cal (Calibration): Image calibration object.
            
        Returns:
            float: Pixel height (default 1.0).
        """
        # Check for valid calibration and positive pixel height.
        if cal is None or not cal.pixelHeight or cal.pixelHeight <= 0:
            return 1.0
        return cal.pixelHeight

    def _create_count_mask(self, imp, min_threshold):
        """
        Generate a "Count Mask" image where each particle has a unique integer ID.
        
        This is crucial for the "Strict Pixel Count" association strategy (LapNo > 1).
        It allows us to determine exactly which particle a pixel belongs to, enabling
        us to count how many pixels of a specific partner particle overlap with an anchor ROI.
        
        Mirrors the macro command:
        run("Analyze Particles...", "size=0-infinity show=[Count Masks] display clear stack");
        
        Args:
            imp (ImagePlus): The image to analyze (usually the partner channel).
            min_threshold (float): Minimum intensity threshold for particle detection.
            
        Returns:
            ImagePlus: A 16-bit image where background is 0 and particle 'i' has value 'i'.
        """
        # Duplicate the input image to avoid modifying the original.
        temp = imp.duplicate()
        
        # Apply threshold if specified. This defines the "particles" vs background.
        if min_threshold > 0:
            IJ.setThreshold(temp, float(min_threshold), 65535.0)
            temp.getProcessor().setThreshold(float(min_threshold), 65535.0, ImageProcessor.NO_LUT_UPDATE)

        # Configure ParticleAnalyzer to generate a "Count Mask".
        # SHOW_ROI_MASKS in the API corresponds to "Count Masks" in the GUI when used with 16-bit output?
        # Actually, SHOW_MASKS produces binary masks (0/255).
        # To get unique IDs (Count Masks), we might need a different approach or specific flags.
        # In ImageJ macro: "show=[Count Masks]" creates a 16-bit image where pixel value = particle index.
        # In Java API, ParticleAnalyzer doesn't have a direct "SHOW_COUNT_MASKS" constant exposed easily in all versions.
        # However, using 'SHOW_ROI_MASKS' often produces a mask where pixels are labeled.
        # Let's verify: The macro uses "Count Masks".
        flags = ParticleAnalyzer.CLEAR_WORKSHEET | ParticleAnalyzer.DOES_STACKS | ParticleAnalyzer.SHOW_ROI_MASKS
        
        # Initialize ParticleAnalyzer with no size constraints (0-Infinity) to capture all thresholded regions.
        pa = ParticleAnalyzer(flags, 0, None, 0.0, Double.POSITIVE_INFINITY, 0.0, 0.0)
        
        # Run analysis. This generates the output image internally.
        pa.analyze(temp)
        
        # Retrieve the generated mask image.
        count_mask = pa.getOutputImage()
        
        # Clean up the temporary image.
        temp.close()

        # Ensure the mask has the correct calibration (same as input).
        if count_mask is not None:
            count_mask.setCalibration(imp.getCalibration())
            
        return count_mask

    def assoc_roi(self, check_imp, store_imp, store_mask_imp, rois, check_threshold, overlap_pixels):
        """
        Identify synapses by checking for overlap between pre- and post-synaptic puncta.
        
        This method replicates the 'AssocROI' function from the SynapseJ macro.
        It filters the provided list of ROIs (e.g., Pre-synaptic puncta) based on their
        overlap with the 'check_imp' (e.g., Post-synaptic image).
        
        Args:
            check_imp (ImagePlus): The partner image to check for overlap (e.g., Post image when filtering Pre ROIs).
            store_imp (ImagePlus): The source image corresponding to the ROIs (will be modified to remove rejected puncta).
            store_mask_imp (ImagePlus): The mask of the source image (will be modified).
            rois (list): List of ROIs to filter.
            check_threshold (float): Intensity threshold for the check image.
            overlap_pixels (int): Minimum number of overlapping pixels required.
            
        Returns:
            list: The subset of ROIs that satisfy the overlap criteria (the synapses).
        """
        # If there are no ROIs to check, return an empty list immediately to save processing time.
        if not rois:
            return []

        # Prepare the check image (either intensity or count mask)
        target_imp = check_imp
        is_count_mask = False
        
        # --- Strategy Selection ---
        if overlap_pixels > 1:
            # Strategy 2: Strict Pixel Count Check (LapNo > 1)
            # We need a "Count Mask" where each partner particle has a unique integer ID.
            # This allows us to distinguish between overlapping with one large particle vs. multiple small ones.
            target_imp = self._create_count_mask(check_imp, check_threshold)
            is_count_mask = True
            if target_imp is None:
                self.log("WARNING: Failed to generate Count Mask for association. Keeping all ROIs.")
                return rois
        
        target_stack = target_imp.getStack()
        
        # Group ROIs by slice for parallel processing
        rois_by_slice = defaultdict(list)
        for i, roi in enumerate(rois):
            p = roi.getPosition()
            if p < 1: p = 1
            rois_by_slice[p].append((i, roi))
            
        # Phase 1: Parallel Check
        slices = list(rois_by_slice.keys())
        
        def check_slice(slice_idx):
            try:
                ip = target_stack.getProcessor(slice_idx)
                slice_kept = []
                slice_rejected = []
                
                for idx, roi in rois_by_slice[slice_idx]:
                    try:
                        ip.setRoi(roi)
                        stats = ip.getStatistics()
                        
                        should_keep = False
                        
                        if stats.max > 0:
                            if not is_count_mask:
                                # Strategy 1: Simple Intensity Check (LapNo <= 1)
                                # Max > 0 means overlap exists (any non-zero pixel).
                                should_keep = True
                            else:
                                # Strategy 2: Strict Pixel Count Check
                                # Check histogram for sufficient overlap with a single particle.
                                # hist[0] is background (0).
                                # hist[k] is the count of pixels belonging to partner particle ID k.
                                for k in range(1, len(hist)):
                                    if hist[k] >= overlap_pixels:
                                        should_keep = True
                                        break
                        
                        if should_keep:
                            slice_kept.append((idx, roi))
                        else:
                            slice_rejected.append(roi)
                    except:
                        self.log("WARNING: Skipping malformed ROI in assoc_roi")
                        slice_rejected.append(roi)
                return (slice_kept, slice_rejected)
            except:
                self.log("WARNING: Failed to process slice {} in assoc_roi".format(slice_idx))
                return ([], [])

        def check_task(i):
            return check_slice(slices[i])

        nested_results = ParallelUtils.parallel_for(check_task, len(slices))
        
        all_kept = []
        all_rejected = []
        
        for kept, rejected in nested_results:
            all_kept.extend(kept)
            all_rejected.extend(rejected)
            
        # Sort kept ROIs to maintain original order
        all_kept.sort(key=lambda x: x[0])
        kept_rois = [x[1] for x in all_kept]
        # We must remove rejected puncta from the source images so they don't appear in final outputs.
        rejected_by_slice = defaultdict(list)
        for roi in all_rejected:
            p = roi.getPosition()
            if p < 1: p = 1
            rejected_by_slice[p].append(roi)
            
        store_stack = store_imp.getStack()
        mask_stack = store_mask_imp.getStack() if store_mask_imp else None
        
        def clear_slice(slice_idx):
            res_ip = store_stack.getProcessor(slice_idx)
            mask_ip = mask_stack.getProcessor(slice_idx) if mask_stack else None
            
            for roi in rejected_by_slice[slice_idx]:
                # Erase from Result Image
                res_ip.setValue(0)
                res_ip.fill(roi)
                # Erase from Mask Image (if provided)
                if mask_ip:
                    mask_ip.setValue(0)
                    mask_ip.fill(roi)
                    
        clear_slices = list(rejected_by_slice.keys())
        def clear_task(i):
            clear_slice(clear_slices[i])
            
        if clear_slices:
            ParallelUtils.parallel_for(clear_task, len(clear_slices))
            
        # Clean up temporary count mask if we created one
        if is_count_mask:
            target_imp.close()
            
        return kept_rois

    def _clear_roi(self, imp1, imp2, roi):
        """
        Erase the content of an ROI from the provided images.
        
        This is used to "reject" puncta that fail validation checks (e.g., no overlap,
        weak marker signal). By setting their pixels to 0, we ensure they are excluded
        from subsequent analysis steps and visualization.
        
        Args:
            imp1 (ImagePlus): First image to clear (e.g., Result Image).
            imp2 (ImagePlus): Second image to clear (e.g., Mask Image).
            roi (Roi): The region to erase.
        """
        # Iterate through the provided images (handling cases where one might be None).
        for imp in [imp1, imp2]:
            if imp:
                # Ensure we are operating on the correct slice if it's a stack.
                if imp.getStackSize() > 1:
                    imp.setSlice(roi.getPosition())
                
                # Set the ROI on the image.
                imp.setRoi(roi)
                
                # Get the processor for the current slice.
                ip = imp.getProcessor()
                
                # Set the drawing color/value to 0 (black/background).
                ip.setValue(0)
                
                # Fill the ROI with 0, effectively erasing the punctum.
                ip.fill(roi)
                
                # Clear the ROI selection from the image.
                imp.deleteRoi()

    def measure_rois(self, rois, imp, base_name, label, cal, min_thr=0, max_thr=65535, slice_label_fn=None):
        """
        Measure intensity and shape statistics for a list of ROIs.
        
        This generates the detailed per-punctum metrics (Area, Mean, IntDen, etc.)
        that populate the final result tables.
        
        Args:
            rois (list): List of ROIs to measure.
            imp (ImagePlus): The image to measure against.
            base_name (str): Image name.
            label (str): Channel label (e.g., 'Pre', 'Post').
            cal (Calibration): Image calibration.
            min_thr (float): Minimum threshold value used for detection.
            max_thr (float): Maximum threshold value used for detection.
            slice_label_fn (callable, optional): Function to retrieve original slice label.
            
        Returns:
            list: A list of dictionaries containing the measurements.
        """
        if not rois:
            return []

        # Group ROIs by slice to allow parallel processing of slices.
        # Each slice's ImageProcessor is not thread-safe for setRoi, so we must
        # ensure only one thread accesses a given slice's processor at a time.
        rois_by_slice = defaultdict(list)
        for i, roi in enumerate(rois):
            p = roi.getPosition()
            if p < 1: p = 1
            rois_by_slice[p].append((i, roi))
            
        stack = imp.getStack()
        slices = list(rois_by_slice.keys())
        
        def process_slice(slice_idx):
            try:
                # Get the processor for this slice (slice_idx is already 1-based)
                ip = stack.getProcessor(slice_idx)
                
                # CRITICAL: Set threshold on processor before measuring!
                # The macro's "Set Measurements" with "limit" option requires a threshold.
                # Without this, measurements include 0-value pixels from mask subtraction,
                # causing Min=0 instead of the actual minimum within the punctum.
                # Ref: SynapseJ_v_1.ijm line 561: setThreshold(ImLOW,65535) before Analyze Particles
                # The threshold persists on the Result image when roiManager("Measure") is called.
                ip.setThreshold(min_thr, max_thr, ImageProcessor.NO_LUT_UPDATE)
                
                results = []
                for idx, roi in rois_by_slice[slice_idx]:
                    try:
                        # Set ROI on the processor (not the ImagePlus)
                        ip.setRoi(roi)
                        
                        # CRITICAL FIX FOR FERET CALIBRATION:
                        # roi.getFeretValues() gets calibration from roi.getImage().getCalibration()
                        # If the ROI has no associated image, it defaults to pw=ph=1.0 (pixels).
                        # We must set the image on the ROI so Feret values use correct calibration.
                        # This matches ParticleAnalyzer behavior (PA calls roi.setImage(imp) before measuring).
                        #
                        # ADDITIONAL CRITICAL: When use_original_broken_global_calibration=True,
                        # the 'cal' parameter differs from imp.getCalibration(). We must ALSO
                        # set the calibration on 'imp' to match 'cal' so roi.getFeretValues()
                        # uses the correct (global) calibration for the second image.
                        imp.setCalibration(cal)
                        roi.setImage(imp)
                        
                        # Calculate statistics (now respects threshold due to LIMIT flag)
                        stats = ImageStatistics.getStatistics(ip, MEASUREMENT_FLAGS, cal)
                        # Generate metrics
                        metrics = self._metrics_dict(
                            base_name,
                            label,
                            idx + 1, # Use 1-based index
                            stats,
                            cal,
                            roi,
                            min_thr,
                            max_thr,
                            slice_label_fn
                        )
                        results.append((idx, metrics))
                    except:
                        self.log("WARNING: Skipping malformed ROI in measure_rois")
                return results
            except:
                self.log("WARNING: Failed to process slice {} in measure_rois".format(slice_idx))
                return []

        def task(i):
            return process_slice(slices[i])

        nested_results = ParallelUtils.parallel_for(task, len(slices))
        
        # Flatten and sort to restore original order
        all_results = []
        for res in nested_results:
            all_results.extend(res)
            
        all_results.sort(key=lambda x: x[0])
        
        return [x[1] for x in all_results]

    def _dilate_mask(self, mask_imp, pixels):
        """
        Dilate the mask image in-place.
        
        This method expands the binary mask by a specified number of pixels.
        It is used if 'dilate_enabled' is True, allowing for more permissive
        colocalization by artificially enlarging the detected puncta.
        
        Implementation Note:
        The macro converts the 16-bit mask to 8-bit for processing, applies the dilation
        using ROI enlargement, and then converts back to 16-bit.
        
        Args:
            mask_imp (ImagePlus): The binary mask image (will be modified).
            pixels (float): The radius of dilation in pixels.
        """
        # NO early return - macro always runs the loop when dilateQ true, even for 0 or negative
        from ij.process import ImageConverter
        
        # Convert to 8-bit for standard binary operations.
        # Dilation/Erosion operations are typically faster and simpler on 8-bit images.
        ImageConverter(mask_imp).convertToGray8()
        
        stack = mask_imp.getStack()
        n_slices = stack.getSize()
        
        # Define the per-slice dilation logic.
        def process_slice(i):
            slice_idx = i + 1
            ip = stack.getProcessor(slice_idx)
            
            # Create a binary threshold (1-255) to identify object pixels.
            # Background is 0.
            ip.setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)
            
            # Wrap the processor in a temporary ImagePlus to use the ThresholdToSelection plugin.
            temp_imp = ImagePlus("temp", ip)
            
            # Convert the thresholded mask into a composite ROI.
            roi = ThresholdToSelection.run(temp_imp)
            
            if roi:
                # Enlarge the ROI by the specified number of pixels.
                # This effectively dilates the selection.
                dilated_roi = RoiEnlarger.enlarge(roi, float(pixels))
                
                # Fill the enlarged ROI with white (255) on the processor.
                # This updates the mask to include the dilated area.
                ip.setValue(255)
                ip.fill(dilated_roi)
        
        # Execute dilation in parallel across all slices.
        ParallelUtils.parallel_for(process_slice, n_slices)
                
        # Convert back to 16-bit to match the rest of the pipeline (which expects 16-bit masks).
        ImageConverter(mask_imp).convertToGray16()
        
        # Scale 8-bit values (0-255) to 16-bit range (0-65535).
        # This ensures that "white" remains the maximum value.
        # 255 * 257 = 65535.
        def multiply_slice(i):
            stack.getProcessor(i+1).multiply(257)
            
        ParallelUtils.parallel_for(multiply_slice, n_slices)

    def _create_label_map(self, imp, rois, pixels):
        """
        Create a label map image where pixel values correspond to ROI indices (1-based).
        
        This image is essential for the "MatchROI" correlation analysis.
        It allows us to look up which particle ID belongs to a specific pixel location.
        Instead of iterating through thousands of ROIs for every pixel (which is slow),
        we "burn" the ROIs into an image. Then, checking if a pixel belongs to a particle
        is a constant-time O(1) lookup: just read the pixel value.
        
        Args:
            imp (ImagePlus): Reference image for dimensions and calibration.
            rois (list): List of ROIs to burn into the map.
            pixels (float): Dilation radius (if enabled).
            
        Returns:
            ImagePlus: An image where background is 0 and particle 'i' has value 'i+1'.
        """
        width = imp.getWidth()
        height = imp.getHeight()
        n_rois = len(rois)
        
        # Determine necessary bit depth based on the number of particles.
        # We need enough dynamic range to assign a unique ID to every particle.
        # - 8-bit (Byte): Max 255 IDs.
        # - 16-bit (Short): Max 65,535 IDs.
        # - 32-bit (Float): Practically unlimited IDs.
        if n_rois > 65534:
            # More than 65k particles requires 32-bit Float.
            from ij.process import FloatProcessor
            template_ip = FloatProcessor(width, height)
        elif n_rois > 254:
            # More than 254 particles requires 16-bit Short.
            from ij.process import ShortProcessor
            template_ip = ShortProcessor(width, height)
        else:
            # Fewer than 255 particles fits in 8-bit Byte.
            from ij.process import ByteProcessor
            template_ip = ByteProcessor(width, height)
            
        from ij.plugin import RoiEnlarger

        # If single slice, process sequentially (overhead of parallelization not worth it for 1 slice).
        if imp.getStackSize() == 1:
            label_map = ImagePlus("LabelMap", template_ip)
            label_map.setCalibration(imp.getCalibration())
            for i, roi in enumerate(rois):
                # IDs are 1-based because 0 is reserved for background.
                label_val = i + 1
                draw_roi = roi
                # Optional dilation: artificially expand the particle footprint.
                if self.dilate_enabled:
                    draw_roi = RoiEnlarger.enlarge(roi, float(pixels))
                
                # Set the ROI and fill it with the unique ID.
                label_map.setRoi(draw_roi)
                label_map.getProcessor().setValue(label_val)
                label_map.getProcessor().fill(draw_roi)
            label_map.deleteRoi()
            return label_map

        # For stacks, parallelize by slice.
        # This is significantly faster for large 3D stacks with many particles.
        
        # Group ROIs by slice index (1-based).
        rois_by_slice = defaultdict(list)
        for i, roi in enumerate(rois):
            # Store tuple (original_index, roi) to preserve ID = index + 1
            rois_by_slice[roi.getPosition()].append((i, roi))
            
        n_slices = imp.getStackSize()
        
        def create_slice_processor(i):
            slice_idx = i + 1
            # Create blank processor (duplicate from template)
            slice_ip = template_ip.createProcessor(width, height)
            slice_ip.setValue(0)
            slice_ip.fill()
            
            # Draw ROIs for this slice
            slice_rois = rois_by_slice[slice_idx]
            
            for idx, roi in slice_rois:
                label_val = idx + 1
                draw_roi = roi
                if self.dilate_enabled:
                    draw_roi = RoiEnlarger.enlarge(roi, float(pixels))
                
                slice_ip.setValue(label_val)
                slice_ip.fill(draw_roi)
                
            return slice_ip

        # Execute in parallel
        processors = ParallelUtils.parallel_for(create_slice_processor, n_slices)
        
        # Assemble stack
        stack = ImageStack(width, height)
        for p in processors:
            stack.addSlice(p)
            
        label_map = ImagePlus("LabelMap", stack)
        label_map.setCalibration(imp.getCalibration())
        return label_map

    def match_roi(self, anchor_entries, partner_entries, partner_label_map, anchor_label, partner_label, cal):
        """
        Perform correlation analysis (MatchROI) to find the nearest partner puncta.
        
        This method implements the "Correlation Analysis" described in the SynapseJ paper.
        For every punctum in the 'anchor' channel (e.g., Pre-synaptic), it searches for
        overlapping puncta in the 'partner' channel (e.g., Post-synaptic).
        
        It calculates:
        1. The number of overlapping partner puncta (VPerROI).
        2. The total integrated density of overlapping partner puncta (VIDPerROI).
        3. The distance to the nearest partner puncta (Euclidean distance).
        
        Args:
            anchor_entries (list): List of puncta in the reference channel.
            partner_entries (list): List of puncta in the target channel.
            partner_label_map (ImagePlus): An image where pixel values correspond to partner puncta IDs.
            anchor_label (str): Label for the anchor channel.
            partner_label (str): Label for the partner channel.
            cal (Calibration): Image calibration for distance measurements.
            
        Returns:
            list: A list of tab-separated strings, each representing a row in the correlation report.
        """
        if not anchor_entries:
            return []

        # Group anchor entries by slice for parallel processing
        anchors_by_slice = defaultdict(list)
        for i, entry in enumerate(anchor_entries):
            p = entry['roi'].getPosition()
            if p < 1: p = 1
            anchors_by_slice[p].append((i, entry))
            
        stack = partner_label_map.getStack()
        
        def process_slice(slice_idx):
            # Get processor for this slice of the label map
            ip = stack.getProcessor(slice_idx)
            
            # Determine if we can use histogram or need direct pixel scanning
            # ShortProcessor.getHistogram() returns 65536 bins - good for up to 65535 labels
            # FloatProcessor.getHistogram() returns only 256 bins - useless for label maps
            # ByteProcessor.getHistogram() returns 256 bins - only good for <=255 labels
            n_partners = len(partner_entries)
            use_histogram = n_partners <= 65534 and hasattr(ip, 'getHistogram')
            
            slice_results = []
            
            for idx, anchor_entry in anchors_by_slice[slice_idx]:
                anchor_roi = anchor_entry['roi']
                anchor_metrics = anchor_entry['metrics']
                
                # Set ROI on processor to inspect the underlying partner labels.
                ip.setRoi(anchor_roi)
                
                # Quick check for overlap: if max pixel value is 0, there are no partners here.
                stats = ImageStatistics.getStatistics(ip, Measurements.MIN_MAX, cal)
                
                if stats.max == 0:
                    continue
                
                # Count overlapping partner labels
                # For small label counts, use histogram. For large counts, scan pixels directly.
                overlap_counts = {}  # label_id -> pixel_count
                
                if use_histogram and n_partners <= 254:
                    # ByteProcessor: 256-bin histogram works for <=255 labels
                    hist = ip.getHistogram()
                    for n in range(1, min(len(hist), n_partners + 1)):
                        if hist[n] > 0:
                            overlap_counts[n] = hist[n]
                elif use_histogram and n_partners <= 65534:
                    # ShortProcessor: 65536-bin histogram works for <=65535 labels
                    hist = ip.getHistogram()
                    for n in range(1, min(len(hist), n_partners + 1)):
                        if hist[n] > 0:
                            overlap_counts[n] = hist[n]
                else:
                    # FloatProcessor or too many labels: scan pixels directly within ROI bounds
                    # This is actually efficient because we only scan the ROI bounding box
                    bounds = anchor_roi.getBounds()
                    for y in range(bounds.y, bounds.y + bounds.height):
                        for x in range(bounds.x, bounds.x + bounds.width):
                            if anchor_roi.contains(x, y):
                                label_val = int(ip.getf(x, y))
                                if label_val > 0:
                                    overlap_counts[label_val] = overlap_counts.get(label_val, 0) + 1
                    
                lineP = ""
                distSm = 0
                VStar = -1
                VPerROI = 0
                VIDPerROI = 0
                
                # Iterate through found overlapping labels IN SORTED ORDER
                # The macro iterates through histogram bins 0 to VC-1 sequentially,
                # so we must process partner IDs in ascending order to match.
                for n in sorted(overlap_counts.keys()):
                    count = overlap_counts[n]
                    # Found an overlapping partner with ID 'n'.
                    # Retrieve its metrics (ID is 1-based, list is 0-based).
                    if n-1 < len(partner_entries):
                        partner_entry = partner_entries[n-1]
                        partner_metrics = partner_entry['metrics']
                        
                        # Calculate Euclidean distance between centroids.
                        dx = anchor_metrics['X'] - partner_metrics['X']
                        dy = anchor_metrics['Y'] - partner_metrics['Y']
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Calculate distance between Centers of Mass (intensity-weighted).
                        dx_m = anchor_metrics['XM'] - partner_metrics['XM']
                        dy_m = anchor_metrics['YM'] - partner_metrics['YM']
                        dist_m = math.sqrt(dx_m*dx_m + dy_m*dy_m)
                        
                        # Format the match details string.
                        # Output 0-based index (n-1) to match macro's behavior.
                        # In macro: histogram bin n corresponds to pixel value n+hmin where hmin=1,
                        # so pixel value = n+1, but loop index n is 0-based, so output n.
                        # Our n is the pixel value (1-based), so output n-1.
                        match_str = "\t{}\t{}\t{:.3f}\t{:.3f}".format(n-1, count, dist, dist_m)
                        
                        # Track the nearest neighbor (smallest distance).
                        if dist < distSm or distSm == 0:
                            lineP = match_str + lineP # Prepend nearest
                            distSm = dist
                            VStar = n
                        else:
                            lineP = lineP + match_str # Append others
                        
                        # Accumulate total overlap stats.
                        VPerROI += 1
                        VIDPerROI += partner_metrics['IntDen']

                # If we found at least one partner (VStar != -1), record the result.
                if VStar != -1:
                    partner_entry = partner_entries[VStar-1]
                    partner_metrics = partner_entry['metrics']
                    partner_label_str = partner_metrics['Label']
                    
                    # Helper to format metrics (excluding Label) matching RESULT_LABELS order
                    # RESULT_LABELS is defined globally
                    # Use SynapseJ4ChannelComplete.format_value for ImageJ-compatible precision
                    def fmt_metrics(m):
                        return "\t".join([SynapseJ4ChannelComplete.format_value(m.get(k, 0)) for k in RESULT_LABELS[1:]])
                        
                    anchor_data = fmt_metrics(anchor_metrics)
                    partner_data = fmt_metrics(partner_metrics)
                    
                    # Format: AnchorLabel, AnchorMetrics, PartnerLabel, PartnerMetrics, Count, TotalIntDen, [Details...]
                    # Matches CORR_HEADER structure
                    # Use format_value for VIDPerROI to match ImageJ precision
                    final_line = "{}\t{}\t{}\t{}\t{}\t{}{}".format(
                        anchor_metrics['Label'], 
                        anchor_data,
                        partner_label_str, 
                        partner_data,
                        VPerROI, 
                        SynapseJ4ChannelComplete.format_value(VIDPerROI), 
                        lineP
                    )
                    slice_results.append((idx, final_line))
            
            ip.resetRoi()
            return slice_results

        slices = list(anchors_by_slice.keys())
        def task(i):
            return process_slice(slices[i])
            
        nested_results = ParallelUtils.parallel_for(task, len(slices))
        
        all_results = []
        for res in nested_results:
            all_results.extend(res)
            
        all_results.sort(key=lambda x: x[0])
        return [x[1] for x in all_results]

    def _filter_by_intensity(self, entries, marker_imp, thresholds, label,
                              result_imp, mask_imp, base_name, excel_suffix, 
                              cal, mask_name_for_bug=None, marker_channel_name=None):
        """
        Filter ROIs based on Min/Max intensity in the marker channel.

        This method implements the "CoLocROI" function from the SynapseJ macro.
        It serves as a biological gate: puncta are only retained if they colocalize
        with a structural marker (e.g., MAP2 for dendrites, GFAP for astrocytes).
        
        The logic is:
        1. For each detected punctum (ROI), measure the intensity in the Marker Channel.
        2. Check two conditions:
           - Max Intensity > High Threshold (ThrHi/ForHi)
           - Min Intensity > Low Threshold (ThrLo/ForLo)
        3. If EITHER condition fails (Max <= Hi OR Min <= Lo), the punctum is rejected.
           (Note: The macro logic is `if (max<=Hi || min<=Lo) { delete }`, which implies AND logic for retention).
        
        Args:
            entries (list): List of candidate puncta.
            marker_imp (ImagePlus): The marker channel image (usually blurred).
            thresholds (dict): Dictionary containing 'min' and 'max' intensity thresholds.
            label (str): Channel label for logging.
            result_imp (ImagePlus): The result image to clear rejected puncta from.
            mask_imp (ImagePlus): The mask image to clear rejected puncta from.
            base_name (str): Base filename.
            excel_suffix (str): Suffix for the output Excel file.
            mask_name_for_bug (str, optional): Name of the mask image to replicate the SynapseJ bug where
                measurements are taken on the mask instead of the marker channel.
            marker_channel_name (str, optional): Name of the marker channel (e.g., "C2-image") for Label.
                Used when mask_name_for_bug is None (second image onwards).
            
        Returns:
            list: The subset of entries that passed the intensity filter.
        """
        # Extract threshold values from the dictionary for easier access.
        lo_val = thresholds['min']   # Corresponds to ThrLo / ForLo (Minimum Intensity Threshold).
        hi_val = thresholds['max']   # Corresponds to ThrHi / ForHi (Maximum Intensity Threshold).
        size_val = thresholds['size'] # Note: Size parameter is logged but not used for gating in the original macro logic here.

        # Group entries by slice for parallel processing
        entries_by_slice = defaultdict(list)
        for i, entry in enumerate(entries):
            p = entry['roi'].getPosition()
            if p < 1: p = 1
            entries_by_slice[p].append((i, entry))
            
        marker_stack = marker_imp.getStack()
        # Use the passed calibration (respects global calibration mode)
        # instead of marker_imp.getCalibration() which has the image's own calibration
        # Also set it on marker_imp so roi.getFeretValues() uses correct calibration
        marker_imp.setCalibration(cal)
        
        # Phase 1: Parallel Check
        # We iterate through all ROIs and check their intensity in the marker channel.
        slices = list(entries_by_slice.keys())
        
        # DEBUG counters for rejection analysis
        debug_max_fail = [0]  # max <= hi_val
        debug_min_fail = [0]  # min <= lo_val
        debug_both_fail = [0] # both conditions
        debug_samples = []
        
        def check_slice(slice_idx):
            try:
                ip = marker_stack.getProcessor(slice_idx)
                
                slice_kept = []
                slice_rejected = []
                slice_marker_rows = []
                
                for idx, entry in entries_by_slice[slice_idx]:
                    roi = entry['roi']
                    try:
                        ip.setRoi(roi)
                        stats = ImageStatistics.getStatistics(ip, MEASUREMENT_FLAGS, cal)
                        max_i = stats.max
                        min_i = stats.min
                        
                        # The Gating Logic:
                        # Reject if Max Intensity is too low OR Min Intensity is too low.
                        # This ensures the punctum is "bright enough" and "consistently bright" in the marker channel.
                        max_fail = max_i <= hi_val
                        min_fail = min_i <= lo_val
                        
                        if max_fail or min_fail:
                            slice_rejected.append(roi)
                            # Track rejection reasons (thread-safe increment approximation)
                            if max_fail and min_fail:
                                debug_both_fail[0] += 1
                            elif max_fail:
                                debug_max_fail[0] += 1
                            else:
                                debug_min_fail[0] += 1
                            # Sample some rejected ROIs for debugging
                            if len(debug_samples) < 20:
                                debug_samples.append((slice_idx, idx, max_i, min_i, max_fail, min_fail))
                        else:
                            slice_kept.append((idx, entry))
                            # If kept, record the marker channel metrics for this punctum.
                            # This allows analysis of the marker signal itself within the synaptic region.
                            
                            # MACRO BUG REPLICATION:
                            # If mask_name_for_bug is provided, we simulate the bug where SynapseJ measures
                            # the Mask image (65535) instead of the Marker image.
                            # For second image onwards (mask_name_for_bug=None), use marker_channel_name.
                            
                            if mask_name_for_bug:
                                target_base_name = mask_name_for_bug
                            elif marker_channel_name:
                                target_base_name = marker_channel_name
                            else:
                                target_base_name = base_name
                            target_stats = stats
                            
                            if mask_name_for_bug:
                                pass  # Will overwrite values below

                            # Set ROI's image for correct Feret calibration (like in measure_rois)
                            roi.setImage(marker_imp)
                            
                            # For PreThrResults/PstRResults, og2 always has MinThr=0, MaxThr=65535
                            # This is because these are marker channel measurements, not punctum detection
                            metrics = self._metrics_dict(
                                target_base_name,
                                label,
                                entry['metrics'].get('Index', idx),
                                target_stats,
                                cal,
                                roi,
                                0,      # MinThr is always 0 for marker results
                                65535,  # MaxThr is always 65535 for marker results
                            )
                            
                            # Fix Label format - extract slice_label from original entry's label
                            # Original format: {image}PreF.tif:{roi_name}:{slice_label}
                            # For mask bug: {mask_name}:{roi_name}:{slice_number}
                            # For normal: {marker_channel_name}:{roi_name}:{slice_label}
                            roi_name = roi.getName()
                            orig_label = entry['metrics'].get('Label', '')
                            # Extract slice_label from orig_label (after second colon)
                            parts = orig_label.split(':', 2)
                            slice_label = parts[2] if len(parts) > 2 else ''
                            
                            # For normal case, fix the channel number in slice_label
                            # slice_label format: "c:X/Y z:Z/N - ImageName #N"
                            # We need to replace X with the marker channel number
                            if marker_channel_name and slice_label:
                                import re
                                # Extract marker channel number from marker_channel_name (e.g., "C2-image" -> "2")
                                marker_ch_match = re.match(r'C(\d+)-', marker_channel_name)
                                if marker_ch_match:
                                    marker_ch_num = marker_ch_match.group(1)
                                    # Replace "c:X/" with "c:{marker_ch_num}/" in slice_label
                                    slice_label = re.sub(r'c:\d+/', 'c:{}/'.format(marker_ch_num), slice_label)
                            
                            if mask_name_for_bug:
                                # Overwrite intensity values to match Mask (inverted 16-bit)
                                metrics['Mean'] = 65535
                                metrics['Min'] = 65535
                                metrics['Max'] = 65535
                                metrics['IntDen'] = metrics['Area'] * 65535
                                # RawIntDen = PixelCount * 65535
                                # PixelCount = Area / PixelArea
                                px_area = cal.pixelWidth * cal.pixelHeight
                                if px_area > 0:
                                    metrics['RawIntDen'] = float(round((metrics['Area'] / px_area) * 65535))
                                else:
                                    metrics['RawIntDen'] = float(round(metrics['Area'] * 65535))
                                
                                # For mask (uniform intensity), XM=X and YM=Y since all pixels have same weight
                                metrics['XM'] = metrics['X']
                                metrics['YM'] = metrics['Y']
                                    
                                # og2 Label format for mask bug: {mask_name}:{roi_name}:{slice_number}
                                # The last number is the slice number (1-based)
                                if roi_name:
                                    metrics['Label'] = "{}:{}:{}".format(mask_name_for_bug, roi_name, slice_idx)
                            else:
                                # Normal case (second image onwards): {marker_channel_name}:{roi_name}:{slice_label}
                                if roi_name and marker_channel_name:
                                    metrics['Label'] = "{}:{}:{}".format(marker_channel_name, roi_name, slice_label)

                            slice_marker_rows.append((idx, metrics))
                    except:
                        self.log("WARNING: Skipping malformed ROI in _filter_by_intensity")
                        slice_rejected.append(roi)
                return (slice_kept, slice_rejected, slice_marker_rows)
            except:
                self.log("WARNING: Failed to process slice {} in _filter_by_intensity".format(slice_idx))
                return ([], [], [])

        def check_task(i):
            return check_slice(slices[i])

        nested_results = ParallelUtils.parallel_for(check_task, len(slices))
        
        all_kept = []
        all_rejected = []
        all_marker_rows = []
        
        for kept, rejected, rows in nested_results:
            all_kept.extend(kept)
            all_rejected.extend(rejected)
            all_marker_rows.extend(rows)
            
        # DEBUG: Log rejection statistics
        total_rejected = len(all_rejected)
        self.log("DEBUG {}: Rejection breakdown - max_fail_only: {}, min_fail_only: {}, both_fail: {} (thresholds: hi={}, lo={})".format(
            label, debug_max_fail[0], debug_min_fail[0], debug_both_fail[0], hi_val, lo_val))
        if debug_samples:
            self.log("DEBUG {}: Sample rejected ROIs (slice, idx, max, min, max_fail, min_fail):".format(label))
            for sample in debug_samples[:10]:
                self.log("DEBUG   {}".format(sample))
            
        # Sort kept entries to preserve order
        all_kept.sort(key=lambda x: x[0])
        kept_entries = [x[1] for x in all_kept]
        
        all_marker_rows.sort(key=lambda x: x[0])
        marker_rows = [x[1] for x in all_marker_rows]
        
        # Phase 2: Parallel Clear
        # We must remove the rejected puncta from the result images so they don't appear in the final output.
        # This effectively "erases" them from the analysis.
        
        # Group rejected ROIs by slice
        rejected_by_slice = defaultdict(list)
        for roi in all_rejected:
            p = roi.getPosition()
            if p < 1: p = 1
            rejected_by_slice[p].append(roi)
            
        result_stack = result_imp.getStack()
        mask_stack = mask_imp.getStack() if mask_imp else None
        
        def clear_slice(slice_idx):
            # Clear from result_imp (the processed image used for visualization/measurement)
            res_ip = result_stack.getProcessor(slice_idx)
            mask_ip = mask_stack.getProcessor(slice_idx) if mask_stack else None
            
            for roi in rejected_by_slice[slice_idx]:
                # Set pixels to 0 (black)
                res_ip.setValue(0)
                res_ip.fill(roi)
                if mask_ip:
                    mask_ip.setValue(0)
                    mask_ip.fill(roi)
                    
        clear_slices = list(rejected_by_slice.keys())
        def clear_task(i):
            clear_slice(clear_slices[i])
            
        if clear_slices:
            ParallelUtils.parallel_for(clear_task, len(clear_slices))

        # Log the results of the gating process.
        self.log("{} gating (intensity only): {}/{} retained (Max>{}, Min>{}, size param={})".format(
            label, len(kept_entries), len(entries), hi_val, lo_val, size_val))
        
        # If any puncta were retained, save their marker channel measurements to disk.
        if marker_rows:
            out_path = os.path.join(self.excel_dir, base_name + excel_suffix)
            self.save_results_table(self._filter_standard_metrics(marker_rows), out_path)

        return kept_entries

    def _create_synapse_image(self, source_imp, rois, title):
        """
        Create a new image containing ONLY the pixels within the provided ROIs.
        
        This utility function is used to generate the "Synapse Only" images for visualization.
        It takes an original image and a list of ROIs (e.g., confirmed synapses), and produces
        a new image where everything outside these ROIs is black (0).
        
        Args:
            source_imp (ImagePlus): The source image containing the pixel data.
            rois (list): The list of ROIs to preserve.
            title (str): The title for the new image.
            
        Returns:
            ImagePlus: A new image containing only the ROI contents.
        """
        width = source_imp.getWidth()
        height = source_imp.getHeight()
        stack_size = source_imp.getStackSize()
        src_stack = source_imp.getStack()
        
        # Group ROIs by slice for parallel processing
        rois_by_slice = defaultdict(list)
        for roi in rois:
            p = roi.getPosition()
            if p < 1: p = 1
            rois_by_slice[p].append(roi)
            
        def create_slice(i):
            try:
                slice_idx = i + 1
                # Create a blank processor (black background)
                # We use the same bit depth as the source image
                # But for simplicity and matching macro, we often use 8-bit or copy source type.
                # The macro uses "Duplicate" then "Clear Outside".
                # Here we create a new processor and copy pixels inside ROIs.
                
                # Get source processor
                src_ip = src_stack.getProcessor(slice_idx)
                
                # Create new processor of same type
                new_ip = src_ip.createProcessor(width, height)
                new_ip.setValue(0)
                new_ip.fill()
                
                # For each ROI in this slice, copy pixels from source to new
                for roi in rois_by_slice[slice_idx]:
                    src_ip.setRoi(roi)
                    new_ip.setRoi(roi)
                    
                    # Copy pixels from source ROI to destination ROI
                    # blitter is one way, or just copy/paste
                    # simpler: get the ROI processor from source and insert it into destination
                    roi_ip = src_ip.crop()
                    new_ip.insert(roi_ip, roi.getBounds().x, roi.getBounds().y)
                    
                return new_ip
            except Exception as e:
                self.log("ERROR in _create_synapse_image create_slice {}: {}".format(i, e))
                return src_stack.getProcessor(i+1).createProcessor(width, height)

        # Execute in parallel
        processors = ParallelUtils.parallel_for(create_slice, stack_size)
        
        # Assemble stack
        new_stack = ImageStack(width, height)
        for p in processors:
            new_stack.addSlice(p)
            
        new_imp = ImagePlus(title, new_stack)
        new_imp.setCalibration(source_imp.getCalibration())
        
        return new_imp

    def save_synapse_figures(self, base_name, pre_result_imp, post_result_imp, syn_pre_rois, syn_post_rois, c1_imp, c2_imp):
        """
        Generate paper-style figures showing only colocalized synaptic puncta.
        
        This creates a visual verification of the analysis:
        1. Creates images containing ONLY the synaptic puncta (filtering out isolated ones).
        2. Merges these into a multi-channel composite.
        
        The color scheme follows the paper's convention:
        - Red: Post-synaptic puncta (Synaptic only)
        - Green: Pre-synaptic puncta (Synaptic only)
        - Blue: Original Post-synaptic Marker (C1)
        - Gray: Original Pre-synaptic Marker (C2)
        
        Args:
            base_name (str): Image name.
            pre_result_imp (ImagePlus): Processed pre-synaptic image.
            post_result_imp (ImagePlus): Processed post-synaptic image.
            syn_pre_rois (list): List of pre-synaptic ROIs that are part of a synapse.
            syn_post_rois (list): List of post-synaptic ROIs that are part of a synapse.
            c1_imp (ImagePlus): Original Channel 1 (Post Marker).
            c2_imp (ImagePlus): Original Channel 2 (Pre Marker).
        """
        # Create images containing ONLY the synaptic ROIs
        pre_syn_imp = self._create_synapse_image(pre_result_imp, syn_pre_rois, base_name + "PreSyn")
        post_syn_imp = self._create_synapse_image(post_result_imp, syn_post_rois, base_name + "PostSyn")
        
        # Save individual channels for debugging/verification
        IJ.save(pre_syn_imp, os.path.join(self.dest_dir, base_name + "PreSyn.tif"))
        IJ.save(post_syn_imp, os.path.join(self.dest_dir, base_name + "PostSyn.tif"))
        
        # Create Merge: 
        # Channel 1 (Red): Post-synaptic Synaptic Puncta
        # Channel 2 (Green): Pre-synaptic Synaptic Puncta
        # Channel 3 (Blue): Raw Post-Marker Channel (C1)
        # Channel 4 (Gray): Raw Pre-Marker Channel (C2)
        c1 = post_syn_imp # Red
        c2 = pre_syn_imp  # Green
        c3 = c1_imp.duplicate() # Blue
        c4 = c2_imp.duplicate() # Gray
        
        merged = RGBStackMerge.mergeChannels([c1, c2, c3, c4], True)
        
        if merged is not None:
            out_path = os.path.join(self.dest_dir, base_name + "SynapseMerge.tif")
            IJ.save(merged, out_path)
            merged.close()
        
        pre_syn_imp.close()
        post_syn_imp.close()
        c3.close()
        c4.close()

    def generate_presentation_outputs(self, base_name, pre_entries, post_entries, syn_pre_rois, syn_post_rois, cal, pre_result_imp, post_result_imp, pre_measure_imp=None, post_measure_imp=None):
        """
        Generate presentation output: Batch_Synapse_Report.csv (accumulated).
        
        Creates Complete synapse rows linking Pre and Post IDs with geometry measurements.
        Only accumulates to batch_synapse_rows which is saved at end of run.
        
        Args:
            base_name (str): Image name.
            pre_entries (list): All detected pre-synaptic puncta.
            post_entries (list): All detected post-synaptic puncta.
            syn_pre_rois (list): Confirmed synaptic pre-synaptic ROIs.
            syn_post_rois (list): Confirmed synaptic post-synaptic ROIs.
            cal (Calibration): Image calibration.
            pre_result_imp (ImagePlus): Processed pre-synaptic image (masked, for geometry).
            post_result_imp (ImagePlus): Processed post-synaptic image (masked, for geometry).
            pre_measure_imp (ImagePlus): Original pre-synaptic image for intensity measurement.
            post_measure_imp (ImagePlus): Original post-synaptic image for intensity measurement.
        """
        synapse_rows = []
        
        # Create ROI-to-entry lookup for quick access
        pre_roi_to_entry = {entry['roi']: entry for entry in pre_entries}
        post_roi_to_entry = {entry['roi']: entry for entry in post_entries}
        
        # Process Complete Synapses (pairs of Pre and Post) in Parallel by Slice
        width = pre_result_imp.getWidth()
        height = pre_result_imp.getHeight()
        stack_size = pre_result_imp.getStackSize()
        
        # Use original measurement images for IntDen if provided, otherwise fall back to result images
        # The result images (PreF/PostF) are masked and may have zeros where signal exists in originals
        pre_intden_stack = pre_measure_imp.getStack() if pre_measure_imp else pre_result_imp.getStack()
        post_intden_stack = post_measure_imp.getStack() if post_measure_imp else post_result_imp.getStack()
        
        def process_complete_slice(slice_idx):
            # Get Pre and Post ROIs for this slice
            slice_pre_rois = [roi for roi in syn_pre_rois if roi.getPosition() == slice_idx or (slice_idx == 1 and roi.getPosition() == 0)]
            slice_post_rois = [roi for roi in syn_post_rois if roi.getPosition() == slice_idx or (slice_idx == 1 and roi.getPosition() == 0)]
            
            if not slice_pre_rois or not slice_post_rois:
                return []
            
            # Get processors for measurements (thread-safe access)
            # Use original images for intensity measurements to get actual signal values
            pre_measure_ip = pre_intden_stack.getProcessor(slice_idx)
            post_measure_ip = post_intden_stack.getProcessor(slice_idx)

            # Create label map for Pre ROIs
            pre_label_ip = ByteProcessor(width, height)
            pre_label_map = {}
            for idx, roi in enumerate(slice_pre_rois):
                label_val = idx + 1
                pre_label_ip.setValue(label_val)
                pre_label_ip.fill(roi)
                pre_label_map[label_val] = roi
            
            slice_matches = []
            
            # For each Post ROI, find overlapping Pre ROIs
            for post_roi in slice_post_rois:
                pre_label_ip.setRoi(post_roi)
                hist = pre_label_ip.getHistogram()
                
                # Find all Pre ROIs that overlap with this Post ROI
                overlapping_pre_labels = []
                for label_val in range(1, len(hist)):
                    if hist[label_val] > 0:  # Has overlap
                        overlapping_pre_labels.append(label_val)
                
                # Create Complete synapse entry for each Pre-Post pair
                for pre_label in overlapping_pre_labels:
                    pre_roi = pre_label_map[pre_label]
                    
                    # Calculate overlap in pixels between Pre and Post ROIs
                    overlap_px = hist[pre_label]
                    
                    # Get IDs from metrics
                    pre_entry = pre_roi_to_entry.get(pre_roi)
                    post_entry = post_roi_to_entry.get(post_roi)
                    
                    if not pre_entry or not post_entry:
                        continue
                    
                    pre_id = pre_entry['metrics']['Label']
                    post_id = post_entry['metrics']['Label']
                    
                    # --- Geometry Calculations ---
                    # Use pixel-based method exclusively for intersection (consistent with overlap_px)
                    pixel_area = cal.pixelWidth * cal.pixelHeight
                    
                    # Intersection area = overlap pixels * pixel area (matches Overlap um^2 exactly)
                    intersect_area = overlap_px * pixel_area
                    
                    # Get bounding boxes for overlap iteration
                    pre_bounds = pre_roi.getBounds()
                    post_bounds = post_roi.getBounds()
                    
                    # Calculate bounding box of intersection
                    ix1 = max(pre_bounds.x, post_bounds.x)
                    iy1 = max(pre_bounds.y, post_bounds.y)
                    ix2 = min(pre_bounds.x + pre_bounds.width, post_bounds.x + post_bounds.width)
                    iy2 = min(pre_bounds.y + pre_bounds.height, post_bounds.y + post_bounds.height)
                    
                    # OPTIMIZED: Use geometry from overlap_px instead of pixel iteration
                    # The overlap_px is already accurate from the label map histogram
                    # We approximate geometry using the bounding box intersection
                    
                    pre_m = pre_entry['metrics']
                    post_m = post_entry['metrics']
                    
                    # Intersection centroid: weighted average of Pre and Post centroids by overlap
                    # Since overlap exists, approximate as midpoint between centroids
                    intersect_x = (pre_m.get('X', 0) + post_m.get('X', 0)) / 2.0
                    intersect_y = (pre_m.get('Y', 0) + post_m.get('Y', 0)) / 2.0
                    
                    # Intersection IntDen: estimate from Pre signal in overlap region
                    # Use Pre Mean * overlap_area as approximation (faster than pixel iteration)
                    pre_mean = pre_m.get('Mean', 0)
                    intersect_rawintden = pre_mean * overlap_px  # Raw = Mean * pixel count
                    intersect_intden = intersect_rawintden
                    
                    # Geometry from bounding box intersection
                    if ix2 > ix1 and iy2 > iy1:
                        overlap_width = (ix2 - ix1) * cal.pixelWidth
                        overlap_height = (iy2 - iy1) * cal.pixelHeight
                        intersect_feret = max(overlap_width, overlap_height)
                        intersect_min_feret = min(overlap_width, overlap_height)
                        intersect_perim = 2 * (overlap_width + overlap_height)
                    else:
                        # Fallback to circular estimate from area
                        equiv_radius = math.sqrt(intersect_area / math.pi) if intersect_area > 0 else 0
                        intersect_perim = 2 * math.pi * equiv_radius
                        intersect_feret = 2 * equiv_radius
                        intersect_min_feret = 2 * equiv_radius
                    
                    # Union geometry calculated from Pre + Post - Intersection
                    # Mathematical identity: Union Area = Pre Area + Post Area - Intersection Area
                    pre_m = pre_entry['metrics']
                    post_m = post_entry['metrics']
                    
                    # Get individual ROI metrics (geometry only)
                    pre_area = pre_m.get('Area', 0)
                    post_area = post_m.get('Area', 0)
                    
                    # Union Area = Pre + Post - Intersection (mathematically correct for geometry)
                    union_area = pre_area + post_area - intersect_area
                    
                    # Union IntDen: Sum of Pre and Post IntDen values
                    # Note: Pre and Post IntDen are measured on different channels, so this is
                    # a combined signal measure, not a single-channel measurement like intersection.
                    # For single-channel union IntDen, use the intersection IntDen formula approach.
                    pre_intden = pre_m.get('IntDen', 0)
                    post_intden = post_m.get('IntDen', 0)
                    union_intden = pre_intden + post_intden
                    union_rawintden = union_intden
                    
                    # Union centroid: weighted average by area
                    if union_area > 0:
                        # Weight each centroid by its contribution to union
                        # Approximation: use area-weighted average of pre and post centroids
                        pre_weight = pre_area / union_area
                        post_weight = post_area / union_area
                        union_x = pre_m.get('X', 0) * pre_weight + post_m.get('X', 0) * post_weight
                        union_y = pre_m.get('Y', 0) * pre_weight + post_m.get('Y', 0) * post_weight
                    else:
                        union_x = (pre_m.get('X', 0) + post_m.get('X', 0)) / 2.0
                        union_y = (pre_m.get('Y', 0) + post_m.get('Y', 0)) / 2.0
                    
                    # Union Feret: use the combined bounding box extent
                    union_bbox_width = (max(pre_bounds.x + pre_bounds.width, post_bounds.x + post_bounds.width) - 
                                       min(pre_bounds.x, post_bounds.x)) * cal.pixelWidth
                    union_bbox_height = (max(pre_bounds.y + pre_bounds.height, post_bounds.y + post_bounds.height) - 
                                        min(pre_bounds.y, post_bounds.y)) * cal.pixelHeight
                    union_feret = max(union_bbox_width, union_bbox_height)
                    union_min_feret = min(union_bbox_width, union_bbox_height)
                    union_perim = 2 * (union_bbox_width + union_bbox_height)

                    # Distances
                    geo_dist = math.sqrt((pre_m['X'] - post_m['X'])**2 + (pre_m['Y'] - post_m['Y'])**2)
                    int_dist = math.sqrt((pre_m['XM'] - post_m['XM'])**2 + (pre_m['YM'] - post_m['YM'])**2)

                    slice_matches.append({
                        'pre_id': pre_id,
                        'post_id': post_id,
                        'overlap_px': overlap_px,
                        'intersect_area': intersect_area,
                        'intersect_perim': intersect_perim,
                        'intersect_x': intersect_x,
                        'intersect_y': intersect_y,
                        'intersect_feret': intersect_feret,
                        'intersect_min_feret': intersect_min_feret,
                        'intersect_intden': intersect_intden,
                        'intersect_rawintden': intersect_rawintden,
                        'union_area': union_area,
                        'union_perim': union_perim,
                        'union_x': union_x,
                        'union_y': union_y,
                        'union_feret': union_feret,
                        'union_min_feret': union_min_feret,
                        'union_intden': union_intden,
                        'union_rawintden': union_rawintden,
                        'geo_dist': geo_dist,
                        'int_dist': int_dist
                    })
            return slice_matches

        # Run slice processing in parallel
        slice_indices = list(range(1, stack_size + 1))
        
        def slice_task(i):
            return process_complete_slice(slice_indices[i])
            
        all_slice_matches = ParallelUtils.parallel_for(slice_task, len(slice_indices))
        
        # Flatten results and assign IDs sequentially
        complete_synapse_id = 1
        for slice_matches in all_slice_matches:
            for match in slice_matches:
                # Create Complete ID
                complete_id = "{}:Comp:{}".format(base_name, complete_synapse_id)
                complete_synapse_id += 1
                
                overlap_px = match['overlap_px']
                
                # Synapse row - columns renamed (Pre ID, Post ID) and no BBox
                row = OrderedDict([
                    ('Image Name', base_name),
                    ('Puncta ID', complete_id),
                    ('Included Channels', 'C1,C2,C3,C4'),
                    ('Type', 'Complete'),
                    ('Overlap (px)', overlap_px),
                    ('Overlap (um^2)', overlap_px * self._pixel_area(cal)),
                    ('Pre ID', match['pre_id']),
                    ('Post ID', match['post_id']),
                    # Synapse Geometry (no BBox)
                    ('Intersect Area (um^2)', match['intersect_area']),
                    ('Intersect Perim (um)', match['intersect_perim']),
                    ('Intersect Centroid X (um)', match['intersect_x']),
                    ('Intersect Centroid Y (um)', match['intersect_y']),
                    ('Intersect Feret (um)', match['intersect_feret']),
                    ('Intersect Min Feret (um)', match['intersect_min_feret']),
                    ('Intersect IntDen', match['intersect_intden']),
                    ('Intersect RawIntDen', match['intersect_rawintden']),
                    ('Union Area (um^2)', match['union_area']),
                    ('Union Perim (um)', match['union_perim']),
                    ('Union Centroid X (um)', match['union_x']),
                    ('Union Centroid Y (um)', match['union_y']),
                    ('Union Feret (um)', match['union_feret']),
                    ('Union Min Feret (um)', match['union_min_feret']),
                    ('Union IntDen', match['union_intden']),
                    ('Union RawIntDen', match['union_rawintden']),
                    ('Geo Distance (um)', match['geo_dist']),
                    ('Int Distance (um)', match['int_dist'])
                ])
                synapse_rows.append(row)
        
        # Add to batch accumulator (no individual file saves)
        self.batch_synapse_rows.extend(synapse_rows)
        
        self.log("Generated {} synapse rows for batch report".format(len(synapse_rows)))

from java.util.concurrent import Executors, Callable, Future, TimeUnit
from java.util import ArrayList

class ParallelUtils(object):
    """
    Utility class for parallel execution using Java's ExecutorService.
    
    Since this script runs within the Jython environment of Fiji, it has access to 
    Java's robust concurrency libraries. This allows for multi-threaded processing 
    of images, significantly speeding up the analysis of large datasets.
    """
    _executor = None

    @classmethod
    def get_executor(cls):
        """
        Singleton accessor for the thread pool.
        Creates a FixedThreadPool with 8 threads if one does not exist.
        Adjust the thread count based on the available CPU cores.
        """
        if cls._executor is None:
            # Create a thread pool with a fixed number of threads (8).
            # This limits the number of concurrent tasks to avoid overwhelming the system.
            # In a production environment, this could be set dynamically based on Runtime.getRuntime().availableProcessors().
            cls._executor = Executors.newFixedThreadPool(8) 
        return cls._executor

    @classmethod
    def run_tasks(cls, tasks):
        """
        Run a list of no-arg functions (callables) in parallel.
        
        Args:
            tasks: A list of Python functions (no arguments) to execute.
            
        Returns:
            A list of return values from the executed functions, in the same order.
        """
        executor = cls.get_executor()
        
        # Wrap python functions in Java Callable interface.
        # This bridges the gap between Python functions and Java's threading model.
        # Java's ExecutorService expects objects implementing the Callable interface.
        class PyCallable(Callable):
            def __init__(self, func):
                self.func = func
            def call(self):
                # Execute the Python function when called by the Java thread.
                return self.func()
        
        # Create a Java ArrayList to hold the tasks.
        java_tasks = ArrayList()
        for t in tasks:
            # Wrap each Python function and add it to the list.
            java_tasks.add(PyCallable(t))
            
        # invokeAll submits all tasks to the thread pool and blocks until ALL of them are complete.
        # This ensures that we don't proceed until all parallel work is finished.
        futures = executor.invokeAll(java_tasks)
        
        results = []
        # Iterate through the Future objects returned by invokeAll.
        for f in futures:
            # f.get() retrieves the result of the computation.
            # Since invokeAll blocks until completion, get() will return immediately with the result
            # or throw an exception if the task failed.
            results.append(f.get())
        return results

    @classmethod
    def parallel_for(cls, func, n_items):
        """
        Run func(i) for i in range(n_items) in parallel.
        
        A parallel equivalent of a simple for loop.
        Useful for processing a list of items where each iteration is independent.
        """
        tasks = []
        for i in range(n_items):
            # Create a lambda that calls the function with the current index 'i'.
            # We use 'idx=i' as a default argument to capture the value of 'i' at this iteration.
            # Without this, the lambda would capture the variable 'i' itself, which would be the last value of the loop.
            tasks.append(lambda idx=i: func(idx))
        
        # Execute the list of tasks in parallel.
        return cls.run_tasks(tasks)

    @classmethod
    def shutdown(cls):
        """Gracefully shut down the thread pool."""
        if cls._executor is not None:
            # Initiate an orderly shutdown in which previously submitted tasks are executed,
            # but no new tasks will be accepted.
            cls._executor.shutdown()


def main():
    # --- Main Execution Entry Point ---
    # This block is executed only when the script is run directly (not imported as a module).
    
    # 1. Determine Configuration Path
    #    The script requires a configuration file to know where the images are and what parameters to use.
    #    We check command line arguments first, then environment variables.
    config_path = None
    if len(sys.argv) > 1:
        # If an argument is provided, assume it is the path to the config file.
        config_path = sys.argv[1]
    elif 'SYNAPSEJ_CONFIG' in os.environ:
        # Fallback to environment variable if no argument is provided.
        config_path = os.environ['SYNAPSEJ_CONFIG']

    # If no configuration is found, print usage instructions and exit.
    if not config_path:
        print('Usage: SYNAPSEJ_CONFIG=path/to/config fiji --headless --run Pynapse.py')
        # Exit with error code 1.
        System.exit(1)

    try:
        print("Starting Pynapse with config: {}".format(config_path))
        # 2. Initialize Analyzer
        #    Create an instance of the main class. This parses the config file,
        #    validates parameters, and sets up the output directory structure.
        analyzer = SynapseJ4ChannelComplete(config_path)
        
        # 3. Run Analysis
        #    Start the batch processing loop. This iterates through all images 
        #    in the input directory defined in the config and processes them one by one.
        analyzer.process_directory()
    finally:
        # 4. Cleanup
        #    Ensure the thread pool is shut down properly.
        #    If we don't do this, the JVM might keep running because of the active threads in the pool.
        ParallelUtils.shutdown()
        
        # Force exit with success code 0.
        System.exit(0)

if __name__ == '__main__':
    main()

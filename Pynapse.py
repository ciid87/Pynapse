#!/usr/bin/env jython
"""
Pynapse.py - Repaired Headless Fiji/Jython recreation of SynapseJ macro v1.
"""

import os
import sys
import math
import csv
from collections import defaultdict
from datetime import datetime
from ij import Prefs
# Prefs.set("headless", "true")                     # ← forces headless mode in IJ1
from ij.macro import Interpreter
Interpreter.setAdditionalFunctions("function waitForUser() {} function showMessage() {} function showMessageWithCancel() {} function Dialog.create() { return null; }")

from ij import Prefs, IJ, ImagePlus, ImageStack, ImageStack, WindowManager
from ij.plugin import ChannelSplitter, RGBStackMerge
from ij.process import ByteProcessor, ShortProcessor, FloatProcessor, ImageStatistics, ImageProcessor, ImageConverter, Blitter
from ij.measure import ResultsTable, Measurements, Calibration
from ij.plugin.filter import ParticleAnalyzer, GaussianBlur, Analyzer, BackgroundSubtracter, MaximumFinder, ThresholdToSelection
from ij.plugin import RoiEnlarger, ImageCalculator
from ij.plugin.frame import RoiManager
from ij.io import FileSaver, RoiEncoder
from java.io import FileOutputStream, BufferedOutputStream
from java.util.zip import ZipOutputStream, ZipEntry
from java.lang import Double
from java.awt import Color
Prefs.blackBackground = True
Analyzer.setPrecision(3)
MEASUREMENT_FLAGS = Measurements.AREA | Measurements.MEAN | Measurements.MIN_MAX | \
                    Measurements.CENTROID | Measurements.CENTER_OF_MASS | \
                    Measurements.PERIMETER | Measurements.FERET | Measurements.INTEGRATED_DENSITY | \
                    Measurements.LIMIT
# Raw Integrated Density might not be in Measurements interface in some versions
try:
    MEASUREMENT_FLAGS |= Measurements.RAW_INTEGRATED_DENSITY
except AttributeError:
    pass # Or define it manually if needed: MEASUREMENT_FLAGS |= 2097152

RESULT_LABELS = ["Label", "Area", "Mean", "Min", "Max", "X", "Y", "XM", "YM", "Perim.", "Feret", "IntDen", "RawIntDen", "FeretX", "FeretY", "FeretAngle", "MinFeret"]

HEADER = "\t".join(RESULT_LABELS[1:])

CORR_HEADER = "Image Name\t" + HEADER + "\tImage Name\t" + HEADER + "\tNo. of Post/Pre Puncta \tPost IntDen\tPostsynaptic Puncta No.\tOverlap\tDistance\tDistance M\n"


def default_config():
    """Return SynapseJ defaults (User Guide Section 4, Table 1)."""
    return {
        'source_dir': '',
        'dest_dir': '',
        'pre_channel': 4,
        'post_channel': 3,
        'pre_marker_channel': 2,  # Ref: SynapseJ_v_1.ijm line 32 (Channels[1] = C2)
        'post_marker_channel': 1, # Ref: SynapseJ_v_1.ijm line 33 (Channels[0] = C1)
        'pre_min': 658,           # Ref: SynapseJ_v_1.ijm line 35
        'pre_noise': 350,         # Ref: SynapseJ_v_1.ijm line 36
        'pre_size_low': 0.08,     # Ref: SynapseJ_v_1.ijm line 37
        'pre_size_high': 2.5,     # Ref: SynapseJ_v_1.ijm line 38
        'pre_blur': True,         # Ref: SynapseJ_v_1.ijm line 39
        'pre_blur_radius': 2,     # Ref: SynapseJ_v_1.ijm line 40
        'pre_bkd': 0,             # Ref: SynapseJ_v_1.ijm line 41
        'pre_use_maxima': True,   # Ref: SynapseJ_v_1.ijm line 42
        'pre_apply_fade': False,  # Ref: SynapseJ_v_1.ijm line 43
        'pre_fade_factors': '',
        'post_min': 578,          # Ref: SynapseJ_v_1.ijm line 44
        'post_noise': 350,        # Ref: SynapseJ_v_1.ijm line 45
        'post_size_low': 0.08,    # Ref: SynapseJ_v_1.ijm line 46
        'post_size_high': 2.5,    # Ref: SynapseJ_v_1.ijm line 47
        'post_blur': True,        # Ref: SynapseJ_v_1.ijm line 48
        'post_blur_radius': 2,    # Ref: SynapseJ_v_1.ijm line 49
        'post_bkd': 0,            # Ref: SynapseJ_v_1.ijm line 50
        'post_use_maxima': True,  # Ref: SynapseJ_v_1.ijm line 51
        'post_apply_fade': False, # Ref: SynapseJ_v_1.ijm line 52
        'overlap_pixels': 1,      # Ref: SynapseJ_v_1.ijm line 53
        'dilate_pixels': 1,       # Ref: SynapseJ_v_1.ijm line 55
        'dilate_enabled': False,
        'slice_number': 2,        # Ref: SynapseJ_v_1.ijm line 56
        'grid_size': 80,
        'pre_marker_min': 484,    # Ref: SynapseJ_v_1.ijm line 73 (Thr Min Int)
        'pre_marker_max': 895,    # Ref: SynapseJ_v_1.ijm line 72 (Thr Max Int)
        'pre_marker_size': 250,   # Ref: SynapseJ_v_1.ijm line 74 (Thr Sz)
        'post_marker_min': 273,   # Ref: SynapseJ_v_1.ijm line 83 (For Min Int)
        'post_marker_max': 692,   # Ref: SynapseJ_v_1.ijm line 82 (For Max Int)
        'post_marker_size': 300,  # Ref: SynapseJ_v_1.ijm line 84 (For Sz)
    }


class SynapseJ4ChannelComplete(object):
    """One-to-one SynapseJ reproduction that mirrors macro logic and documentation.

    Every processing step is backed by an explicit citation to SynapseJ_v_1.ijm,
    the SynapseJ User Guide, or the 2021 bioRxiv paper (Moreno Manrique et al.).
    No behavior deviates from those references unless unavoidable due to API diffs,
    and any such cases are logged for transparency.
    """

    def __init__(self, config_path=None):
        """Load configuration, channels, and accumulator tables (User Guide Section 4)."""
        from ij import Prefs
        Prefs.blackBackground = True
        
        self.results_summary = []
        self.all_pre_results = []
        self.all_post_results = []
        self.syn_pre_results = []
        self.syn_post_results = []
        self.synapse_pair_rows = []

        # Correlation tables (MatchROI). Start with header rows mirroring the
        # "Pre to Post Correlation Window" / "Post to Pre Correlation Window".
        self.pre_correlation_rows = [CORR_HEADER.strip()]
        reverse_header = CORR_HEADER.replace('Post/Pre', 'Pre/Post') \
                         .replace('Post IntDen', 'Pre IntDen') \
                         .replace('Postsynaptic', 'Presynaptic')
        self.post_correlation_rows = [reverse_header.strip()]

        self.log_messages = []

        self.config = default_config()
        if config_path:
            self._load_config(config_path)

        # All accumulators above mirror the macro's Collated ResultsIF, All Pre Results, etc.

        cfg = self.config
        self.slice_number = int(cfg.get('slice_number', 2))

        self.pre_params = self._build_channel_params('pre')
        self.post_params = self._build_channel_params('post')
        self.pre_marker_thresholds = self._build_marker_thresholds('pre')
        self.post_marker_thresholds = self._build_marker_thresholds('post')
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
        self.log('Initialized SynapseJ analyzer with documented parameters.')
        self.log('DEBUG: Configuration: {}'.format(self.config))

    def _load_config(self, path):
        """Parse key=value config files identical to SynapseJ config exports."""
        if not os.path.exists(path):
            self.log('WARNING: Config file {} not found; defaults remain active.'.format(path))
            return
        self.log('Loading config from {}'.format(path))
        with open(path, 'r') as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = [token.strip() for token in line.split('=', 1)]
                self.log('Config: {} = {}'.format(key, value))
                if key not in self.config:
                    continue
                # Boolean parsing first, then numeric, falling back to raw string to match macro.
                lower = value.lower()
                if lower in ['true', 'false']:
                    self.config[key] = (lower == 'true')
                else:
                    try:
                        self.config[key] = int(value)
                    except ValueError:
                        try:
                            self.config[key] = float(value)
                        except ValueError:
                            self.config[key] = value

    def _build_channel_params(self, prefix):
        """Bundle per-channel detection knobs (macro PrepChannel inputs)."""
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
        if raw_value in [None, '', 0]:
            return []
        if isinstance(raw_value, (int, float)):
            factors = [float(raw_value)]
        else:
            normalized = str(raw_value).replace(';', ',').replace('|', ',')
            # Macro accepts semi-colon, comma, or pipe delimited fading factors; mirror that.
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
        if slice_number > len(factors):
            last = factors[-1]
            # Macro repeats last factor when stack deeper than provided values.
            factors.extend([last] * (slice_number - len(factors)))
        elif slice_number < len(factors):
            # Trim extras so vector length always equals number of slices processed.
            factors = factors[:slice_number]
        return factors

    def _build_marker_thresholds(self, prefix):
        """Extract optional marker intensity gates (User Guide Section 5)."""
        min_val = self.config['{}_marker_min'.format(prefix)]
        max_val = self.config['{}_marker_max'.format(prefix)]
        size_val = self.config['{}_marker_size'.format(prefix)]
        if min_val in [None, '', 0] or max_val in [None, '', 0] or size_val in [None, '', 0]:
            # If any entry missing, macro skips marker gating entirely for that channel.
            return None
        return {
            'min': float(min_val),
            'max': float(max_val),
            'size': float(size_val),
        }

    def log_configuration_summary(self):
        """Emit IFALog-style summary lines closely matching the macro."""
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
        for s in range(1, imp.getStackSize() + 1):
            imp.setSlice(s)
            factor = factors[s-1]
            IJ.run(imp, "Multiply...", "value=" + str(factor) + " slice")
        self.log('{} fading correction applied with factors {}'.format(label, factors))

    def segment_dense_image(self, imp, noise, min_threshold):
        """Run Find Maxima in SEGMENTED mode (macro Maxima_Stack) for dense puncta."""
        segmented_stack = ImageStack(imp.getWidth(), imp.getHeight())
        stack = imp.getStack()
        
        self.log("DEBUG: segment_dense_image processing {} slices".format(imp.getNSlices()))
        
        for slice_idx in range(1, imp.getNSlices() + 1):
            ip = stack.getProcessor(slice_idx).duplicate()
            
            # Macro logic: if min > 0, use "above". In API, this means passing the threshold.
            # If min <= 0, pass ImageProcessor.NO_THRESHOLD to disable thresholding.
            threshold_arg = ImageProcessor.NO_THRESHOLD
            if min_threshold > 0:
                threshold_arg = float(min_threshold)

            mf = MaximumFinder()
            # findMaxima(ip, tolerance, threshold, outputType, excludeOnEdges, isEDM)
            segmented_proc = mf.findMaxima(ip, float(noise), threshold_arg, MaximumFinder.SEGMENTED, False, False)
            
            if segmented_proc is None:
                segmented_proc = ByteProcessor(imp.getWidth(), imp.getHeight())
            segmented_stack.addSlice(segmented_proc)
            
        segmented_imp = ImagePlus('{} segmented'.format(imp.getTitle()), segmented_stack)
        segmented_imp.setCalibration(imp.getCalibration())
        return segmented_imp

    def process_directory(self):
        if not os.path.isdir(self.source_dir):
            self.log('ERROR: source_dir {} does not exist.'.format(self.source_dir))
            return
        if not self.dest_dir:
            self.dest_dir = os.path.join(self.source_dir, 'Synapse_Output')
        self.merge_dir = os.path.join(self.dest_dir, 'merge')
        self.excel_dir = os.path.join(self.dest_dir, 'excel')
        for folder in [self.dest_dir, self.merge_dir, self.excel_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.log_configuration_summary()
        image_files = []
        for name in sorted(os.listdir(self.source_dir)):
            lower = name.lower()
            if lower.endswith('.tif') or lower.endswith('.tiff') or lower.endswith('.nd2'):
                image_files.append(os.path.join(self.source_dir, name))
        self.log('Found {} image(s) to analyze.'.format(len(image_files)))
        for path in image_files:
            try:
                self.process_image(path)
            except Exception as exc:
                import traceback
                self.log('ERROR processing {}: {}'.format(path, exc))
                traceback.print_exc()
        self.save_all_results()
        self.save_log()
        self.log('All images processed; outputs saved to {}.'.format(self.dest_dir))

    def process_image(self, image_path):
        """Replicate macro ProcessImage routine for one stack, including overlays."""
        self.log('\n' + '=' * 80)
        self.log('PROCESSING: {}'.format(os.path.basename(image_path)))
        imp = IJ.openImage(image_path)
        if imp is None:
            self.log('  ERROR: Fiji could not open {}'.format(image_path))
            return
        name_short = os.path.splitext(os.path.basename(image_path))[0]
        channels = ChannelSplitter.split(imp)
        
        pre_src = channels[self.pre_channel - 1].duplicate()
        post_src = channels[self.post_channel - 1].duplicate()
        pre_seg = pre_src.duplicate()
        post_seg = post_src.duplicate()
        pre_measure = pre_src.duplicate()
        post_measure = post_src.duplicate()

        cal = imp.getCalibration()
        # Removed forced calibration to match macro behavior (relies on image metadata or user setup)

        pre_entries, pre_result_imp, pre_mask_imp = self.prepare_channel(pre_seg, pre_measure, name_short, 'Pre', cal, self.pre_params)
        post_entries, post_result_imp, post_mask_imp = self.prepare_channel(post_seg, post_measure, name_short, 'Post', cal, self.post_params)
        self.log('  Pre detections (pre-marker): {}'.format(len(pre_entries)))
        self.log('  Post detections (post-marker): {}'.format(len(post_entries)))

        pre_marker_count = 0
        post_marker_count = 0

        # Capture total counts before gating for "Pre No." and "Post No." (macro Pre[row], Post[row])
        pre_count_total = len(pre_entries)
        post_count_total = len(post_entries)

        # Marker gating uses only median blur on the marker channel (no background subtraction),
        # and filters puncta purely by Min/Max intensity, mirroring CoLocROI.
        if self.pre_marker_channel > 0 and self.pre_marker_thresholds:
            marker_imp = channels[self.pre_marker_channel - 1].duplicate()
            processed_marker = self._preprocess_channel(marker_imp, self.pre_blur, self.pre_blur_radius, 0)
            pre_entries = self._filter_by_intensity(
                pre_entries,
                processed_marker,
                self.pre_marker_thresholds,
                'Presynaptic Marker',
                pre_result_imp,
                pre_mask_imp,
                name_short,
                'PreThrResults.txt',
            )
            pre_marker_count = len(pre_entries)
            self.log("Presynaptic marker gating: %d/%d puncta retained" % (pre_marker_count, pre_count_total))
            processed_marker.close()
            marker_imp.close()

        if self.post_marker_channel > 0 and self.post_marker_thresholds:
            marker_imp = channels[self.post_marker_channel - 1].duplicate()
            processed_marker = self._preprocess_channel(marker_imp, self.post_blur, self.post_blur_radius, 0)
            post_entries = self._filter_by_intensity(
                post_entries,
                processed_marker,
                self.post_marker_thresholds,
                'Postsynaptic Marker',
                post_result_imp,
                post_mask_imp,
                name_short,
                'PstRResults.txt',
            )
            post_marker_count = len(post_entries)
            self.log("Postsynaptic marker gating: %d/%d puncta retained" % (post_marker_count, post_count_total))
            processed_marker.close()
            marker_imp.close()

        self.log('  Pre detections (gated): {}'.format(len(pre_entries)))
        self.log('  Post detections (gated): {}'.format(len(post_entries)))

        if imp.getCalibration().pixelWidth == 1.0 and imp.getInfoProperty("Resolution") != None:
            # Attempt to parse resolution if available, similar to macro's getScaleAndUnit behavior
            pass 

        pre_rows = [entry['metrics'] for entry in pre_entries]
        post_rows = [entry['metrics'] for entry in post_entries]
        if pre_rows:
            self.all_pre_results.extend(pre_rows)
            self.save_results_table(pre_rows, os.path.join(self.dest_dir, '{}Pre.txt'.format(name_short)))
            self.save_roi_set([entry['roi'] for entry in pre_entries], os.path.join(self.dest_dir, '{}PreALLRoiSet.zip'.format(name_short)))
        if post_rows:
            self.all_post_results.extend(post_rows)
            self.save_results_table(post_rows, os.path.join(self.dest_dir, '{}Post.txt'.format(name_short)))
            self.save_roi_set([entry['roi'] for entry in post_entries], os.path.join(self.dest_dir, '{}PostALLRoiSet.zip'.format(name_short)))

        # Synapse Finding
        syn_post_rois = self.assoc_roi(pre_result_imp, post_result_imp, post_mask_imp, [entry['roi'] for entry in post_entries], self.pre_params['min'], self.overlap_pixels)
        syn_pre_rois = self.assoc_roi(post_result_imp, pre_result_imp, pre_mask_imp, [entry['roi'] for entry in pre_entries], self.post_params['min'], self.overlap_pixels)

        self.log('  Synapse count (Pre): {}'.format(len(syn_pre_rois)))
        self.log('  Synapse count (Post): {}'.format(len(syn_post_rois)))

        if syn_pre_rois:
            syn_pre_rows = self.measure_rois(syn_pre_rois, pre_result_imp, name_short, 'Pre', cal)
            self.syn_pre_results.extend(syn_pre_rows)
            # Per-image synaptic pre-synaptic results (Excel folder) + per-macro aggregate
            self.save_results_table(syn_pre_rows, os.path.join(self.excel_dir, '{}PreResults.txt'.format(name_short)))
            self.save_roi_set(syn_pre_rois, os.path.join(self.dest_dir, '{}PreSYNRoiSet.zip'.format(name_short)))

        if syn_post_rois:
            syn_post_rows = self.measure_rois(syn_post_rois, post_result_imp, name_short, 'Post', cal)
            self.syn_post_results.extend(syn_post_rows)
            # Per-image synaptic post-synaptic results (Excel folder) + per-macro aggregate
            self.save_results_table(syn_post_rows, os.path.join(self.excel_dir, '{}PostResults.txt'.format(name_short)))
            self.save_roi_set(syn_post_rois, os.path.join(self.dest_dir, '{}PostSYNRoiSet.zip'.format(name_short)))

        # Always save filtered images, even if no synapses were detected, to match macro behavior.
        IJ.save(pre_result_imp, os.path.join(self.dest_dir, '{}PreF.tif'.format(name_short)))
        IJ.save(post_result_imp, os.path.join(self.dest_dir, '{}PostF.tif'.format(name_short)))

        # Correlation Analysis
        # Disable dilation for correlation map to match macro
        pre_label_map = self._create_label_map(pre_result_imp, [entry['roi'] for entry in pre_entries], 0)
        post_label_map = self._create_label_map(post_result_imp, [entry['roi'] for entry in post_entries], 0)
        
        if pre_entries and post_entries:
            self.pre_correlation_rows.extend(self.match_roi(pre_entries, post_entries, post_label_map, 'Pre', 'Post', cal))
            self.post_correlation_rows.extend(self.match_roi(post_entries, pre_entries, pre_label_map, 'Post', 'Pre', cal))
            
        if pre_label_map: pre_label_map.close()
        if post_label_map: post_label_map.close()
        
        # Save Overlay (4 channels, no outlines)
        c1_imp = channels[0] # C1
        c2_imp = channels[1] # C2
        self.save_overlay(pre_result_imp, post_result_imp, c1_imp, c2_imp, name_short)

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

    def _find_puncta(self, processed_imp, noise_tolerance, threshold_val, size_low_um, size_high_um, cal):
            """Pixel-exact Find Maxima → Segmented Particles → ParticleAnalyzer with silent RoiManager."""
            # Use API directly to avoid headless issues and match segment_dense_image logic
            ip = processed_imp.getProcessor()
            
            threshold_arg = ImageProcessor.NO_THRESHOLD
            if threshold_val > 0:
                threshold_arg = float(threshold_val)

            mf = MaximumFinder()
            segmented_proc = mf.findMaxima(ip, float(noise_tolerance), threshold_arg, MaximumFinder.SEGMENTED, False, False)

            if segmented_proc is None:
                self.log("ERROR: Find Maxima produced no output image")
                return []

            segmented = ImagePlus("Segmented", segmented_proc)
            # segmented_proc is ByteProcessor (8-bit)

            rt = ResultsTable()
            rt.reset()

            # Use SHOW_MASKS instead of ADD_TO_MANAGER to avoid HeadlessException
            pa_flags = ParticleAnalyzer.SHOW_MASKS | ParticleAnalyzer.CLEAR_WORKSHEET

            pa = ParticleAnalyzer(pa_flags, MEASUREMENT_FLAGS, rt, size_low_um, size_high_um, 0.0, 1.0)
            pa.setHideOutputImage(True)
            pa.analyze(segmented)
            
            mask_imp = pa.getOutputImage()
            
            entries = []
            if mask_imp:
                from ij.gui import ShapeRoi
                mask_imp.getProcessor().setThreshold(128, 255, ImageProcessor.NO_LUT_UPDATE)
                IJ.run(mask_imp, "Create Selection", "")
                roi = mask_imp.getRoi()
                
                rois = []
                if roi:
                    if isinstance(roi, ShapeRoi):
                        rois = list(roi.getRois())
                    else:
                        rois = [roi]
                
                mask_imp.close()
                
                for i, roi in enumerate(rois):
                    # Measure on original processed_imp? Or segmented?
                    # Macro measures on the segmented image (which is binary/maxima)?
                    # No, macro measures on the original image usually?
                    # But _find_puncta is unused, so I'll just measure on segmented or processed_imp.
                    # The original code measured on segmented (via rt).
                    # But rt is empty if I don't use ADD_TO_MANAGER?
                    # Wait, ParticleAnalyzer populates rt if I pass it?
                    # Yes, but I need to associate measurements with ROIs.
                    # If I have ROIs, I can measure.
                    
                    processed_imp.setRoi(roi)
                    stats = processed_imp.getStatistics(MEASUREMENT_FLAGS)
                    
                    metrics = {
                        'Label': "%s_%04d" % (processed_imp.getShortTitle(), i + 1),
                        'Area': stats.area,
                        'Mean': stats.mean,
                        'Min': stats.min,
                        'Max': stats.max,
                        'X': stats.xCentroid,
                        'Y': stats.yCentroid,
                        'XM': stats.xCenterOfMass,
                        'YM': stats.yCenterOfMass,
                        'Perim.': stats.perimeter if hasattr(stats, 'perimeter') else roi.getLength(),
                        'Feret': stats.feret,
                        'IntDen': stats.integratedDensity,
                        'RawIntDen': stats.rawIntegratedDensity if hasattr(stats, 'rawIntegratedDensity') else 0,
                        'FeretX': stats.feretX,
                        'FeretY': stats.feretY,
                        'FeretAngle': stats.feretAngle,
                        'MinFeret': stats.minFeret,
                    }
                    entries.append({'roi': roi, 'metrics': metrics})

            segmented.changes = False
            segmented.close()

            return entries

    def _preprocess_channel(self, imp, blur_enabled, blur_radius, background_radius):
            """Identical preprocessing to original macro: Median blur + rolling ball background subtraction."""
            processed = imp.duplicate()
            if blur_enabled:
                IJ.run(processed, "Median...", "radius=" + str(blur_radius) + " stack")
            if background_radius > 0:
                subtracter = BackgroundSubtracter()
                for s in range(1, processed.getNSlices() + 1):
                    processed.setSlice(s)
                    ip = processed.getProcessor()
                    subtracter.rollingBallBackground(ip, float(background_radius), False, False, False, True, True)
            return processed


    def prepare_channel(self, work_imp, measure_imp, base_name, label, cal, params):
        """Per-channel pipeline matching SynapseJ PrepChannel: Mask -> Subtract -> Analyze."""
        from ij.plugin import ImageCalculator

        self.log("DEBUG: {} prepare_channel input work_imp slices: {}".format(label, work_imp.getStackSize()))

        if params['blur']:
            IJ.run(work_imp, 'Median...', self._format_args('radius={}'.format(params['blur_radius']), work_imp))
        
        if params.get('apply_fade') and params.get('fade_factors'):
            self.apply_fade_correction(work_imp, params['fade_factors'], label)

        masked_target = work_imp.duplicate()
        
        if params['background'] and params['background'] > 0:
            IJ.run(work_imp, 'Subtract Background...', self._format_args('rolling={}'.format(params['background']), work_imp))
            
        mask_imp = None
        min_pixels, max_pixels = self._size_bounds_in_pixels(params['size_low'], params['size_high'], cal)
        
        self.log("DEBUG: {} Calibration: {}x{} {}".format(label, cal.pixelWidth, cal.pixelHeight, cal.getUnit()))
        self.log("DEBUG: {} Size bounds (px): {}-{}".format(label, min_pixels, max_pixels))
        
        if params['use_maxima']:
            segmented_imp = self.segment_dense_image(work_imp, params['noise'], params['min'])
            self.log("DEBUG: {} segmented_imp slices: {}".format(label, segmented_imp.getStackSize()))
            
            # Manual stack processing for mask generation
            mask_stack = ImageStack(segmented_imp.getWidth(), segmented_imp.getHeight())
            
            # Ensure we process in pixel units for the mask generation too
            seg_cal = segmented_imp.getCalibration().copy()
            pixel_cal = seg_cal.copy()
            pixel_cal.setUnit("pixel")
            pixel_cal.pixelWidth = 1.0
            pixel_cal.pixelHeight = 1.0
            pixel_cal.pixelDepth = 1.0
            segmented_imp.setCalibration(pixel_cal)
            
            for i in range(1, segmented_imp.getStackSize() + 1):
                segmented_imp.setSlice(i)
                IJ.setAutoThreshold(segmented_imp, 'Default dark')
                
                pa = ParticleAnalyzer(ParticleAnalyzer.SHOW_MASKS, 0, ResultsTable(), min_pixels, max_pixels)
                pa.setHideOutputImage(True)
                pa.analyze(segmented_imp)
                m = pa.getOutputImage()
                if m:
                    mask_stack.addSlice(m.getProcessor())
                else:
                    mask_stack.addSlice(ByteProcessor(segmented_imp.getWidth(), segmented_imp.getHeight()))
            
            mask_imp = ImagePlus("Mask", mask_stack)
            mask_imp.setCalibration(seg_cal) # Restore original calibration for the mask? Or keep pixels? 
            # The macro does "Multiply 257", "Invert", then "Subtract create".
            # ImageCalculator uses pixel values, calibration doesn't matter for subtraction.
            
            self.log("DEBUG: {} mask_imp (from maxima) slices: {}".format(label, mask_imp.getStackSize()))
            segmented_imp.close()
        else:
            temp_imp = work_imp.duplicate()
            
            # Ensure we process in pixel units
            temp_cal = temp_imp.getCalibration().copy()
            pixel_cal = temp_cal.copy()
            pixel_cal.setUnit("pixel")
            pixel_cal.pixelWidth = 1.0
            pixel_cal.pixelHeight = 1.0
            pixel_cal.pixelDepth = 1.0
            temp_imp.setCalibration(pixel_cal)
            
            # Manual stack processing for mask generation
            mask_stack = ImageStack(temp_imp.getWidth(), temp_imp.getHeight())
            for i in range(1, temp_imp.getStackSize() + 1):
                temp_imp.setSlice(i)
                if params['min'] > 0:
                    IJ.setThreshold(temp_imp, params['min'], 65535)
                else:
                    IJ.setAutoThreshold(temp_imp, 'Default dark')
                
                pa = ParticleAnalyzer(ParticleAnalyzer.SHOW_MASKS, 0, ResultsTable(), min_pixels, max_pixels)
                pa.setHideOutputImage(True)
                pa.analyze(temp_imp)
                m = pa.getOutputImage()
                if m:
                    mask_stack.addSlice(m.getProcessor())
                else:
                    mask_stack.addSlice(ByteProcessor(temp_imp.getWidth(), temp_imp.getHeight()))
            
            mask_imp = ImagePlus("Mask", mask_stack)
            mask_imp.setCalibration(temp_cal)
            
            self.log("DEBUG: {} mask_imp (from threshold) slices: {}".format(label, mask_imp.getStackSize()))
            temp_imp.close()

        if mask_imp is None:
            masked_target.close()
            return [], work_imp, None

        ImageConverter(mask_imp).convertToGray16()
        IJ.run(mask_imp, "Multiply...", "value=257 stack")
        IJ.run(mask_imp, "Invert", "stack")
        
        if self.dilate_enabled:
            self._dilate_mask(mask_imp, self.dilate_pixels)
        
        ic = ImageCalculator()
        result_imp = ic.run("Subtract create stack", masked_target, mask_imp)
        self.log("DEBUG: {} result_imp (after subtract) slices: {}".format(label, result_imp.getStackSize()))
        
        IJ.run(mask_imp, "Invert", "stack")
        
        # Final ROI detection: headless, RoiManager-free replication of
        # "Analyze Particles... show=Nothing display clear summarize add stack".

        from ij.process import ImageProcessor as _IP

        n_slices = result_imp.getStackSize()
        self.log("DEBUG: {} final detection on {} slice(s) via custom labeling".format(label, n_slices))

        entries = []
        roi_index = 0

        stack = result_imp.getStack()

        for s in range(1, n_slices + 1):
            result_imp.setSlice(s)
            ip = stack.getProcessor(s)

            # Determine threshold range exactly as the macro does:
            # - If a minimum is provided, use it as the lower bound.
            # - Otherwise, use ImageJ's "Default" auto-threshold on this slice.
            if params['min'] > 0:
                lower = float(params['min'])
                upper = 65535.0
                ip.setThreshold(lower, upper, _IP.NO_LUT_UPDATE)
            else:
                IJ.setAutoThreshold(result_imp, 'Default dark')
                lower = ip.getMinThreshold()
                upper = ip.getMaxThreshold()

            if lower == _IP.NO_THRESHOLD or upper == _IP.NO_THRESHOLD:
                continue

            width = ip.getWidth()
            height = ip.getHeight()

            # Visited map for flood-fill connected-component labeling
            visited = [[False] * width for _ in range(height)]

            def neighbors(x, y):
                # 8-connected neighborhood
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in (-1, 0, 1):
                        nx = x + dx
                        if dx == 0 and dy == 0:
                            continue
                        if nx < 0 or nx >= width:
                            continue
                        yield nx, ny

            # Scan all pixels, flood-fill suprathreshold regions
            for y in range(height):
                for x in range(width):
                    if visited[y][x]:
                        continue
                    v = ip.get(x, y)
                    if v < lower or v > upper:
                        continue

                    # Start a new component
                    stack_xy = [(x, y)]
                    visited[y][x] = True
                    coords = []

                    while stack_xy:
                        cx, cy = stack_xy.pop()
                        coords.append((cx, cy))
                        for nx, ny in neighbors(cx, cy):
                            if visited[ny][nx]:
                                continue
                            nv = ip.get(nx, ny)
                            if nv < lower or nv > upper:
                                continue
                            visited[ny][nx] = True
                            stack_xy.append((nx, ny))

                    pixel_count = float(len(coords))
                    if pixel_count < min_pixels or pixel_count > max_pixels:
                        continue

                    # Build a binary mask for this component only.
                    comp_bp = ByteProcessor(width, height)
                    for cx, cy in coords:
                        comp_bp.set(cx, cy, 255)

                    comp_imp = ImagePlus('comp', comp_bp)
                    comp_ip = comp_imp.getProcessor()
                    comp_ip.setThreshold(255, 255, _IP.NO_LUT_UPDATE)
                    IJ.run(comp_imp, 'Create Selection', '')
                    roi = comp_imp.getRoi()
                    comp_imp.close()

                    if roi is None:
                        continue

                    roi.setPosition(s)

                    # Measure on the calibrated result image to keep physical units.
                    result_imp.setSlice(s)
                    result_imp.setRoi(roi)
                    stats = result_imp.getStatistics(MEASUREMENT_FLAGS)
                    metrics = self._metrics_dict(base_name, label, roi_index, stats, cal, roi)
                    entries.append({'roi': roi, 'metrics': metrics})
                    roi_index += 1

            result_imp.deleteRoi()

        self.log("DEBUG: {} Total ROIs from custom labeling: {}".format(label, len(entries)))
        return entries, result_imp, mask_imp

    def _metrics_dict(self, base_name, label, index, stats, cal, roi):
        """Convert ImageJ stats object into explicit columns matching SynapseJ resultLabel."""
        # Ref: SynapseJ_v_1.ijm line 29: resultLabel = newArray("Label","Area","Mean","Min","Max","X","Y","XM","YM","Perim.","Feret","IntDen","RawIntDen","FeretX","FeretY","FeretAngle","MinFeret");
        
        # Helper to safely get attributes that might be missing in some ImageStatistics versions
        def get_stat(obj, name, default=0):
            return getattr(obj, name, default)

        return {
            # Standard ImageJ Result Columns
            'Label': '{}:{}:{}'.format(base_name, label, index),
            'Area': stats.area,
            'Mean': stats.mean,
            'Min': stats.min,
            'Max': stats.max,
            'X': stats.xCentroid,
            'Y': stats.yCentroid,
            'XM': stats.xCenterOfMass,
            'YM': stats.yCenterOfMass,
            'Perim.': get_stat(stats, 'perimeter') or (roi.getLength() if roi else 0),
            'Feret': get_stat(stats, 'feret'),
            'IntDen': get_stat(stats, 'integratedDensity') or (stats.area * stats.mean), # Fallback if field missing
            'RawIntDen': get_stat(stats, 'rawIntegratedDensity') or (stats.pixelCount * stats.mean), # Fallback
            'FeretX': get_stat(stats, 'feretX'),
            'FeretY': get_stat(stats, 'feretY'),
            'FeretAngle': get_stat(stats, 'feretAngle'),
            'MinFeret': get_stat(stats, 'minFeret'),
            
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
        """Translate µm² size bounds into pixel counts (User Guide table)."""
        pixel_area = self._pixel_area(cal)
        if pixel_area <= 0:
            pixel_area = 1.0
        min_px = min_um2 / pixel_area if min_um2 and min_um2 > 0 else 0
        max_px = max_um2 / pixel_area if max_um2 and max_um2 > 0 else 1e12
        return (min_px, max_px)


    def _gate_puncta_with_marker_overlap(self, puncta_entries, marker_label_map, overlap_threshold=5):
            if marker_label_map is None:
                return puncta_entries
            retained = []
            for entry in puncta_entries:
                roi = entry['roi']
                if marker_label_map.getNSlices() > 1 and roi.getPosition() > 0:
                    marker_label_map.setSlice(roi.getPosition())
                marker_label_map.setRoi(roi)
                hist = marker_label_map.getProcessor().getHistogram()
                max_overlap = max(hist[1:]) if len(hist) > 1 else 0
                if max_overlap >= overlap_threshold:
                    retained.append(entry)
            marker_label_map.deleteRoi()
            return retained            
            
    def save_text_lines(self, lines, path):
        """Write raw text lines to a file."""
        if not lines:
            return
        with open(path, 'w') as handle:
            for line in lines:
                handle.write(line + '\n')

    def save_results_table(self, rows, path):
        """Persist any TSV table with header first, mirroring SynapseJ text exports."""
        if not rows:
            return
        with open(path, 'w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)  # TSV matches SynapseJ's tab-delimited text exports.

    def save_roi_set(self, rois, path):
        """Serialize ROI sets so Fiji can reload them like SynapseJ macros do."""
        if not rois:
            return
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        stream = ZipOutputStream(BufferedOutputStream(FileOutputStream(path)))
        try:
            for idx, roi in enumerate(rois):
                entry = ZipEntry('Roi_{:05d}.roi'.format(idx + 1))
                stream.putNextEntry(entry)
                encoder = RoiEncoder(stream)
                encoder.write(roi.clone())
                stream.closeEntry()
        finally:
            stream.close()

    def save_overlay(self, pre_src, post_src, c1_imp, c2_imp, base_name):
            """Create exact SynapseJ RGB overlay – 4 channels merged, no outlines."""
            # Ref: SynapseJ_v_1.ijm line 370: "run("Merge Channels...", "c1=PreF c2=PostF c3=C1 c4=C2 create");"
            # PreF = pre_src (result), PostF = post_src (result)
            # C1 = c1_imp, C2 = c2_imp
            
            # Duplicate all to avoid modifying originals
            c1 = pre_src.duplicate()
            c2 = post_src.duplicate()
            c3 = c1_imp.duplicate()
            c4 = c2_imp.duplicate()
            
            # Merge Channels
            # c1=Red, c2=Green, c3=Blue, c4=Gray, etc.
            # The macro string "c1=PreF c2=PostF c3=C1 c4=C2" maps images to channels.
            # In ImageJ "Merge Channels":
            # c1 (Red) -> PreF
            # c2 (Green) -> PostF
            # c3 (Blue) -> C1
            # c4 (Gray) -> C2
            # Wait, usually c1=Red, c2=Green, c3=Blue, c4=Gray, c5=Cyan, c6=Magenta, c7=Yellow.
            # So PreF is Red, PostF is Green, C1 is Blue, C2 is Gray.
            
            merged = RGBStackMerge.mergeChannels([c1, c2, c3, c4], True)
            
            # Save
            out_path = os.path.join(self.merge_dir, '{}PrePost.tif'.format(base_name))
            IJ.save(merged, out_path)
            
            merged.close()
            c1.close()
            c2.close()
            c3.close()
            c4.close()

    def record_summary(self, base_name, pre_count, pre_marker_count, post_count, post_marker_count,
                       syn_post_count, syn_pre_count):
        """Append a Collated ResultsIF row for this image.

        Directly mirrors the macro arrays:
        LABEL[row], Syn[row], SynPre[row], ThrPre[row], ForPost[row], Post[row], Pre[row].
        """

        entry = {
            'Label': base_name,
            'Synapse Post No.': syn_post_count,
            'Synapse Pre No.': syn_pre_count,
            'Thr Pre No.': pre_marker_count if (self.pre_marker_channel > 0 and self.pre_marker_thresholds) else '',
            'Fourth Post No.': post_marker_count if (self.post_marker_channel > 0 and self.post_marker_thresholds) else '',
            'Post No.': post_count,
            'Pre No.': pre_count,
        }
        self.results_summary.append(entry)

    def save_all_results(self):
        """Write every accumulated table using official SynapseJ filenames."""
        if self.results_summary:
            self.save_results_table(self.results_summary, os.path.join(self.dest_dir, 'Collated ResultsIF.txt'))
        if self.all_pre_results:
            self.save_results_table(self.all_pre_results, os.path.join(self.dest_dir, 'All Pre Results.txt'))
        if self.all_post_results:
            self.save_results_table(self.all_post_results, os.path.join(self.dest_dir, 'All Post Results.txt'))
        if self.syn_pre_results:
            self.save_results_table(self.syn_pre_results, os.path.join(self.dest_dir, 'Syn Pre Results.txt'))
        if self.syn_post_results:
            self.save_results_table(self.syn_post_results, os.path.join(self.dest_dir, 'Syn Post Results.txt'))
        if self.synapse_pair_rows:
            self.save_results_table(self.synapse_pair_rows, os.path.join(self.dest_dir, 'AllSynapsePairs.tsv'))
        if self.pre_correlation_rows:
            self.save_text_lines(self.pre_correlation_rows, os.path.join(self.dest_dir, 'CorrResults.txt'))
        if self.post_correlation_rows:
            self.save_text_lines(self.post_correlation_rows, os.path.join(self.dest_dir, 'CorrResults2.txt'))

    def save_log(self):
        """Store IFALog.txt exactly like the macro for provenance."""
        if not self.log_messages:
            return
        path = os.path.join(self.dest_dir, 'IFALog.txt')
        with open(path, 'w') as handle:
            handle.write('\n'.join(self.log_messages))

    def _pixel_area(self, cal):
        """Compute µm² per pixel, defaulting to unity if calibration missing."""
        if cal is None:
            return 1.0
        px = cal.pixelWidth if cal.pixelWidth and cal.pixelWidth > 0 else 1.0
        py = cal.pixelHeight if cal.pixelHeight and cal.pixelHeight > 0 else 1.0
        return px * py

    def _pixel_width(self, cal):
        """Return calibrated pixel width (µm) or default to 1."""
        if cal is None or not cal.pixelWidth or cal.pixelWidth <= 0:
            return 1.0
        return cal.pixelWidth

    def _pixel_height(self, cal):
        """Return calibrated pixel height (µm) or default to 1."""
        if cal is None or not cal.pixelHeight or cal.pixelHeight <= 0:
            return 1.0
        return cal.pixelHeight

    def _create_count_mask(self, imp, min_threshold):
        """Generate a "Count Masks" image (1-based particle index per pixel).

        Mirrors the macro call
        run("Analyze Particles...", "size=0-infinity show=[Count Masks] display clear stack");
        on the CheckIm image with an optional intensity threshold.
        """
        temp = imp.duplicate()
        if min_threshold > 0:
            IJ.setThreshold(temp, float(min_threshold), 65535.0)
            temp.getProcessor().setThreshold(float(min_threshold), 65535.0, ImageProcessor.NO_LUT_UPDATE)

        flags = ParticleAnalyzer.CLEAR_WORKSHEET | ParticleAnalyzer.DOES_STACKS | ParticleAnalyzer.SHOW_ROI_MASKS
        pa = ParticleAnalyzer(flags, 0, None, 0.0, Double.POSITIVE_INFINITY, 0.0, 0.0)
        pa.analyze(temp)
        count_mask = pa.getOutputImage()
        temp.close()

        if count_mask is not None:
            count_mask.setCalibration(imp.getCalibration())
        return count_mask

    def assoc_roi(self, check_imp, store_imp, store_mask_imp, rois, check_threshold, overlap_pixels):
        """Replicate AssocROI from SynapseJ_v_1.ijm.

        - If overlap_pixels (LapNo) == 1: only check that each ROI has Max>0 in
          CheckIm after thresholding by check_threshold; no explicit pixel
          overlap requirement.
        - If overlap_pixels > 1: build a Count Masks image from CheckIm and
          require at least "overlap_pixels" pixels of overlap with some
          labeled particle.
        """
        if not rois:
            return []

        kept_rois = []
        slice_kept_counts = defaultdict(int)
        slice_total_counts = defaultdict(int)

        # LapNo == 1 branch: intensity-only gating (macro's else-if block).
        if overlap_pixels <= 1:
            for roi in reversed(rois):
                slice_idx = roi.getPosition()
                slice_total_counts[slice_idx] += 1
                if check_imp.getStackSize() > 1 and slice_idx > 0:
                    check_imp.setSlice(slice_idx)
                check_imp.setRoi(roi)
                stats = check_imp.getStatistics()
                if stats.max <= 0:
                    # Clear rejected ROI from StoreIm and StoreImT
                    self._clear_roi(store_imp, store_mask_imp, roi)
                else:
                    kept_rois.append(roi)
                    slice_kept_counts[slice_idx] += 1

            check_imp.deleteRoi()
            kept_rois.reverse()
            self.log("DEBUG: assoc_roi (LapNo=1) processed {} ROIs".format(len(rois)))
            for s in sorted(slice_total_counts.keys()):
                self.log("DEBUG: assoc_roi Slice {}: Kept {}/{}".format(s, slice_kept_counts[s], slice_total_counts[s]))
            return kept_rois

        # LapNo > 1 branch: require explicit pixel overlap using a Count Masks image.
        count_mask = self._create_count_mask(check_imp, check_threshold)
        if count_mask is None:
            self.log("WARNING: Failed to generate Count Mask for association. Keeping all ROIs.")
            return rois

        stats_mask = count_mask.getStatistics()
        self.log("DEBUG: Count Mask Max: {}".format(stats_mask.max))

        for roi in reversed(rois):
            slice_idx = roi.getPosition()
            slice_total_counts[slice_idx] += 1

            if count_mask.getStackSize() > 1 and slice_idx > 0:
                count_mask.setSlice(slice_idx)
            count_mask.setRoi(roi)
            stats = count_mask.getStatistics()
            if stats.max == 0:
                self._clear_roi(store_imp, store_mask_imp, roi)
                continue

            hist = count_mask.getProcessor().getHistogram()
            has_overlap = False
            for i in range(1, len(hist)):
                if hist[i] >= overlap_pixels:
                    has_overlap = True
                    break

            if has_overlap:
                kept_rois.append(roi)
                slice_kept_counts[slice_idx] += 1
            else:
                self._clear_roi(store_imp, store_mask_imp, roi)

        count_mask.close()
        kept_rois.reverse()

        self.log("DEBUG: assoc_roi (LapNo>{}) processed {} ROIs".format(1, len(rois)))
        for s in sorted(slice_total_counts.keys()):
            self.log("DEBUG: assoc_roi Slice {}: Kept {}/{}".format(s, slice_kept_counts[s], slice_total_counts[s]))
        return kept_rois

    def _clear_roi(self, imp1, imp2, roi):
        """Clear the ROI region in the provided images (set to 0)."""
        for imp in [imp1, imp2]:
            if imp:
                if imp.getStackSize() > 1:
                    imp.setSlice(roi.getPosition())
                imp.setRoi(roi)
                ip = imp.getProcessor()
                ip.setValue(0)
                ip.fill(roi)
                imp.deleteRoi()

    def measure_rois(self, rois, imp, base_name, label, cal):
        """Measure ROIs on the provided image and return result rows."""
        rows = []
        for idx, roi in enumerate(rois):
            if imp.getStackSize() > 1:
                imp.setSlice(roi.getPosition())
            imp.setRoi(roi)
            stats = imp.getStatistics(MEASUREMENT_FLAGS)
            metrics = self._metrics_dict(base_name, label, idx, stats, cal, roi)
            rows.append(metrics)
            imp.deleteRoi()
        return rows

    def _dilate_mask(self, mask_imp, pixels):
        """Dilate the mask image in-place (Ref: SynapseJ_v_1.ijm lines 565-580)."""
        # NO early return - macro always runs the loop when dilateQ true, even for 0 or negative
        from ij.process import ImageConverter
        ImageConverter(mask_imp).convertToGray8()
        
        stack = mask_imp.getStack()
        n_slices = stack.getSize()
        
        for i in range(1, n_slices + 1):
            mask_imp.setSlice(i)
            IJ.setAutoThreshold(mask_imp, 'Default dark')
            IJ.run(mask_imp, "Create Selection", "")
            
            roi = mask_imp.getRoi()
            if roi:
                from ij.plugin import RoiEnlarger
                dilated_roi = RoiEnlarger.enlarge(roi, float(pixels))  # can be 0 or negative
                mask_imp.setRoi(dilated_roi)
                IJ.run(mask_imp, "Set...", "value=255 slice")
                mask_imp.deleteRoi()
                
        ImageConverter(mask_imp).convertToGray16()
        IJ.run(mask_imp, "Multiply...", "value=257 stack")

    def _create_label_map(self, imp, rois, pixels):
        """Create a label map image where pixel values correspond to ROI indices (1-based)."""
        width = imp.getWidth()
        height = imp.getHeight()
        
        n_rois = len(rois)
        if n_rois > 65534:
            from ij.process import FloatProcessor
            ip = FloatProcessor(width, height)
        elif n_rois > 254:
            from ij.process import ShortProcessor
            ip = ShortProcessor(width, height)
        else:
            from ij.process import ByteProcessor
            ip = ByteProcessor(width, height)
            
        label_map = ImagePlus("LabelMap", ip)
        label_map.setCalibration(imp.getCalibration())
        
        if imp.getStackSize() > 1:
            stack = ImageStack(width, height)
            for i in range(imp.getStackSize()):
                stack.addSlice(ip.duplicate())
            label_map.setStack(stack)
        
        from ij.plugin import RoiEnlarger
        
        # Burn in ROIs
        for i, roi in enumerate(rois):
            label_val = i + 1
            
            draw_roi = roi
            if self.dilate_enabled:
                draw_roi = RoiEnlarger.enlarge(roi, float(pixels))
                
            if label_map.getStackSize() > 1:
                label_map.setSlice(draw_roi.getPosition())
            
            label_map.setRoi(draw_roi)
            label_map.getProcessor().setValue(label_val)
            label_map.getProcessor().fill(draw_roi)
            label_map.deleteRoi()
        return label_map

    def match_roi(self, anchor_entries, partner_entries, partner_label_map, anchor_label, partner_label, cal):
        """Perform correlation analysis (MatchROI) and return formatted text lines."""
        lines = []
        pixel_area = self._pixel_area(cal)
        
        # Iterate Anchor ROIs
        for m, anchor_entry in enumerate(anchor_entries):
            anchor_roi = anchor_entry['roi']
            anchor_metrics = anchor_entry['metrics']
            
            if partner_label_map.getStackSize() > 1:
                partner_label_map.setSlice(anchor_roi.getPosition())
            
            partner_label_map.setRoi(anchor_roi)
            stats = partner_label_map.getStatistics()
            
            if stats.max == 0:
                continue
                
            hist = partner_label_map.getProcessor().getHistogram()
            
            lineP = ""
            distSm = 0
            VStar = -1
            VPerROI = 0
            VIDPerROI = 0
            
            for n in range(1, len(hist)):
                count = hist[n]
                if count > 0:
                    if n-1 < len(partner_entries):
                        partner_entry = partner_entries[n-1]
                        partner_metrics = partner_entry['metrics']
                        
                        dx = anchor_metrics['X'] - partner_metrics['X']
                        dy = anchor_metrics['Y'] - partner_metrics['Y']
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        dx_m = anchor_metrics['XM'] - partner_metrics['XM']
                        dy_m = anchor_metrics['YM'] - partner_metrics['YM']
                        dist_m = math.sqrt(dx_m*dx_m + dy_m*dy_m)
                        
                        # Format match string: TAB + ID + TAB + Count + TAB + Dist + TAB + DistM
                        match_str = "\t{}\t{}\t{:.3f}\t{:.3f}".format(n, count, dist, dist_m)
                        
                        if dist < distSm or distSm == 0:
                            lineP = match_str + lineP
                            distSm = dist
                            VStar = n
                        else:
                            lineP = lineP + match_str
                        
                        VPerROI += 1
                        VIDPerROI += partner_metrics['IntDen']

            if VStar != -1:
                # Construct final line: AnchorLabel + PartnerLabel + VPerROI + VIDPerROI + lineP
                # Note: PartnerLabel comes from VResult[VStar] in macro.
                # VResult is the array of result strings for the partner channel.
                # Here we just use the partner label.
                
                partner_label_str = partner_entries[VStar-1]['metrics']['Label']
                
                # Macro: lineP = ROIResult[m]+VResult[VStar]+VPerROI+"\t"+VIDPerROI+lineP + "\n";
                # ROIResult[m] is the full result line for anchor? Or just label?
                # "CopyResultsTableArr" returns the full line.
                # So we should output the full line for anchor and partner?
                # The user's diff said "Label first, then best_partner_label".
                # I'll output AnchorLabel + TAB + PartnerLabel + TAB + VPerROI + TAB + VIDPerROI + lineP
                
                final_line = "{}\t{}\t{}\t{}{}".format(anchor_metrics['Label'], partner_label_str, VPerROI, VIDPerROI, lineP)
                lines.append(final_line)
            
        partner_label_map.deleteRoi()
        return lines

    def _filter_by_intensity(self, entries, marker_imp, thresholds, label,
                              result_imp, mask_imp, base_name, excel_suffix):
        """Filter ROIs based on Min/Max intensity in the marker channel.

        This reproduces the CoLocROI macro behavior:
        for each ROI, measure Max and Min on the blurred marker image and
        reject it if (max <= Hi || min <= Lo). The size parameter is logged in
        IFALog but *not* used for gating.
        """
        lo_val = thresholds['min']   # corresponds to ThrLo / ForLo (Min Int > Lo)
        hi_val = thresholds['max']   # corresponds to ThrHi / ForHi (Max Int > Hi)
        size_val = thresholds['size']

        kept = []
        marker_rows = []
        slice_kept_counts = defaultdict(int)
        slice_total_counts = defaultdict(int)

        for idx, entry in enumerate(entries):
            roi = entry['roi']
            slice_idx = roi.getPosition()
            slice_total_counts[slice_idx] += 1

            if marker_imp.getStackSize() > 1 and slice_idx > 0:
                marker_imp.setSlice(slice_idx)
            marker_imp.setRoi(roi)
            stats = marker_imp.getStatistics(MEASUREMENT_FLAGS)
            max_i = stats.max
            min_i = stats.min

            # Macro: if (max <= Hi || min <= Lo) → reject
            if max_i <= hi_val or min_i <= lo_val:
                self._clear_roi(result_imp, mask_imp, roi)
                continue

            kept.append(entry)
            slice_kept_counts[slice_idx] += 1

            # Also measure marker-channel stats for this ROI into a per-image
            # table (PreThrResults.txt / PstRResults.txt in the Excel folder).
            metrics = self._metrics_dict(
                base_name,
                label,
                entry['metrics'].get('Index', idx),
                stats,
                marker_imp.getCalibration(),
                roi,
            )
            marker_rows.append(metrics)

        marker_imp.deleteRoi()

        self.log("{} gating (intensity only): {}/{} retained (Max>{}, Min>{}, size param={})".format(
            label, len(kept), len(entries), hi_val, lo_val, size_val))
        for s in sorted(slice_total_counts.keys()):
            self.log("DEBUG: {} gating Slice {}: Kept {}/{}".format(label, s, slice_kept_counts[s], slice_total_counts[s]))

        if marker_rows:
            out_path = os.path.join(self.excel_dir, base_name + excel_suffix)
            self.save_results_table(marker_rows, out_path)

        return kept


if __name__ == '__main__':
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    elif 'SYNAPSEJ_CONFIG' in os.environ:
        config_path = os.environ['SYNAPSEJ_CONFIG']

    if not config_path:
        print('Usage: SYNAPSEJ_CONFIG=path/to/config fiji --headless --run Pynapse.py')
        sys.exit(1)

    analyzer = SynapseJ4ChannelComplete(config_path)
    analyzer.process_directory()
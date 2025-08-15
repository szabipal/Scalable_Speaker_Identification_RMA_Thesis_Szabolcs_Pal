"""
simple preprocessing → feature-extraction driver.

what it does
- splits raw audio into small wave chunks via scripts/preprocessing/preprocess.py
- extracts features depending on `model_type`:
  * phonetic: mfcc complementary + mel-band complementary (low/mid/high)
  * spectral: mfcc + mel spectrogram features
- organizes outputs under: <output_base_dir>/{wave_chunks, features/<model_type>}

notes
- this file shells out to existing scripts with subprocess; it doesn’t re-implement dsp.
- it only ensures folders exist and passes through the right paths/flags.
"""

import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class PreprocessingAndFeatureExtractor:
    """
    minimal pipeline wrapper around the repo’s preprocessing/feature scripts.

    config keys
    - raw_audio_dir: folder with source audio (e.g., libriSpeech subset)
    - output_base_dir: root where wave_chunks/ and features/ are written
    - model_type: 'phonetic' or 'spectral' (chooses which feature scripts to run)
    """
    def __init__(self, config):
        """
        init paths and create output directories.

        Args:
            config (dict): {
                'raw_audio_dir': path to raw audio files,
                'output_base_dir': base path for outputs,
                'model_type': 'phonetic' | 'spectral'
            }
        """
        self.raw_audio_dir = Path(config["raw_audio_dir"])
        self.output_base_dir = Path(config["output_base_dir"])
        self.model_type = config["model_type"]

        self.wave_chunks_dir = self.output_base_dir / "wave_chunks"
        self.feature_output_dir = self.output_base_dir / "features" / self.model_type

        self._ensure_directories()

    def _ensure_directories(self):
        """create wave chunk + feature output folders if missing."""
        self.wave_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.feature_output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_audio(self):
        """
        call the repo’s chunking script.

        runs:
          python scripts/preprocessing/preprocess.py --input <raw> --output <wave_chunks_dir>
        """
        print(" Starting audio preprocessing...")

        subprocess.run([
            "python", "scripts/preprocessing/preprocess.py",
            "--input", str(self.raw_audio_dir),
            "--output", str(self.wave_chunks_dir)
        ], check=True)

        print(f" Saved wave chunks to: {self.wave_chunks_dir}")


    def extract_features(self):
        """
        dispatch to phonetic or spectral extractors.

        phonetic → complementary features (mfcc + mel bands)
        spectral → plain mfcc + mel
        """
        if self.model_type == "phonetic":
            self._extract_phonetic_features()
        elif self.model_type == "spectral":
            self._extract_spectral_features()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _extract_phonetic_features(self):
        """
        run complementary feature scripts.

        outputs under: <output_base_dir>/features/phonetic/{mfcc,low,mid,high}
        """
        print(" Extracting phonetic features...")

        # mfcc complementary
        subprocess.run([
            "python", "scripts/preprocessing/process_mfcc_compl_features.py",
            "--input", str(self.wave_chunks_dir),
            "--output", str(self.feature_output_dir / "mfcc")
        ], check=True)

        # mel complementary (low, mid, high)
        subprocess.run([
            "python", "scripts/preprocessing/process_mel_band_compl_features.py",
            "--input", str(self.wave_chunks_dir),
            "--output", str(self.feature_output_dir)
        ], check=True)

    def _extract_spectral_features(self):
        """
        run standard spectral feature scripts.

        outputs under: <output_base_dir>/features/spectral/{mfcc,mel}
        """
        print(" Extracting spectral features...")

        subprocess.run([
            "python", "scripts/preprocessing/process_mfcc_features.py",
            "--input", str(self.wave_chunks_dir),
            "--output", str(self.feature_output_dir / "mfcc")
        ], check=True)

        subprocess.run([
            "python", "scripts/preprocessing/process_mel_features.py",
            "--input", str(self.wave_chunks_dir),
            "--output", str(self.feature_output_dir)
        ], check=True)

    def run_pipeline(self):
        """one-shot convenience: preprocess then extract."""
        self.preprocess_audio()
        self.extract_features()
        print("✅ Pipeline complete.")


if __name__ == "__main__":
    from pathlib import Path

    # example config; adjust to your local paths
    config = {
        "raw_audio_dir": "data/dev-clean",          
        "output_base_dir": "data/processed_dev_2",  
        "model_type": "phonetic"                   
    }

    # pipeline = PreprocessingAndFeatureExtractor(config)
    # pipeline.run_pipeline()


    # config = {
    #     "raw_audio_dir": "data/dev-clean",
    #     "output_base_dir": "data/processed_dev_2",
    #     "model_type": "spectral"
    # }

    pipeline = PreprocessingAndFeatureExtractor(config)
    pipeline.run_pipeline()

    # # repeated example configs left as-is
    # config = {
    #     "raw_audio_dir": "data/dev-clean",          
    #     "output_base_dir": "data/processed_dev_2",  
    #     "model_type": "phonetic"                    
    # }

    # config = {
    #     "raw_audio_dir": "data/dev-clean",
    #     "output_base_dir": "data/processed_dev_2",
    #     "model_type": "phonetic"
    # }

    # config = {
    #     "raw_audio_dir": "data/dev-clean",
    #     "output_base_dir": "data/processed_dev_2",
    #     "model_type": "phonetic"
    # }

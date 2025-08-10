# To install mfa, need to install conda first. Then run:
# conda create -n aligner python=3.8
# conda activate aligner
# conda install -c conda-forge montreal-forced-aligner
#
# then download the mfa dictionary and model files from the MFA website:
# https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_uk_mfa-v2.0.0

import subprocess
import os
import shutil

def is_mfa_installed():
    """Check if MFA is installed and available in PATH."""
    return shutil.which("mfa") is not None

def run_mfa_alignment(data_dir, dict_path, model_path, output_dir):
    """
    Run Montreal Forced Aligner (MFA) via subprocess.
    """
    if not is_mfa_installed():
        raise EnvironmentError(
            "Montreal Forced Aligner (MFA) is not installed or not in your PATH. "
            "Please activate your conda environment with MFA installed."
        )
    cmd = [
        "mfa", "align",
        data_dir,
        dict_path,
        model_path,
        output_dir,
        "--clean"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MFA failed: {result.stderr}")
    # Return the path to the first TextGrid file in output_dir
    textgrids = [f for f in os.listdir(output_dir) if f.endswith(".TextGrid")]
    if not textgrids:
        raise FileNotFoundError("No TextGrid file found in MFA output directory.")
    return os.path.join(output_dir, textgrids[0])
"""
Generate Transcript & Minutes of Meeting from Audio File using WhisperX
"""
import sys
import types
import torchaudio

# Patch torchaudio.backend
_backend = types.ModuleType("torchaudio.backend")
sys.modules["torchaudio.backend"] = _backend
torchaudio.backend = _backend

# Create a dummy AudioMetaData since its location changed in torchaudio 2.11
class AudioMetaData:
    def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding

# Patch torchaudio.backend.common
_common = types.ModuleType("torchaudio.backend.common")
_common.AudioMetaData = AudioMetaData
sys.modules["torchaudio.backend.common"] = _common
_backend.common = _common

import os
import gc
import argparse
from datetime import timedelta
from dotenv import load_dotenv
from crewai.tools import tool
import numpy as np
from langchain_groq import ChatGroq
load_dotenv()

# ──────────────────────────────────────────────
# Compatibility patches for torchaudio 2.x
# ──────────────────────────────────────────────
import torchaudio
import soundfile as sf
from dataclasses import dataclass as _dataclass

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

if not hasattr(torchaudio, "AudioMetaData"):
    @_dataclass
    class _AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""
    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, "info"):
    def _torchaudio_info(filepath, **kwargs):
        info = sf.info(str(filepath))
        return torchaudio.AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )
    torchaudio.info = _torchaudio_info

# ──────────────────────────────────────────────
# Compatibility patch for huggingface_hub 1.x
# ──────────────────────────────────────────────
import huggingface_hub
if not hasattr(huggingface_hub, "_patched_for_use_auth_token"):
    original_hf_hub_download = huggingface_hub.hf_hub_download
    def patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = patched_hf_hub_download
    
    if hasattr(huggingface_hub, "snapshot_download"):
        original_snapshot_download = huggingface_hub.snapshot_download
        def patched_snapshot_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original_snapshot_download(*args, **kwargs)
        huggingface_hub.snapshot_download = patched_snapshot_download
    huggingface_hub._patched_for_use_auth_token = True

for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("pyannote."):
        mod = sys.modules[mod_name]
        if hasattr(mod, "hf_hub_download"):
            mod.hf_hub_download = huggingface_hub.hf_hub_download
        if hasattr(mod, "snapshot_download"):
            mod.snapshot_download = huggingface_hub.snapshot_download

try:
    import speechbrain.utils.importutils
    original_ensure_module = speechbrain.utils.importutils.LazyModule.ensure_module
    def patched_ensure_module(self, *args, **kwargs):
        try:
            return original_ensure_module(self, *args, **kwargs)
        except ImportError:
            return None
    speechbrain.utils.importutils.LazyModule.ensure_module = patched_ensure_module
except Exception:
    pass

# ──────────────────────────────────────────────
# Compatibility patch for PyTorch 2.6+ Error Handling
# ──────────────────────────────────────────────
import torch
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    try:
        return original_torch_load(*args, **kwargs)
    except TypeError as e:
        if "weights_only" in str(e):
            kwargs.pop("weights_only", None)
            return original_torch_load(*args, **kwargs)
        raise
torch.load = patched_torch_load

if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    try:
        import torch.torch_version
        import pyannote.audio.core.task
        torch.serialization.add_safe_globals([
            torch.torch_version.TorchVersion,
            pyannote.audio.core.task.Specifications
        ])
    except Exception:
        pass


import whisperx
import whisperx.asr

# ──────────────────────────────────────────────
# Compatibility patch for WhisperX VAD HTTPError (S3 Bucket 301/403)
# ──────────────────────────────────────────────
import whisperx.vad
original_load_vad_model = whisperx.vad.load_vad_model
def patched_load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, model_fp=None):
    from pyannote.audio import Model
    import torch
    import whisperx
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # 1. RIP OUT THE FALLBACK. If there's no token, crash loudly.
    if not hf_token:
        raise ValueError("FATAL: No HF_TOKEN found! You MUST provide a HuggingFace token. The original WhisperX S3 bucket is dead (301 Error).")
    
    print("      (Bypassing dead WhisperX S3 bucket, securely fetching Pyannote VAD model from HuggingFace...)")
    
    # 2. Force Pyannote download using your token
    vad_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    hyperparameters = {"onset": vad_onset, "offset": vad_offset, "min_duration_on": 0.1, "min_duration_off": 0.1}
    vad_pipeline = whisperx.vad.VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    
    try:
        vad_pipeline.instantiate(hyperparameters)
    except ValueError as e:
        if "onset" in str(e):
            valid_params = {k: v for k, v in hyperparameters.items() if k not in ["onset", "offset"]}
            vad_pipeline.instantiate(valid_params)
            vad_pipeline.onset = hyperparameters.get("onset", vad_onset)
            vad_pipeline.offset = hyperparameters.get("offset", vad_offset)
        else:
            raise e
            
    return vad_pipeline

# Overwrite in both namespaces since `asr.py` natively binds it via "from .vad import load_vad_model"
whisperx.vad.load_vad_model = patched_load_vad_model
whisperx.asr.load_vad_model = patched_load_vad_model

load_dotenv()

# ──────────────────────────────────────────────
# Step 1: Processing Audio with WhisperX Pipeline
# ──────────────────────────────────────────────

def process_audio(audio_path, hf_token=None):
    
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    # line 169 — handle empty string too, not just None
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")
    # CRITICAL: Export the HF_TOKEN to env vars before load_model! 
    # The VAD monkey-patch reads this immediately to bypass the dead S3 bucket.
    os.environ["HF_TOKEN"] = hf_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure CPU uses float32 or int8 safely
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"\n[1/5] Loading initial whisper model on {device.upper()}...")
    
    # Newer faster-whisper packages require these extra parameters strictly.
    # Set multilingual=True per user request, and set the rest to None to gracefully bypass the bug
    fallback_asr_opts = {
        "multilingual": True,
        "hotwords": None,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None
    }
    
    try:
        model = whisperx.load_model("small", device, compute_type="float16", asr_options=fallback_asr_opts)
    except ValueError as e:
        if "compute type" in str(e).lower():
            # Graceful degrade if float16 isn't supported on their specific GPU/CPU
            try:
                model = whisperx.load_model("small", device, compute_type="float32", asr_options=fallback_asr_opts)
            except ValueError:
                model = whisperx.load_model("small", device, compute_type="int8", asr_options=fallback_asr_opts)
        else:
            raise e
            
    print(f"[2/5] Transcribing...")
    # WhisperX loads audio via ffmpeg natively
    audio_data = whisperx.load_audio(audio_path)
    result = model.transcribe(audio_data, batch_size=4)

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[3/5] Aligning timestamps to words...")
    # Add exception handling for load_align_model incase the audio language isn't supported gracefully
    try:
        model_a, metadata = whisperx.load_align_model(language_code=result.get("language", "en"), device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
        del model_a
    except Exception as e:
        print(f"      Alignment skipped due to error: {e}")
        pass
    
    # Free memory
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[4/5] Running diarization...")
    # Inject token via env var as backup
    os.environ["HF_TOKEN"] = hf_token
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio_data)

    print(f"[5/5] Assigning speakers to transcript text...")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Free memory
    del diarize_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Parse to simple key-value entries
    transcript_entries = []
    speaker_map = {}
    
    for seg in result["segments"]:
        spoken_text = seg.get("text", "").strip()
        if not spoken_text:
            continue
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker not in speaker_map:
            speaker_map[speaker] = f"person_{len(speaker_map) + 1}"
            
        transcript_entries.append({
            "person": speaker_map[speaker],
            "text": spoken_text
        })

    return transcript_entries, speaker_map


# ──────────────────────────────────────────────
# Step 2: Save File Outputs
# ──────────────────────────────────────────────
def save_transcript(transcript_entries, output_path):
    """Save the transcript with person_i labels to a .txt file strictly as key-value pairs."""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in transcript_entries:
            f.write(f'{entry["person"]} : {entry["text"]}\n')
    print(f"\n✅ Transcript saved to: {output_path}")


def generate_minutes_of_meeting(transcript_entries, speaker_map, output_path):
    """Generate LLM MoM summary from the transcribed entries"""
    full_transcript = "\n".join(
        f'{entry["person"]} : {entry["text"]}' for entry in transcript_entries
    )

    num_speakers = len(speaker_map)
    speakers_list = ", ".join(speaker_map.values())

    try:
        from crewai import LLM
        api_key = os.getenv("GROQ_API_KEY")
        llm = LLM(model="groq/llama-3.3-70b-versatile",api_key=api_key,temperature=0,max_tokens=None,timeout=None,max_retries=2)
        prompt = f"""You are a professional meeting minutes writer. 
Given the following transcript of a conversation between {num_speakers} participants ({speakers_list}), 
generate structured Minutes of Meeting (MoM).

Include the following sections:
1. **Meeting Overview** - Brief summary of what the meeting/conversation was about
2. **Participants** - List of participants ({speakers_list})
3. **Key Discussion Points** - Main topics discussed, organized with bullet points
4. **Decisions Made** - Any decisions or conclusions reached
5. **Action Items** - Any tasks or follow-ups mentioned (with responsible person if identifiable)
6. **Summary** - A brief overall summary of the meeting

TRANSCRIPT:
{full_transcript}

MINUTES OF MEETING:"""

        print("\n📝 Generating Minutes of Meeting using LLM...")
        res = llm.invoke(input=prompt)
        mom_text = res.content
    except Exception as e:
        print(f"\n⚠️  LLM unavailable ({e}). Generating rule-based summary...")
        mom_text = _generate_basic_mom(transcript_entries, speaker_map)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  MINUTES OF MEETING\n")
        f.write("=" * 70 + "\n\n")
        f.write(mom_text)
        f.write("\n")

    print(f"✅ Minutes of Meeting saved to: {output_path}")
    return mom_text

def _generate_basic_mom(transcript_entries, speaker_map):
    """Fallback: rule-based generation when LLM fails"""
    speakers = list(speaker_map.values())
    speaker_counts = {}
    speaker_texts = {}
    for entry in transcript_entries:
        p = entry["person"]
        speaker_counts[p] = speaker_counts.get(p, 0) + 1
        if p not in speaker_texts:
            speaker_texts[p] = []
        speaker_texts[p].append(entry["text"])

    lines = ["## Meeting Overview", f"A conversation between {len(speakers)} participants.\n", "## Participants"]
    for spk in speakers:
        lines.append(f"- {spk} ({speaker_counts.get(spk, 0)} speaking segments)")
    lines.extend(["", "## Conversation Summary", "Below are the first few talking points from each participant:\n"])
    
    for spk in speakers:
        texts = speaker_texts.get(spk, [])
        lines.append(f"**{spk}:**")
        for t in texts[:3]:
            lines.append(f"  - \"{t}\"")
        if len(texts) > 3:
            lines.append(f"  - ... and {len(texts) - 3} more segments")
        lines.append("")
    return "\n".join(lines)



# ──────────────────────────────────────────────
# Main Control Flow
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Main Control Flow
# ──────────────────────────────────────────────
from typing import Optional

@tool
def transcribe_and_generate_mom(audio_file: str, hf_token: Optional[str] = None) -> str:
    """
    Transcribes an audio file using WhisperX, performs speaker diarization, 
    and generates a Minutes of Meeting (MoM) summary.
    
    Inputs:
    - audio_file (str): The ABSOLUTE path to the local audio file to process.
    - hf_token (str, optional): HuggingFace token for Pyannote VAD model.
    
    Returns:
    A string detailing success or failure, including the absolute paths to the 
    generated transcript text file and the MoM text file.
    """
    # Force absolute path immediately
    abs_audio_file = os.path.abspath(audio_file)

    if not os.path.exists(abs_audio_file):
        return f"Error: Audio file not found at path: {abs_audio_file}"

    base_name = os.path.splitext(abs_audio_file)[0]
    transcript_path = f"{base_name}_transcript.txt"
    mom_path = f"{base_name}_minutes_of_meeting.txt"

    print(f"\n{'='*70}\n  Processing: {abs_audio_file}\n{'='*70}\n")

    try:
        # Call your processing logic directly every time
        transcript_entries, speaker_map = process_audio(abs_audio_file, hf_token=hf_token)

        save_transcript(transcript_entries, transcript_path)
        mom_text = generate_minutes_of_meeting(transcript_entries, speaker_map, mom_path)
        
        print(f"\n{'='*70}\n  All done!")
        
        # Return a highly descriptive string for the LLM
        return (
            f"Success: Audio processed successfully. "
            f"Transcript saved at absolute path: {transcript_path} | "
            f"Minutes of Meeting (MoM) saved at absolute path: {mom_path}"
        )
    except Exception as e:
        return f"Error: Failed to process audio due to exception: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transcript & MoM using WhisperX")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote")
    args = parser.parse_args()
    transcribe_and_generate_mom(args.audio_file, hf_token=args.hf_token)

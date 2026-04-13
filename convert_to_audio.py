import mimetypes
import os
from moviepy import VideoFileClip, AudioFileClip
from crewai.tools import tool

@tool
def convert_to_audio(file_path: str) -> str:
    """
    Converts a local video file to an audio file (MP3). 
    Input must be a valid local file path (e.g., '/path/to/video.mp4').
    Returns the ABSOLUTE path to the generated audio file, or an error message if it fails.
    """
    # 1. Force absolute path immediately to avoid context errors between agents
    abs_file_path = os.path.abspath(file_path)

    if not os.path.exists(abs_file_path):
        return f"Error: File not found at absolute path: {abs_file_path}"
    
    mime_type, _ = mimetypes.guess_type(abs_file_path)
    
    if not mime_type:
        return f"Error: Could not determine the file type of {abs_file_path}"
        
    if mime_type.startswith('audio/'):
        return f"Success: File is already an audio file. Path: {abs_file_path}"
        
    if mime_type.startswith('video/'):
        # 2. This guarantees the output is generated in the exact same absolute directory
        base_name = os.path.splitext(abs_file_path)[0]
        output_file = f"{base_name}.mp3"
        
        if os.path.exists(output_file):
            return f"Success: Output file already exists. Path: {output_file}"

        try:
            video = VideoFileClip(abs_file_path)
            if video.audio:
                video.audio.write_audiofile(output_file, logger=None)
                video.audio.close()
                video.close()
                # 3. Return the absolute path so Agent 2 has zero ambiguity
                return f"Success: Video converted to audio. Path: {output_file}"
            else:
                video.close()
                return "Error: Video does not contain an audio stream."
        except Exception as e:
            # 4. FALLBACK LOGIC: Handle .mp4 containers that are actually audio-only
            try:
                audio_clip = AudioFileClip(abs_file_path)
                audio_clip.write_audiofile(output_file, logger=None)
                audio_clip.close()
                return f"Success: Audio extracted via fallback (audio-only container). Path: {output_file}"
            except Exception as fallback_e:
                return f"Error: Failed to convert video ({str(e)}) AND fallback audio extraction failed ({str(fallback_e)})"
    else:
        return "Error: File is neither a recognized video nor audio file."
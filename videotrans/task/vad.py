import time
import traceback

from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
import numpy as np


def get_speech_timestamp_silero(input_wav,
                         threshold=None,
                         min_speech_duration_ms=None,
                         max_speech_duration_ms=None,
                         min_silent_duration_ms=None):
        # Prevent filling in errors
        min_speech_duration_ms=0#int(max(min_speech_duration_ms,0))
        min_silent_duration_ms=int(max(min_silent_duration_ms,50))
        max_speech_duration_ms=int(min(max(max_speech_duration_ms,min_speech_duration_ms+1000),30000))
        logger.debug(f'[silero-VAD]Fix:VAD segmentation parameters:{threshold=},{min_speech_duration_ms=}ms,{max_speech_duration_ms=}ms,{min_silent_duration_ms=}ms')

        sampling_rate = 16000
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import (
            VadOptions,
            get_speech_timestamps
        )
        vad_p = {
            "threshold": threshold,
            "min_speech_duration_ms": min_speech_duration_ms,
            "max_speech_duration_s": float(max_speech_duration_ms/1000.0),
            "min_silence_duration_ms": min_silent_duration_ms,
        }
        def convert_to_milliseconds(timestamps):
            milliseconds_timestamps = []
            for timestamp in timestamps:
                milliseconds_timestamps.append(
                    [
                       int(round(timestamp["start"] / sampling_rate * 1000)),
                       int(round(timestamp["end"] / sampling_rate * 1000)),
                    ]
                )

            return milliseconds_timestamps

        speech_chunks = get_speech_timestamps(decode_audio(input_wav,
                                            sampling_rate=sampling_rate),
                                            vad_options=VadOptions(**vad_p)
        )
        return convert_to_milliseconds(speech_chunks)


def get_speech_timestamp(input_wav=None,
                         threshold=None,
                         min_speech_duration_ms=None,
                         max_speech_duration_ms=None,
                         min_silent_duration_ms=None):
    #Limit the range
    #The shortest voice duration shall not be less than 250ms
    min_speech_duration_ms=int(max(250,min_speech_duration_ms))
    #The mute threshold for cutting must not be less than 50ms
    min_silent_duration_ms=int(max(50,min_silent_duration_ms))

    logger.debug(f'[Ten-VAD]Fix after:VAD sentence segmentation parameters:{threshold=},{min_speech_duration_ms=}ms,{max_speech_duration_ms=}ms,{min_silent_duration_ms=}ms')
    frame_duration_ms = 16
    hop_size = 256
    st_=time.time()
    try:
        sr, data = Wavfile.read(input_wav)
    except Exception as e:
        msg=traceback.format_exc()
        logger.exception(f"Error reading wav file: {msg}",exc_info=True)
        return False

    # Calculate audio energy for adaptive threshold adjustment
    audio_energy = np.mean(np.abs(data)) if len(data) > 0 else 0
    # Adjust the threshold according to the audio energy to deal with excessive noise.
    adjusted_threshold = threshold
    if audio_energy > 10000:  # High energy audio (may be noisy)
        adjusted_threshold = max(threshold * 1.2, 0.3)  # Raise the threshold
    elif audio_energy < 1000:  # low energy audio
        adjusted_threshold = min(threshold * 0.8, 0.2)  # Lower the threshold

    logger.debug(f'[Ten-VAD] Audio Energy:{audio_energy}, adjusted threshold:{adjusted_threshold}')

    min_sil_frames = min_silent_duration_ms / frame_duration_ms
    initial_segments = _detect_raw_segments(data, adjusted_threshold, min_sil_frames, max_speech_frames=None)

    # --- The second stage: refine very long fragments more than 2s---
    refined_segments = []
    max_frames_limit = max_speech_duration_ms / frame_duration_ms
    tighter_min_sil_frames = (min_silent_duration_ms / 2) / frame_duration_ms
    _n=0
    _len=len(initial_segments)
    for s, e in initial_segments:
        duration = e - s
        _n+=1
        # It needs to be cropped again if it is greater than 2000ms
        if duration > (max_frames_limit+125):
            #Extract the audio data of this segment
            sub_data = data[s * hop_size: e * hop_size]
            # Re-detect using a halved silence threshold, with a maximum duration limit
            sub_segs = _detect_raw_segments(sub_data, adjusted_threshold, tighter_min_sil_frames,
                                                 max_speech_frames=max_frames_limit)

            for ss, se in sub_segs:
                refined_segments.append([s + ss, s + se])
        else:
            refined_segments.append([s, e])

    if not refined_segments:
        return False

    # --- Phase 3: Millisecond conversion & forced hard truncation protection ---
    # Even if it is subdivided twice, if someone speaks for 30 seconds without pausing, it still needs to be hard truncated.
    segments_ms = []
    for s, e in refined_segments:
        start_ms = int(s * frame_duration_ms)
        end_ms = int(e * frame_duration_ms)

        # Loop to ensure max_speech_duration_ms is not exceeded
        curr_s = start_ms
        while (end_ms - curr_s) > max_speech_duration_ms:
            # Try to truncate at silence instead of truncate abruptly
            # Calculate the middle silent area of the current block
            block_data = data[int(curr_s/1000*sr):int((curr_s + max_speech_duration_ms)/1000*sr)]
            # Find the last silent area
            block_segments = _detect_raw_segments(block_data, adjusted_threshold, min_sil_frames/2, max_speech_frames=None)
            if block_segments and len(block_segments) > 1:
                # If there are multiple segments, use the beginning of the last segment as the truncation point
                last_segment_start = block_segments[-2][1] * hop_size / sr * 1000
                truncate_point = int(curr_s + last_segment_start)
                if truncate_point > curr_s + max_speech_duration_ms * 0.8:
                    segments_ms.append([curr_s, truncate_point])
                    curr_s = truncate_point
                    continue
            # If no suitable truncation point is found, use hard truncation
            segments_ms.append([curr_s, curr_s + int(max_speech_duration_ms)])
            curr_s += int(max_speech_duration_ms)

        if end_ms - curr_s > 0:
            segments_ms.append([curr_s, end_ms])
    
    logger.debug(f'[Ten-VAD] Segmentation time{int(time.time() - st_)}s')
    
    speech_len = len(segments_ms)
    if speech_len <= 1:
        return segments_ms

    # --- Optimized fragment merging strategy ---
    merged_segments = []
    # The minimum speech segment is not allowed to be less than 500ms. It may not be effectively recognized and an error will be reported.
    min_speech_duration_ms = max(min_speech_duration_ms or 1000, 500)
    
    # First round: merge consecutive short clips
    temp_segments = []
    current_merge = None
    current_duration = 0
    
    for i, segment in enumerate(segments_ms):
        duration = segment[1] - segment[0]
        
        if duration < min_speech_duration_ms:
            #Short fragments need to be merged
            if current_merge is None:
                current_merge = segment.copy()
                current_duration = duration
            else:
                # Calculate the distance from the current merged segment
                gap = segment[0] - current_merge[1]
                # If the interval is small, merge into the current segment
                if gap < min_silent_duration_ms:
                    current_merge[1] = segment[1]
                    current_duration += duration + gap
                else:
                    # If the interval is large, end the current merge and start a new merge
                    temp_segments.append(current_merge)
                    current_merge = segment.copy()
                    current_duration = duration
        else:
            # Long fragments, check if there are any outstanding merges
            if current_merge is not None:
                # Calculate the distance from the previous merged segment
                gap = segment[0] - current_merge[1]
                # If the interval is small, merge into the current long segment
                if gap < min_silent_duration_ms * 1.5:
                    segment[0] = current_merge[0]
                else:
                    # Otherwise, add merged segments
                    temp_segments.append(current_merge)
                current_merge = None
                current_duration = 0
            temp_segments.append(segment)
    
    # Process the last merged segment
    if current_merge is not None:
        temp_segments.append(current_merge)
    
    # Second round: Check the merged fragments to make sure there are no fragments that are too short
    for i, segment in enumerate(temp_segments):
        duration = segment[1] - segment[0]
        
        if duration >= min_speech_duration_ms:
            merged_segments.append(segment)
        else:
            # Still too short, try merging to adjacent segments
            if i == 0 and len(temp_segments) > 1:
                # First fragment, merge to next one
                temp_segments[i+1][0] = segment[0]
            elif i == len(temp_segments) - 1 and len(merged_segments) > 0:
                #The last fragment is merged into the previous one
                merged_segments[-1][1] = segment[1]
            elif len(merged_segments) > 0 and i < len(temp_segments) - 1:
                #Middle fragment, merge to the closer side
                prev_gap = segment[0] - merged_segments[-1][1]
                next_gap = temp_segments[i+1][0] - segment[1]
                
                if prev_gap <= next_gap:
                    merged_segments[-1][1] = segment[1]
                else:
                    temp_segments[i+1][0] = segment[0]
            else:
                # If it cannot be merged, add it as a separate fragment
                merged_segments.append(segment)
    
    logger.debug(f'[Ten-VAD] When split and merged for sharing:{int(time.time()-st_)}s')
    return merged_segments

def _detect_raw_segments(data, threshold, min_silent_frames, max_speech_frames=None):
    'Internal helper function: Detects speech segments based on a given silence threshold and maximum length.\n    '
    hop_size = 256
    
    ten_vad_instance = TenVad(hop_size, threshold)
    
    # Make sure the data is a one-dimensional array
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Dimensionality reduction to mono
    
    # Calculate the number of valid frames and ensure that the length of each frame is hop_size
    num_frames = (data.shape[0] - hop_size) // hop_size + 1

    segments = []
    triggered = False
    speech_start_frame = 0
    silence_frame_count = 0

    for i in range(num_frames):
        # Ensure that the frame length taken each time is hop_size
        audio_frame = data[i * hop_size: (i + 1) * hop_size]
        
        # Make sure the audio frame length is correct
        if len(audio_frame) != hop_size:
            continue
            
        # Make sure the data type is correct
        if audio_frame.dtype != np.int16:
            audio_frame = audio_frame.astype(np.int16)

        _, is_speech = ten_vad_instance.process(audio_frame)

        if triggered:
            current_speech_len = i - speech_start_frame
            if is_speech == 1:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            #End conditions: 1. Silence meets the length 2. (Optional) Forced cut off when the maximum length is reached
            is_silence_timeout = silence_frame_count >= min_silent_frames
            is_max_timeout = max_speech_frames is not None and current_speech_len >= max_speech_frames

            if is_silence_timeout or is_max_timeout:
                if is_max_timeout:
                    end_frame = i
                else:
                    end_frame = i - silence_frame_count

                segments.append([speech_start_frame, end_frame])
                triggered = False
                silence_frame_count = 0
        else:
            if is_speech == 1:
                triggered = True
                speech_start_frame = i
                silence_frame_count = 0


    if triggered:
        end_frame = num_frames - silence_frame_count
        segments.append([speech_start_frame, end_frame])

    return segments


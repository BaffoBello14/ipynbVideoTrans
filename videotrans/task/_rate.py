"""
#Detailed synchronization principle description

Align translated dubbing and original video timelines with audio speed-up and video slow-down.


*For the sake of simplicity, complex methods such as frame interpolation, frame patching, optical flow method, etc. are not considered for the slow video speed. Just use setpts=X*PTS -fps_mode vfr*
*Audio acceleration uses https://breakfastquay.com/rubberband/


Main implementation principles

# Function overview, use python3 to develop video translation function:
1. For the video pronunciated in language A, separate the silent picture video file and the audio file, use speech recognition to identify the original subtitles from the audio file, then translate the subtitles into subtitles in language B, then dub the subtitles in language B into language dubbing in language B, and then synchronize the audio and video, align and merge the subtitles in language B and the dubbing in language B with the silent video separated from A into a new video.
2. The part we are currently working on is "dubbing, subtitles, and video alignment." Language B subtitles are dubbed one by one, and the dubbing of each subtitle generates a wav audio file.
3. Because the languages ​​are different, each dubbing may be longer than the duration of the subtitle. For example, the duration of the subtitle is 3 seconds. If the duration of the dubbed mp3 is less than or equal to 3 seconds, it will not be affected. However, if the duration of the dubbing is longer than 3 seconds, there will be a problem. The audio clips need to be automatically accelerated to 3 seconds to achieve synchronization. Alignment can also be achieved by cutting out the segment of the original video corresponding to the original subtitle and playing it at a slow speed to extend the duration of the video until it matches the dubbing duration. Of course, you can also auto-accelerate audio and slow down video at the same time, thereby avoiding too much audio acceleration or too much video slowdown.


## Preprocessing

**When there is audio acceleration and/or video slowness, first change the end_time of each subtitle to be consistent with the start_time of the next subtitle, that is, remove the silence interval between subtitles**

## Strategy when audio and video are enabled at the same time
1. If the dubbing duration is less than the subtitle duration of the current clip, there is no need for audio acceleration and video slowdown
2. If the dubbing duration is greater than the subtitle duration of the current clip, calculate the acceleration factor required to shorten the dubbing duration to match the subtitle duration.
    - If the multiple is less than or equal to 1.2, then the audio can be accelerated as it is without slowing down the video.
    - If the multiplier is greater than 1.2, audio acceleration and video slowdown will each bear half of the burden, ignoring all restrictions.

## When only using audio acceleration

1. If the dubbing duration is less than the original subtitle duration of the current clip, no audio acceleration is required
2. If the dubbing duration is greater than the original subtitle duration of the current clip, the dubbing duration will be forcibly shortened to the matching duration. The maximum multiple shall not exceed the predetermined maximum limit.
3. Pay attention to the silent interval between the beginning and end and the subtitles, especially the silence space that may remain after utilization. The length of the final synthesized audio should be equal to the video length when the video exists (self.novoice_mp4). When it does not exist, the length should not be less than self.raw_total_time.

## Only when the video is slow
1. If the dubbing duration is less than the original subtitle duration of the current clip, there is no need to slow down the video and it will be cropped directly from the start time of this subtitle to the start time of the next subtitle. If this is the first subtitle, it will be cropped from time 0.
2. If the dubbing duration is greater than the original subtitle duration of the current clip, the video clip (duration is total_a) is forced to be slowly extended to the same length as the dubbing. Note here that the maximum PTS multiple does not exceed max_video_pts_rate.
3. When cropping, attention should be paid to the area before the first subtitle (the start time may be greater than 0) and the area after the last subtitle (the end time may be less than the end of the video)
4. There is no need to slow down the clips, directly crop the start time of this subtitle to the start time of the next subtitle.


## When there is no `audio acceleration` or `video slowdown`

- The first step is to splice audio by subtitles
1. If the first subtitle does not start from 0, silence will be filled in front.
2. If the time between the start time of this subtitle and the start time of the next subtitle is greater than or equal to the dubbing time of this subtitle, the dubbing file will be spliced ​​directly. If the difference is greater than 0, that is, there is still enough space, then silence will be filled in afterwards.
3. If the time between the start time of this subtitle and the start time of the next subtitle is less than the dubbing time of this subtitle, it will be spliced directly without any other processing.
4. If it is the last subtitle, just splice the dubbing clip directly without judging whether there is space behind it.

- The second step is to check whether the video file exists
1. self.novoice_mp4 is not None, and the file exists, then the video exists. At this time, compare the combined audio duration and video duration.
    - If the audio duration is less than the video duration, silence will be padded at the end of the audio until the length is consistent
    - If the audio duration is greater than the video duration, the final freeze frame of the video will be extended until it is consistent with the audio duration.
2. If the video file does not exist, no other processing is required


## Video clips smaller than 1024B can be considered invalid and filtered out. Container and meta information plus a frame of pictures must be larger than 1024B.

## In the absence of slow video participation, organize the subtitle timeline according to dubbing to ensure that the sound and subtitles appear and disappear at the same time.
## When there is slow video participation, align the dubbing clips with the video clips one by one. If the dubbing clip is shorter than the current video clip, add mute. If it is larger, ignore it and continue splicing. The subtitle timeline will be displayed based on the dubbing.
=================================================================================================

## Points to note
1. ffmpeg cannot process videos accurately to the millisecond level, so when using PTS for slow speed, the final output video may be shorter or longer than expected.
2. Fps is not fixed, it may be 25, 29 or 30, etc. The length of some clips may be less than 1 frame. If FFmpeg performs variable speed processing, it will most likely fail. Therefore, when audio acceleration or video slowing is involved, the gaps between the current subtitle and the next subtitle should be given to the current subtitle in advance, that is, the end time of the current subtitle is changed to the start time of the next subtitle.
"""


import os
import shutil
import time
from pathlib import Path

#Introduce soundfile and audio processing
import soundfile as sf
import numpy as np  # Add numpy for channel processing
from pydub import AudioSegment

# Try to import pyrubberband
try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND = True
except ImportError:
    HAS_RUBBERBAND = False

from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang
from videotrans.process.signelobj import GlobalProcessManager
from videotrans.util import tools


def _cut_video_get_duration(i, task, novoice_mp4_original, preset, crf):
    '\n    Trim video clips and process in slow motion (PTS) if desired.\n    '
    task['actual_duration'] = 0 
    
    # Force the use of absolute paths
    input_video_path = Path(novoice_mp4_original).resolve().as_posix()
    
    # Original clip duration
    source_duration_ms = task['end'] - task['start']
    if source_duration_ms <= 0:
        logger.warning(f"[Video-Cut] clip{i}Original duration <=0 ({task['start']}-{task['end']}), skip processing")
        return task

    # Target duration
    target_duration_ms = task.get('target_time', source_duration_ms)
    
    ss_time = tools.ms_to_time_string(ms=task['start'], sepflag='.')
    source_duration_s = source_duration_ms / 1000.0
    target_duration_s = target_duration_ms / 1000.0
    
    #PTS coefficient
    pts_factor = task.get('pts', 1.0)
    
    flag = f'[Video-Cut] clip{i} [Original:{task["start"]}-{task["end"]}ms] [target:{target_duration_ms}ms] [PTS:{pts_factor:.4f}]'
    logger.debug(f"{flag} Ready to start processing...")

    # Main command build
    cmd = [
        '-y',
        '-ss', ss_time,
        '-t', f'{source_duration_s:.6f}',
        '-i', input_video_path, # Use absolute path
        '-an',
        '-c:v', 'libx264', 
        '-g', '1',
        '-preset', preset, 
        '-crf', crf,
        '-pix_fmt', 'yuv420p'
    ]

    filter_complex = []
    if abs(pts_factor - 1.0) >= 0.001:
        filter_complex.append(f"setpts={pts_factor+0.005}*PTS")
    else:
        filter_complex.append("setpts=PTS")

    cmd.extend(['-vf', ",".join(filter_complex)])
    cmd.extend(['-fps_mode', 'vfr']) # Critical fixes
    cmd.extend(['-t', f'{target_duration_s:.6f}']) # Forced limit on output duration
    
    cmd.append(os.path.basename(task['filename']))

    # Get the working directory (used to store temporary files)
    work_dir = Path(task['filename']).parent.as_posix()
    
    try:
        # Execute FFmpeg
        tools.runffmpeg(cmd, force_cpu=True, cmd_dir=work_dir)
        
        file_path = Path(task['filename'])
        
        # Check whether it is successful, if it fails, execute the back-up logic
        if not file_path.exists() or file_path.stat().st_size < 1024:
            logger.warning(f"{flag} Variable speed generation failed or the file is invalid, try cutting without variable speed...")
            
            # [Correction] The bottom line command must also include fps_mode and setpts=PTS to ensure splicing compatibility
            cmd_backup = [
                '-y', 
                '-ss', ss_time, 
                '-t', f'{source_duration_s:.6f}', # Original duration of use
                '-i', input_video_path,
                '-an', 
                '-c:v', 'libx264', 
                '-g', '1',
                '-preset', preset, 
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                '-vf', 'setpts=PTS',  # Explicitly add
                '-fps_mode', 'vfr',   # Explicitly add
                os.path.basename(task['filename'])
            ]
            tools.runffmpeg(cmd_backup, force_cpu=True, cmd_dir=work_dir)

        # Check again
        if file_path.exists() and file_path.stat().st_size >= 1024:
            try:
                real_time = tools.get_video_duration(task["filename"])
            except Exception as e:
                logger.error(f"{flag} Failed to get duration:{e}")
                real_time = 0

            task['actual_duration'] = real_time
            logger.debug(f"{flag} Done. Actual generation time:{real_time}ms")
        else:
            task['actual_duration'] = 0
            logger.error(f"{flag} The final build failed.")
            
    except Exception as e:
        logger.error(f"{flag} Handling exceptions:{e}")
        try:
            if Path(task['filename']).exists():
                Path(task['filename']).unlink()
        except:
            pass
            
    return task


def _change_speed_rubberband(
    input_path,
    target_duration,
    *,
    allow_slowdown=False,
    max_slowdown_extra_ms=0,
    max_slowdown_ratio=1.15,
):
    '\n    Audio speed shifting using Rubber Band (speed up and optional slow-down).\n    '
    if not HAS_RUBBERBAND:
        logger.warning(f"[Audio-RB] Rubberband is not installed, skip:{input_path}")
        return

    try:
        y, sr = sf.read(input_path)
        if len(y) == 0:
            logger.warning(f"[Audio-RB] Empty audio file:{input_path}")
            return
            
        current_duration = int((len(y) / sr) * 1000)
        
        if target_duration <= 0:
            target_duration = 1

        if target_duration > current_duration:
            if not allow_slowdown or max_slowdown_extra_ms <= 0:
                logger.debug(
                    f"[Audio-RB] target ({target_duration}ms) > current ({current_duration}ms), "
                    "skip slowdown (disabled or no budget)."
                )
                return
            gap = target_duration - current_duration
            try:
                ratio_cap = float(max_slowdown_ratio or 1.0)
            except (TypeError, ValueError):
                ratio_cap = 1.15
            if ratio_cap < 1.0:
                ratio_cap = 1.0
            max_extra_by_ratio = max(0, int(current_duration * (ratio_cap - 1.0)))
            extra_budget = min(gap, int(max_slowdown_extra_ms), max_extra_by_ratio)
            if extra_budget < 1:
                logger.debug(
                    f"[Audio-RB] slowdown skipped (gap={gap}ms, budget cap): "
                    f"current={current_duration}ms target={target_duration}ms"
                )
                return
            target_duration = current_duration + extra_budget

        time_stretch_rate = current_duration / target_duration
        
        # Limit range
        time_stretch_rate = max(0.2, min(time_stretch_rate, 50.0))
        
        logger.debug(
            f"[Audio-RB] {input_path} Original:{current_duration}ms -> target:{target_duration}ms "
            f"rate:{time_stretch_rate:.3f}"
        )

        y_stretched = pyrb.time_stretch(y, sr, time_stretch_rate)
        
        # [Key Correction] Make sure the output is Stereo (2 channels) to prevent subsequent ffmpeg concat errors.
        # If it is mono (ndim=1), copy it to dual channel
        if y_stretched.ndim == 1:
            y_stretched = np.column_stack((y_stretched, y_stretched))
        
        sf.write(input_path, y_stretched, sr)
        
    except Exception as e:
        logger.error(f"[Audio-RB] Audio processing failed{input_path}: {e}")


class SpeedRate:
    MIN_CLIP_DURATION_MS = 40
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_CHANNELS = 2

    def __init__(self,
                 *,
                 queue_tts=None,
                 shoud_videorate=False,
                 shoud_audiorate=False,
                 uuid=None,
                 novoice_mp4=None,
                 raw_total_time=0,
                 target_audio=None,
                 cache_folder=None,
                 remove_silent_mid=False,
                 align_sub_audio=True,
                 stretch_short_max_ms=0,
                 stretch_short_max_ratio=1.15,
                 ):
        self.align_sub_audio = align_sub_audio
        self.stretch_short_max_ms = int(max(0, stretch_short_max_ms or 0))
        try:
            self.stretch_short_max_ratio = float(stretch_short_max_ratio or 1.0)
        except (TypeError, ValueError):
            self.stretch_short_max_ratio = 1.15
        if self.stretch_short_max_ratio < 1.0:
            self.stretch_short_max_ratio = 1.0
        self.raw_total_time = raw_total_time if raw_total_time is not None else 0
        self.remove_silent_mid = remove_silent_mid
        self.queue_tts = queue_tts
        self.len_queue = len(queue_tts)
        self.shoud_videorate = shoud_videorate
        self.shoud_audiorate = shoud_audiorate
        self.uuid = uuid
        self.novoice_mp4_original = novoice_mp4
        self.novoice_mp4 = novoice_mp4
        self.cache_folder = cache_folder if cache_folder else Path(
            f'{TEMP_DIR}/{str(uuid if uuid else time.time())}').as_posix()
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

        self.stop_show_process = False
        self.video_info = {}
        self.target_audio = target_audio

        self.max_audio_speed_rate = float(settings.get('max_audio_speed_rate', 100))
        self.max_video_pts_rate = float(settings.get('max_video_pts_rate', 10))

        self.audio_data = []
        self.audio_slow_data = []
        self.video_for_clips = [] 

        self.crf = "20"
        self.preset = "veryfast"
        
        try:
            if Path(ROOT_DIR + "/crf.txt").exists():
                self.crf = str(int(Path(ROOT_DIR + "/crf.txt").read_text()))
            if Path(ROOT_DIR + "/preset.txt").exists():
                preset_tmp = str(Path(ROOT_DIR + "/preset.txt").read_text().strip())
                if preset_tmp in ['ultrafast', 'veryfast', 'medium', 'slow']:
                    self.preset = preset_tmp
        except:
            pass

        self.audio_speed_rubberband = shutil.which("rubberband")
        self._rubberband_ok = bool(HAS_RUBBERBAND)
        logger.debug(
            f"[SpeedRate] Init. AudioRate={self.shoud_audiorate}, VideoRate={self.shoud_videorate}, "
            f"pyrubberband={HAS_RUBBERBAND}, rubberband_cli={bool(self.audio_speed_rubberband)}, "
            f"stretch_short_max_ms={self.stretch_short_max_ms}, ratio={self.stretch_short_max_ratio}"
        )

    def run(self):
        if not self.shoud_audiorate and not self.shoud_videorate:
            logger.debug('[SpeedRate] does not enable variable speed and enters normal splicing mode.')
            self._run_no_rate_change_mode()
            return self.queue_tts
        
        logger.debug('[SpeedRate] Enables variable speed and enters alignment mode.')
        
        # 1. Preprocessing
        self._prepare_data()
        
        # 2. Calculation
        self._calculate_adjustments()
        
        # 3. Audio speed change (speed up, then optional mild slow-down for short clips)
        if self.audio_data:
            tools.set_process(text='Processing audio speed...', uuid=self.uuid)
            if self._rubberband_ok:
                self._execute_audio_speedup_rubberband()
            else:
                 logger.warning('[SpeedRate] pyrubberband is not available, skipping audio physical speed shifting.')
        if self.audio_slow_data:
            tools.set_process(text='Stretching short dubbing clips...', uuid=self.uuid)
            if self._rubberband_ok:
                self._execute_audio_slowdown_rubberband()
            else:
                logger.warning(
                    '[SpeedRate] pyrubberband is not available, skipping mild stretch of short dubbing.'
                )
        
        # 4. Video speed change
        if self.shoud_videorate and self.video_for_clips:
            tools.set_process(text='Processing video speed...', uuid=self.uuid)
            processed_video_clips = self._video_speeddown()
            
            # Write back
            for clip in processed_video_clips:
                tts_idx = clip.get('tts_index',-1)
                if tts_idx <0:
                    continue
                real_duration = clip.get('actual_duration', 0)
                if tts_idx is not None  and 0 <= tts_idx < len(self.queue_tts):
                    # [Key] This is the final video slot length
                    self.queue_tts[tts_idx]['final_duration'] = real_duration
            
            self._concat_video(processed_video_clips)
            
            #Total update time
            if Path(self.novoice_mp4).exists():
                try:
                    self.raw_total_time = tools.get_video_duration(self.novoice_mp4)
                    logger.debug(f"[SpeedRate] New video generated, total duration:{self.raw_total_time}ms")
                except: pass
        else:
            # Unchanged video, the duration is the original slot duration
            for it in self.queue_tts:
                it['final_duration'] = it['source_duration']
            
        # 5. Audio alignment and splicing
        tools.set_process(text='Concatenating final audio...', uuid=self.uuid)
        self._concat_audio_aligned()

        return self.queue_tts

    def _prepare_data(self):
        'Data cleaning and preprocessing'
        tools.set_process(text="Preparing data...", uuid=self.uuid)
        
        if self.novoice_mp4_original and tools.vail_file(self.novoice_mp4_original):
            self.raw_total_time = tools.get_video_duration(self.novoice_mp4_original)

        for i in range(len(self.queue_tts)):
            current = self.queue_tts[i]

            # Subtitle starting point, used to split video and change speed
            current['start_time_source']=current['start_time']
            # If the video is slow and less than 100ms, set it to start from 0 to prevent errors in short video clips.
            if self.shoud_videorate and i == 0 and current['start_time']<100:
                current['start_time_source'] = 0
            
            # Fill the gap, change the end time of the subtitles to the next start time, and increase the speed range to reduce the speed change range
            # Except for item 0, the starting time point remains unchanged and only the end point is moved.
            if i < len(self.queue_tts) - 1:
                next_sub = self.queue_tts[i+1]
                current['end_time_source'] = next_sub['start_time']
                current['end_time'] = next_sub['start_time']
            else:
                current['end_time_source'] = self.raw_total_time
                current['end_time'] = self.raw_total_time

            current['source_duration'] = current['end_time_source'] - current['start_time_source']
            
            # Check the dubbing file
            if not current.get('filename') or not Path(current['filename']).exists():
                # Generate placeholder silence
                dummy_wav = Path(self.cache_folder, f'silent_place_{i}.wav').as_posix()
                AudioSegment.silent(duration=current['source_duration']).export(dummy_wav, format="wav")
                current['filename'] = dummy_wav
                current['dubb_time'] = current['source_duration']
                logger.debug(f"[Prepare] Subtitles[{current['line']}] No dubbing, generated{current['source_duration']}ms mute placeholder")
            else:
                current['dubb_time'] = len(AudioSegment.from_file(current['filename']))

    def _calculate_adjustments(self):
        'Calculation strategy'
        tools.set_process(text="Calculating sync adjustments...", uuid=self.uuid)
        # The video is slow, there may be a video without sound before the 0th subtitle
        if self.shoud_videorate and self.queue_tts[0]['start_time_source']>0:
            self.video_for_clips.append({
                    "start": 0,
                    "end": self.queue_tts[0]['start_time_source'],
                    "target_time": self.queue_tts[0]['start_time_source'],
                    "pts": 1,
                    "tts_index": -1,
                    "line": -1
            })
            
        
        for i, it in enumerate(self.queue_tts):
            source_dur = it['source_duration']
            dubb_dur = it['dubb_time']
            
            if self.shoud_videorate and source_dur <= 0:
                logger.warning(f"[Calc] subtitles[{it['line']}] Video slot <=0, skip")
                self.video_for_clips.append({
                    "start": 0, 
                    "end": 0, 
                    "target_time": 0, 
                    "pts": 1,
                    "tts_index": i, 
                })
                continue
            if source_dur<=0:
                continue

            video_target = source_dur
            audio_target = source_dur
            
            mode_log = ""
            # Only audio acceleration
            if self.shoud_audiorate and not self.shoud_videorate:
                mode_log = "Only Audio"
                if dubb_dur > source_dur:
                    ratio = dubb_dur / source_dur
                    if ratio > self.max_audio_speed_rate:
                        audio_target = int(dubb_dur / self.max_audio_speed_rate)
                    else:
                        audio_target = source_dur

            elif not self.shoud_audiorate and self.shoud_videorate:
                mode_log = "Only Video"
                if dubb_dur > source_dur:
                    video_target = dubb_dur
                    pts = video_target / source_dur
                    if pts > self.max_video_pts_rate:
                        video_target = int(source_dur * self.max_video_pts_rate)

            elif self.shoud_audiorate and self.shoud_videorate:
                mode_log = "Both"
                if dubb_dur > source_dur:
                    ratio = dubb_dur / source_dur
                    if ratio <= 1.2:
                        audio_target = source_dur
                        video_target = source_dur
                    else:
                        diff = dubb_dur - source_dur
                        joint_target = int(source_dur + (diff / 2))
                        audio_target = joint_target
                        video_target = joint_target
            


            #Register task
            if self.shoud_audiorate and audio_target < dubb_dur:
                self.audio_data.append({
                    "filename": it['filename'],
                    "dubb_time": dubb_dur,
                    "target_time": audio_target
                })
            elif (
                self.shoud_audiorate
                and self.stretch_short_max_ms > 0
                and dubb_dur < source_dur
            ):
                gap = source_dur - dubb_dur
                if gap > 0:
                    self.audio_slow_data.append({
                        "filename": it['filename'],
                        "dubb_time": dubb_dur,
                        "target_time": source_dur,
                        "max_slowdown_extra_ms": min(gap, self.stretch_short_max_ms),
                    })

            if self.shoud_videorate:
                pts = video_target / source_dur if source_dur > 0 else 1.0
                self.video_for_clips.append({
                    "start": it['start_time_source'],
                    "end": it['end_time_source'],
                    "target_time": video_target,
                    "pts": pts,
                    "tts_index": i,
                    "line": it['line']
                })
            
                it['final_duration'] = video_target
            # Record decision log
            logger.debug(f"[Calc] Mode={mode_log} Line={it['line']} | Source={source_dur} Dubb={dubb_dur} -> TargetV={video_target} TargetA={audio_target}")


    def _execute_audio_speedup_rubberband(self):
        logger.debug(f"[Audio] Start processing{len(self.audio_data)} audio speed changing tasks")
        all_task = []
        for d in self.audio_data:
            all_task.append(GlobalProcessManager.submit_task_cpu(
                _change_speed_rubberband, 
                input_path=d['filename'], 
                target_duration=d['target_time']
            ))
        for task in all_task:
            try: task.result()
            except: pass

    def _execute_audio_slowdown_rubberband(self):
        logger.debug(f"[Audio] Start mild stretch for {len(self.audio_slow_data)} short clips")
        all_task = []
        for d in self.audio_slow_data:
            all_task.append(
                GlobalProcessManager.submit_task_cpu(
                    _change_speed_rubberband,
                    input_path=d['filename'],
                    target_duration=d['target_time'],
                    allow_slowdown=True,
                    max_slowdown_extra_ms=d['max_slowdown_extra_ms'],
                    max_slowdown_ratio=self.stretch_short_max_ratio,
                )
            )
        for task in all_task:
            try:
                task.result()
            except Exception:
                pass

    def _video_speeddown(self):
        data = []
        for i, clip_info in enumerate(self.video_for_clips):
            clip_info['queue_index'] = clip_info.get('tts_index',-1)
            clip_info['filename'] = Path(self.cache_folder, f"clip_{i}_{clip_info['pts']:.3f}.mp4").as_posix()
            data.append(clip_info)
            
        all_task = []
        logger.debug(f"[Video] Submit{len(data)} video processing tasks")
        for i, d in enumerate(data):
            kw = {
                "i": i, 
                "task": d, 
                "novoice_mp4_original": self.novoice_mp4_original, 
                "preset": self.preset, 
                "crf": self.crf
            }
            all_task.append(GlobalProcessManager.submit_task_cpu(_cut_video_get_duration, **kw))

        processed_clips = []
        for task in all_task:
            try:
                res = task.result()
                if res: processed_clips.append(res)
            except Exception as e:
                logger.error(f"[Video] Task exception:{e}")
        
        processed_clips.sort(key=lambda x: x.get('queue_index', -1))
        return processed_clips

    def _concat_video(self, processed_clips):
        txt_content = []
        valid_cnt = 0
        for clip in processed_clips:
            if clip.get('actual_duration', 0) > 0 and Path(clip['filename']).exists():
                path = clip['filename'].replace("\\", "/")
                txt_content.append(f"file '{path}'")
                valid_cnt += 1
            else:
                logger.warning(f"[Video-Concat] Ignore invalid segments:{clip.get('filename')}")
        
        if valid_cnt == 0: 
            logger.error('[Video-Concat] No valid segments, skip splicing')
            return

        concat_list = Path(self.cache_folder, "video_concat.txt").as_posix()
        with open(concat_list, 'w', encoding='utf-8') as f:
            f.write("\n".join(txt_content))
            
        output_path = Path(self.cache_folder, "merged_video.mp4").as_posix()
        logger.debug(f"[Video-Concat] merge{valid_cnt} fragments ->{output_path}")
        
        cmd = ['-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_path]
        tools.runffmpeg(cmd, force_cpu=True, cmd_dir=self.cache_folder)

        if Path(output_path).exists():
            shutil.move(output_path, self.novoice_mp4)

    def _concat_audio_aligned(self):
        logger.debug('[Audio] Start aligning splicing...')
        audio_list = []
        current_timeline = self.queue_tts[0]['start_time']
        if current_timeline>0:
            audio_list.append(self._create_silen_file("head_0", current_timeline))
        
        for i, it in enumerate(self.queue_tts):
            # When the video is slow, use the actual duration of the video clip, otherwise use the subtitle interval duration
            slot_duration = it.get('final_duration', it['source_duration'])
            
            # [Background correction] If the video slot fails (0ms), fall back to source_duration
            # This at least ensures that the audio will not be messed up due to video failure and maintains audio continuity.
            if slot_duration <= 0:
                logger.warning(f"[Audio-Sync] Subtitles[{it['line']}] The video slot duration is 0, and the original duration is used for fallback:{it['source_duration']}ms")
                slot_duration = max(1, it['source_duration'])

            current_slot_audio_len = 0
            
            # 2. Dubbing file
            audio_file = it['filename']
            if Path(audio_file).exists():
                try:
                    seg = AudioSegment.from_file(audio_file)
                    if seg.channels != self.AUDIO_CHANNELS:
                        seg = seg.set_channels(self.AUDIO_CHANNELS)
                        seg.export(audio_file, format='wav')
                    
                    current_slot_audio_len += len(seg)
                except Exception as e:
                    logger.error(f"[Audio-Sync] Failed to read dubbing{audio_file}: {e}")
            
            # 3. Length alignment
            log_flag = ""
                        
            if current_slot_audio_len > slot_duration:
                # When the dubbing length is longer than the video clip or subtitle interval clip, and the video is slow, the final generated video (slot_duration) may be dozens of milliseconds shorter than theoretically required.
                log_flag = f"Audio overflow truncation{current_slot_audio_len}->{slot_duration}"
                
                try:
                    # Force phase audio when video is slow
                    if self.shoud_videorate:
                        cut_seg = AudioSegment.from_file(audio_file)[:slot_duration]
                        final_slot_path = Path(self.cache_folder, f"final_slot_cut_{i}.wav").as_posix()
                        cut_seg.export(final_slot_path, format='wav')
                        audio_list.append(final_slot_path)
                    else:
                        audio_list.append(audio_file) # Put it as it is
                except Exception as e:
                    logger.error(f"Failed to truncate audio:{e}")
                    audio_list.append(audio_file) # If it fails, put it as it is

            elif current_slot_audio_len < slot_duration:
                #Add tail silence
                diff = slot_duration - current_slot_audio_len
                log_flag = f"Add silence at the end of audio{diff}ms"
                audio_list.append(audio_file)
                audio_list.append(self._create_silen_file(f"tail_{i}", diff))
            else:
                log_flag = 'match'
                audio_list.append(audio_file)

            logger.debug(f"[Audio-Sync] Line={it['line']} | {log_flag} | [{current_slot_audio_len=} {slot_duration=}] | Timeline: {current_timeline} -> {current_timeline+slot_duration}")

            it['start_time'] = current_timeline
            it['end_time'] = current_timeline + slot_duration
            current_timeline += slot_duration

        self._exec_concat_audio(audio_list)
    
    def _run_no_rate_change_mode(self):
        # Splice directly when the speed is not changed
        tools.set_process(text=tr("Merging audio (No Speed Change)..."), uuid=self.uuid)
        
        audio_concat_list = []
        total_audio_duration = 0

        for i, it in enumerate(self.queue_tts):

            prev_end = 0 if i == 0 else self.queue_tts[i-1].get('end_pos_for_concat', 0)
            start_time = it['start_time']
            # The previous silent interval
            gap = start_time - prev_end
            
            if not self.remove_silent_mid and gap > 0:
                audio_concat_list.append(self._create_silen_file(f"gap_{i}", gap))
                total_audio_duration += gap
            
            dubb_len = 0
            if it.get('filename') and Path(it['filename']).exists():
                audio_concat_list.append(it['filename'])
                dubb_len = len(AudioSegment.from_file(it['filename']))
            elif it.get('filename'):
                dur = max(0, it['end_time'] - it['start_time'])
                if dur > 0:
                    audio_concat_list.append(self._create_silen_file(f"sub_{i}", dur))
                    dubb_len = dur
            
            total_audio_duration += dubb_len
            it['end_pos_for_concat'] = total_audio_duration
            
            if self.align_sub_audio:
                it['start_time'] = total_audio_duration - dubb_len
                it['end_time'] = total_audio_duration

        if self.raw_total_time > total_audio_duration:
            audio_concat_list.append(self._create_silen_file("tail_end", self.raw_total_time - total_audio_duration))

        self._exec_concat_audio(audio_concat_list)

    def _create_silen_file(self, name, duration_ms):
        path = Path(self.cache_folder, f"silence_{name}.wav").as_posix()
        duration_ms = max(1, int(duration_ms))
        AudioSegment.silent(duration=duration_ms, frame_rate=self.AUDIO_SAMPLE_RATE) \
                    .set_channels(self.AUDIO_CHANNELS) \
                    .export(path, format="wav")
        return path

    def _exec_concat_audio(self, file_list):
        if not file_list: return
        
        concat_txt = Path(self.cache_folder, 'final_audio_concat.txt').as_posix()
        tools.create_concat_txt(file_list, concat_txt=concat_txt)
        
        temp_wav = Path(self.cache_folder, 'final_audio_temp.wav').as_posix()
        # Force the use of cache_folder as cwd to avoid relative path problems
        cmd = ['-y', '-f', 'concat', '-safe', '0', '-i', concat_txt, '-c:a', 'copy', temp_wav]
        tools.runffmpeg(cmd, force_cpu=True, cmd_dir=self.cache_folder)
        
        if Path(temp_wav).exists():
            shutil.move(temp_wav, self.target_audio)
            logger.debug(f"[Audio-Concat] Final audio generated:{self.target_audio}")
        else:
            logger.error('[Audio-Concat] Final audio generation failed')


# Specifically for dubbing subtitles and processing them separately
class TtsSpeedRate(SpeedRate):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.shoud_videorate=False
        self.max_audio_speed_rate=100


    def run(self):
        if not self.shoud_audiorate:
            logger.debug('[SpeedRate] does not enable variable speed and enters normal splicing mode.')
            self._run_no_rate_change_mode()
            return self.queue_tts
        # Deleting the timeline is illegal
        self.queue_tts=[it for it in self.queue_tts if it['end_time']-it['start_time']>0]

        logger.debug('[SpeedRate] Enables variable speed and enters alignment mode.')

        # 1. Preprocessing
        self._prepare_data()

        # 2. Calculation
        self._calculate_adjustments()

        # 3. Audio speed change (speed up, then optional mild stretch for short clips)
        if self.audio_data:
            tools.set_process(text='Processing audio speed...', uuid=self.uuid)
            if self._rubberband_ok:
                self._execute_audio_speedup_rubberband()
            else:
                 logger.warning('[SpeedRate] pyrubberband is not available, skipping audio physical speed shifting.')
        if self.audio_slow_data:
            tools.set_process(text='Stretching short dubbing clips...', uuid=self.uuid)
            if self._rubberband_ok:
                self._execute_audio_slowdown_rubberband()
            else:
                logger.warning(
                    '[SpeedRate] pyrubberband is not available, skipping mild stretch of short dubbing.'
                )

        tools.set_process(text='Concatenating final audio...', uuid=self.uuid)
        self._concat_audio_aligned()

        return self.queue_tts

    def _prepare_data(self):
        'Data cleaning and preprocessing'
        tools.set_process(text="Preparing data...", uuid=self.uuid)
        
        _len=len(self.queue_tts)
        for i in range(_len):
            current = self.queue_tts[i]
            if i<_len-1:
                current['end_time']=self.queue_tts[i+1]['start_time']
                        
            current['source_duration'] = current['end_time'] - current['start_time']

            # Check the dubbing file
            if not current.get('filename') or not Path(current['filename']).exists():
                # Generate placeholder silence
                dummy_wav = Path(self.cache_folder, f'silent_place_{i}.wav').as_posix()
                AudioSegment.silent(duration=current['source_duration']).export(dummy_wav, format="wav")
                current['filename'] = dummy_wav
                current['dubb_time'] = current['source_duration']
                logger.debug(f"[Prepare] Subtitles[{current['line']}] No dubbing, generated{current['source_duration']}ms mute placeholder")
            else:
                current['dubb_time'] = len(AudioSegment.from_file(current['filename']))

    def _calculate_adjustments(self):
        'Calculation strategy'
        tools.set_process(text="Calculating sync adjustments...", uuid=self.uuid)

        for i, it in enumerate(self.queue_tts):
            source_dur = it['source_duration']
            dubb_dur = it['dubb_time']
            if dubb_dur<=0 or source_dur<=0:
                continue
            audio_target = dubb_dur


            mode_log = f"[dubbing subtitles]{i=}"
            if dubb_dur > source_dur:
                self.audio_data.append({
                    "filename": it['filename'],
                    "dubb_time": dubb_dur,
                    "target_time": source_dur # No limit, force acceleration to alignment
                })
            elif (
                self.stretch_short_max_ms > 0
                and dubb_dur < source_dur
            ):
                gap = source_dur - dubb_dur
                if gap > 0:
                    self.audio_slow_data.append({
                        "filename": it['filename'],
                        "dubb_time": dubb_dur,
                        "target_time": source_dur,
                        "max_slowdown_extra_ms": min(gap, self.stretch_short_max_ms),
                    })

            logger.debug(f"[Calc] Mode={mode_log} Line={it['line']} | Source={source_dur} Dubb={dubb_dur} -> TargetA={audio_target}")


    def _concat_audio_aligned(self):
        logger.debug('[Audio] Start aligning splicing...')

        audio_concat_list = []
        total_audio_duration = 0

        # Restore original timeline
        for i, it in enumerate(self.queue_tts):
            # Add leading silence
            if i == 0 and it['start_time']>0:
                audio_concat_list.append(self._create_silen_file(f"gap_{i}", it['start_time']))

            #Real dubbing duration
            if it.get('filename') and Path(it['filename']).exists():
                audio_concat_list.append(it['filename'])
                dubb_len = len(AudioSegment.from_file(it['filename']))
            else:
                audio_concat_list.append(self._create_silen_file(f"sub_{i}", it['source_duration']))
                dubb_len = it['source_duration']
            # If the real dubbing is shorter than the subtitle interval, add silence at the end
            if dubb_len<it['source_duration']:
                audio_concat_list.append(self._create_silen_file(f"end_{i}", it['source_duration']-dubb_len))
        self._exec_concat_audio(audio_concat_list)


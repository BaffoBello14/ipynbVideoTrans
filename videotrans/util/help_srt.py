#Convert ordinary text to legal srt string
import copy
import os,json,re
from datetime import timedelta
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR

def clean_text_for_srtdict(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\w\s,.?!;:"\'%，。！？；：“”‘’、\-\u4e00-\u9fff]', '', text,flags=re.I | re.S)

    text = re.sub(r'\s+([，；。！？])', r'\1', text,flags=re.I | re.S)
    text = re.sub(r'\s+([,;:.!?])', r'\1', text,flags=re.I | re.S)
    text = re.sub(r'([,;:.!?])(?=[A-Za-z0-9])', r'\1 ', text,flags=re.I | re.S)
    text = re.sub(r'\s+', ' ', text,flags=re.I | re.S)
    text = text.strip()
    return text


def process_text_to_srt_str(input_text: str):
    if is_srt_string(input_text):
        return input_text

    # Split the text into lists according to newline characters
    text_lines = [line.strip() for line in input_text.replace("\n", "").splitlines() if line.strip()]

    # Split lines larger than 50 characters
    text_str_list = []
    for line in text_lines:
        if len(line) > 50:
            # Split into multiple strings according to punctuation marks
            split_lines = re.split(r'[,.，。]', line)
            text_str_list.extend([l.strip() for l in split_lines if l.strip()])
        else:
            text_str_list.append(line)
    #Create a list of subtitle dictionary objects
    dict_list = []
    start_time_in_seconds = 0  #Initial time in seconds

    for i, text in enumerate(text_str_list, start=1):
        # Calculate the start time and end time (increase by 1s each time)
        start_time = ms_to_time_string(seconds=start_time_in_seconds)
        end_time = ms_to_time_string(seconds=start_time_in_seconds + 1)
        start_time_in_seconds += 1

        #Create subtitle dictionary object
        srt = f"{i}\n{start_time} --> {end_time}\n{text}"
        dict_list.append(srt)

    return "\n\n".join(dict_list)


# Determine whether it is an srt string
def is_srt_string(input_text):
    input_text = input_text.strip()
    if not input_text:
        return False

    # Split the text into lists according to newline characters
    text_lines = input_text.replace("\n", "").splitlines()
    if len(text_lines) < 3:
        return False

    # Regular expression: the first line should be 1 to 2 pure numbers
    first_line_pattern = r'^\d{1,2}$'

    # Regular expression: the second line conforms to the time format
    second_line_pattern = r'^\s*?\d{1,2}:\d{1,2}:\d{1,2}(\W\d+)?\s*-->\s*\d{1,2}:\d{1,2}:\d{1,2}(\W\d+)?\s*$'

    # If the first two lines meet the conditions, return the original string
    if not re.match(first_line_pattern, text_lines[0].strip()) or not re.match(second_line_pattern,
                                                                               text_lines[1].strip()):
        return False
    return True


# Remove special characters from translation results
def cleartext(text: str, remove_start_end=True):
    res_text = text.replace('&#39;', "").replace('&quot;', '').replace("\u200b", " ").strip()
    # Delete multiple consecutive punctuation marks and keep only one
    res_text = re.sub(r'([，。！？,.?]\s?){2,}', ',', res_text,flags=re.I | re.S)
    return res_text


def ms_to_time_string(*, ms=0, seconds=None, sepflag=','):
    # Calculate hours, minutes, seconds and milliseconds
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    return f"{hours:02}:{minutes:02}:{seconds:02}{sepflag}{milliseconds:03}"


# Convert the non-standard hour:minute:second,|.millisecond format into aa:bb:cc,ddd format
# eg 001:01:2,4500 01:54,14 Wait for processing
def format_time(s_time="", separate=','):
    if not s_time.strip():
        return f'00:00:00{separate}000'
    hou, min, sec, ms = 0, 0, 0, 0

    tmp = s_time.strip().split(':')
    if len(tmp) >= 3:
        hou, min, sec = tmp[-3].strip(), tmp[-2].strip(), tmp[-1].strip()
    elif len(tmp) == 2:
        min, sec = tmp[0].strip(), tmp[1].strip()
    elif len(tmp) == 1:
        sec = tmp[0].strip()

    if re.search(r',|\.', str(sec)):
        t = re.split(r',|\.', str(sec))
        sec = t[0].strip()
        ms = t[1].strip()
    else:
        ms = 0
    hou = f'{int(hou):02}'[-2:]
    min = f'{int(min):02}'[-2:]
    sec = f'{int(sec):02}'
    ms = f'{int(ms):03}'[-3:]
    return f"{hou}:{min}:{sec}{separate}{ms}"


def srt_str_to_listdict(srt_string):
    'Parse SRT subtitle strings to more precisely handle the relationship between number lines and time lines'
    srt_list = []
    time_pattern = r'\s?(\d+):(\d+):(\d+)([,.]\d+)?\s*?-{1,2}>\s*?(\d+):(\d+):(\d+)([,.]\d+)?\n?'
    lines = srt_string.splitlines()
    i = 0

    while i < len(lines):
        time_match = re.match(time_pattern, lines[i].strip())
        if time_match:
            # Parse timestamp
            start_time_groups = time_match.groups()[0:4]
            end_time_groups = time_match.groups()[4:8]

            def parse_time(time_groups):
                h, m, s, ms = time_groups
                ms = ms.replace(',', '').replace('.', '') if ms else "0"
                try:
                    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
                except (ValueError, TypeError):
                    return None

            start_time = parse_time(start_time_groups)
            end_time = parse_time(end_time_groups)

            if start_time is None or end_time is None:
                i += 1
                continue

            i += 1
            text_lines = []
            while i < len(lines):
                current_line = lines[i].strip()
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""  # Get the next line, or an empty string if there is none

                if re.match(time_pattern, next_line):  # Determine whether the next line is a time line
                    if re.fullmatch(r'\d+', current_line):  # If the current line is purely numeric, skip
                        i += 1
                        break
                    else:
                        if current_line:
                            text_lines.append(current_line)
                        i += 1
                        break

                if current_line:
                    text_lines.append(current_line)
                    i += 1
                else:
                    i += 1

            text = ('\n'.join(text_lines)).strip()
            text = re.sub(r'</?[a-zA-Z]+>', '', text.replace("\r", '').strip(),flags=re.I | re.S)
            text = re.sub(r'\n{2,}', '\n', text,flags=re.I | re.S).strip()
            it = {
                "line": len(srt_list) + 1,  # Subtitle index, converted to integer
                "start_time": int(start_time),
                "end_time": int(end_time),  #Start and end time
                "text": text if text else "",  # subtitle text
            }
            it['startraw'] = ms_to_time_string(ms=it['start_time'])
            it['endraw'] = ms_to_time_string(ms=it['end_time'])
            it["time"] = f"{it['startraw']} --> {it['endraw']}"
            srt_list.append(it)
        else:
            i += 1  # Skip non-time lines

    return srt_list


# Format the string or subtitle file content into a valid subtitle array object
# Format to valid srt format
def format_srt(content):
    result = []
    try:
        result = srt_str_to_listdict(content)
    except Exception as e:
        result = srt_str_to_listdict(process_text_to_srt_str(content))
    return result


# Convert srt file or legal srt string into dictionary object
def get_subtitle_from_srt(srtfile, *, is_file=True):
    def _readfile(file):
        content = ""
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except UnicodeDecodeError as e:
            try:
                with open(file, 'r', encoding='gbk') as f:
                    content = f.read().strip()
            except UnicodeDecodeError as e:
                logger.exception(e, exc_info=True)
                raise
        except BaseException:
            raise
        return content

    if is_file:
        content = _readfile(srtfile)
    else:
        content = srtfile.strip()

    if len(content) < 1:
        raise RuntimeError(f"The srt subtitles were not read. The file may be empty or the format does not conform to the SRT specification\n:{srtfile=}\n{content=}")
    result = format_srt(copy.copy(content))

    # Convert txt file to a subtitle
    if len(result) < 1:
        result = [
            {"line": 1,
             "start_time":0,
             "end_time":2000,
             "startraw":"00:00:00,000",
             "endraw":"00:00:02,000",
             "time": "00:00:00,000 --> 00:00:02,000",
             "text": "\n".join(content)}
        ]
    return result



# Get the srt subtitle string from the subtitle object
def get_srt_from_list(srt_list):
    txt = ""
    line = 0
    # it may contain a complete timestamp it['time'] 00:00:01,123 --> 00:00:12,345
    # Start and end timestamps it['startraw']=00:00:01,123 it['endraw']=00:00:12,345
    # Start and end millisecond values it['start_time']=126 it['end_time']=678
    for it in srt_list:
        #if not it.get('text','').strip():
        #    continue
        line += 1
        if "startraw" not in it:
            # There is a complete start and end timestamp string hour: minute: second, millisecond --> hour: minute: second, millisecond
            if 'time' in it:
                startraw, endraw = it['time'].strip().split(" --> ")
                startraw = format_time(startraw.strip().replace('.', ','), ',')
                endraw = format_time(endraw.strip().replace('.', ','), ',')
            elif 'start_time' in it and 'end_time' in it:
                # Existence of start and end millisecond values
                startraw = ms_to_time_string(ms=it['start_time'])
                endraw = ms_to_time_string(ms=it['end_time'])
            else:
                raise Exception(
                    tr("There is no time/startraw/start_time in the subtitle in any valid timestamp form."))
        else:
            # There are separate start and end hours: minutes: seconds, milliseconds strings
            startraw = it['startraw']
            endraw = it['endraw']

        txt += f"{line}\n{startraw} --> {endraw}\n{it['text']}\n\n"
    return txt


def set_ass_font(srtfile: str) -> str:
    from . import help_ffmpeg
    "Convert SRT to ASS and customize the style:\n    - Use the main style globally (Default)\n    - If the subtitle text contains '###', the text after '###' uses the sub-style (Bottom)\n    - Remove '###'"
    if not os.path.exists(srtfile) or os.path.getsize(srtfile) == 0:
        return os.path.basename(srtfile)

    # ---------- 1. Convert SRT to temporary SRT (replace newline characters) and call ffmpeg to generate ASS ----------
    srt_str = ""
    for it in get_subtitle_from_srt(srtfile, is_file=True):
        t = re.sub(r'\n|\\n', r'\\N', it['text'].strip())
        if t:
            srt_str += f'{it["line"]}\n{it["startraw"]} --> {it["endraw"]}\n{t}\n\n'
    edit_srt = srtfile[:-4] + '-edit.srt'
    with open(edit_srt, 'w', encoding='utf-8') as f:
        f.write(srt_str.strip())
    ass_file_path = f'{srtfile[:-3]}ass'
    help_ffmpeg.runffmpeg(['-y', '-i', edit_srt, ass_file_path])

    # ---------- 2. Read JSON style configuration ----------
    JSON_FILE = f'{ROOT_DIR}/videotrans/ass.json'
    if not os.path.exists(JSON_FILE):
        logger.debug(f"[set_ass_font] Warning: JSON configuration file does not exist:{JSON_FILE}, skip style replacement")
        return ass_file_path

    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            style = json.load(f)
    except Exception as e:
        logger.exception(f"[set_ass_font] Error: Unable to read or parse JSON file{JSON_FILE}: {e}", exc_info=True)
        return ass_file_path

    # ---------- 3. Build two Style rows: Default (main style) and Bottom (secondary style) ----------
    #Main style attributes (maintain original logic)
    default_style = (
        f"Style: {style.get('Name', 'Default')},"
        f"{style.get('Fontname', 'Arial')},"
        f"{style.get('Fontsize', 16)},"
        f"{style.get('PrimaryColour', '&H00FFFFFF&')},"
        f"{style.get('SecondaryColour', '&H00FFFFFF&')},"
        f"{style.get('OutlineColour', '&H00000000&')},"
        f"{style.get('BackColour', '&H00000000&')},"
        f"{style.get('Bold', 0)},"
        f"{style.get('Italic', 0)},"
        f"{style.get('Underline', 0)},"
        f"{style.get('StrikeOut', 0)},"
        f"{style.get('ScaleX', 100)},"
        f"{style.get('ScaleY', 100)},"
        f"{style.get('Spacing', 0)},"
        f"{style.get('Angle', 0)},"
        f"{style.get('BorderStyle', 1)},"
        f"{style.get('Outline', 1)},"
        f"{style.get('Shadow', 0)},"
        f"{style.get('Alignment', 2)},"
        f"{style.get('MarginL', 10)},"
        f"{style.get('MarginR', 10)},"
        f"{style.get('MarginV', 10)},"
        f"{style.get('Encoding', 1)}\n"
    )

    # Secondary style: inherit the main style, but use bottom-specific values for Fontsize and PrimaryColour
    bottom_fontsize = style.get('Bottom_Fontsize', 14)          #Default 14
    bottom_color = style.get('Bottom_PrimaryColour', '&H0000FFFF&')  #Default yellow
    
    bottom_bold = style.get('Bottom_Bold', 0)  # bold
    bottom_italic = style.get('Bottom_Italic', 0)  # Whether to italicize
    
    bottom_secondarycolour=style.get('Bottom_SecondaryColour', '&H00FFFFFF&')
    bottom_outlinecolour=style.get('Bottom_OutlineColour', '&H00000000&')
    bottom_backcolour=style.get('Bottom_BackColour', '&H00000000&')
    
    bottom_style = (
        f"Style: Bottom,"                               # Fixed name "Bottom"
        f"{style.get('Fontname', 'Arial')},"
        f"{bottom_fontsize},"
        f"{bottom_color},"
        f"{bottom_secondarycolour},"
        f"{bottom_outlinecolour},"
        f"{bottom_backcolour},"
        f"{bottom_bold},"
        f"{bottom_italic},"
        f"{style.get('Underline', 0)},"
        f"{style.get('StrikeOut', 0)},"
        f"{style.get('ScaleX', 100)},"
        f"{style.get('ScaleY', 100)},"
        f"{style.get('Spacing', 0)},"
        f"{style.get('Angle', 0)},"
        f"{style.get('BorderStyle', 1)},"
        f"{style.get('Outline', 1)},"
        f"{style.get('Shadow', 0)},"
        f"{style.get('Alignment', 2)},"
        f"{style.get('MarginL', 10)},"
        f"{style.get('MarginR', 10)},"
        f"{style.get('MarginV', 10)},"
        f"{style.get('Encoding', 1)}\n"
    )

    # ---------- 4. Read the ASS file and replace the [V4+ Styles] block ----------
    try:
        with open(ass_file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except Exception as e:
        logger.exception(f"[set_ass_font] Error: Unable to read ASS file:{e}", exc_info=True)
        return ass_file_path

    # Match the [V4+ Styles] block, keep the Format line, and replace it with two Style lines
    pattern = r'(^\[V4\+ Styles\]\s*\r?\n' \
              r'Format:[^\r\n]*\r?\n' \
              r'(?:Style:[^\r\n]*\r?\n)*)' \
              r'(?=\[|$)'

    def replacer(match):
        # Extract the original Format line (if there is none, use the default format)
        format_line = None
        for line in match.group(0).splitlines():
            if line.strip().startswith("Format:"):
                format_line = line.strip() + "\n"
                break
        if not format_line:
            format_line = "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Return the complete [V4+ Styles] block, including Default and Bottom styles
        return f"[V4+ Styles]\n{format_line}{default_style}{bottom_style}"

    try:
        new_content, _ = re.subn(pattern, replacer, content, flags=re.MULTILINE)
    except Exception as e:
        logger.exception(f"[set_ass_font] Error: Regular replacement style failed:{e}", exc_info=True)
        return ass_file_path

    # ---------- 5. Process each Dialogue in [Events] and apply substyles to lines containing '###' ----------
    lines = new_content.splitlines(keepends=True)
    processed_lines = []
    inside_events = False
    dialogue_pattern = re.compile(r'^(Dialogue:.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,)(.*)$')

    for line in lines:
        # Check whether to enter the [Events] area
        if line.strip().startswith('[Events]'):
            inside_events = True
        elif line.strip().startswith('[') and inside_events:
            inside_events = False  # Exit when encountering next section

        if inside_events and line.startswith('Dialogue:'):
            match = dialogue_pattern.match(line.rstrip('\r\n'))
            if match:
                prefix = match.group(1)      # Fixed field in front
                text = match.group(2)        # Subtitle text content
                # Check if it contains '###'
                if '###' in text:
                    # Split into two parts: left (main language) and right (secondary language)
                    # Note: The text may contain \N newlines, but ### usually does not span lines
                    parts = text.split('###', 1)
                    left = parts[0]
                    right = parts[1] if len(parts) > 1 else ''
                    # Build new text: left part + secondary style switch + right part + restore main style
                    # Note: If the left part is empty, start directly with the sub-style; if the right part is empty, no style will be added (but in theory there should be content after ###)
                    new_text = ''
                    if left:
                        new_text += left
                    if right:
                        # Use {\rBottom} to switch to the secondary style, and use {\r} to return to Default when finished.
                        new_text += f'{{\\rBottom}}{right}{{\\r}}'
                    # Replace original line
                    line = f'{prefix}{new_text}\n'
            else:
                # Non-standard format, leave it as is
                pass
        processed_lines.append(line)

    #Write back ASS file
    try:
        with open(ass_file_path, 'w', encoding='utf-8-sig', newline='') as f:
            f.writelines(processed_lines)
    except Exception as e:
        logger.exception(f"[set_ass_font] Error: Unable to write to ASS file:{e}", exc_info=True)

    return ass_file_path
    

# Simple line break, no line breaks are retained, used for video translation subtitle embedding
def simple_wrap(text,maxlen=15,language="en"):
    # List of punctuation and spaces
    flag = [
        ",", ".", "?", "!", ";",
        "，", "。", "？", "；", "！", " "
    ]
    text=re.sub(r"\r?(\n|\\n)",' ',text,flags=re.I).strip()
    _len=len(text)
    if _len<maxlen+4:
        return text
    #If it is a language that does not require spaces such as Chinese, Japanese, Korean, Cantonese, etc.
    text_lilst=[]
    current_text=""
    offset=2 if language[:2] in ['zh','ja','ko','yue'] else 8
    maxlen=max(3,maxlen)
    offset=min(offset,maxlen//2)

    i=0
    while i <_len:
        current_text=current_text.lstrip()
        if i>=_len-offset:
            # If there are less than 4 characters at the end, all characters are given to the last line without distinction.
            current_text+=text[i:]
            # print(f'The last is less than 4 characters')
            break
        if len(current_text)<maxlen-offset:
            current_text+=text[i]
            i+=1
            # print('Normal append')
            continue
        #Judge whether i+1,i+2,i+3,i+4 conform to punctuation,
        if maxlen-offset<=len(current_text)<=maxlen and text[i] in flag:
            # This is currently punctuation and can be wrapped.
            current_text+=text[i]
            # print(f'Wrap between maxlen-offset and maxlen {text[i]=}')
            i+=1
            text_lilst.append(current_text)
            current_text=''
            continue
        # Then determine whether the next four characters meet the line break conditions.
        raw_i=i
        for next_i in range(1,offset+1):
            if text[i+next_i] in flag:
                pos_i=i+next_i+1
                current_text+=text[i:pos_i]
                # print(f'Change number at +offset {next_i=},{pos_i=},{text[i:pos_i]=}')
                raw_i=pos_i

                text_lilst.append(current_text)
                current_text=''
                break
        if raw_i!=i:
            i=raw_i
            continue
        # No suitable punctuation line break found, forced line break
        current_text+=text[i]
        if len(current_text)>=maxlen:
            # print(f'offset+4 was not found suitable, forcing a line break there, {len(current_text)=} {text[i]=}')
            text_lilst.append(current_text)
            current_text=''
        i+=1

    if current_text and len(current_text)<maxlen/3:
        text_lilst[-1]+=current_text
    elif current_text:
        text_lilst.append(current_text)
    # print(f'{maxlen=},{offset=}')
    return ("\n".join(text_lilst)).strip()

def textwrap(text, maxlen=15):
    '0. If the text length is less than maxlen, it will be returned directly.\n    1. text removes all line breaks beforehand.\n    2. When maxlen is reached, if the current character is a punctuation mark, group it here. Otherwise, search up to 4 characters backward,\n       Group at the first punctuation found. If neither is found, hard split at maxlen.\n    3. If the number of groups is greater than 1 and the length of the last group is less than 3, merge the last group into the previous group.\n    4. Finally, all groups are connected using newline characters and returned.\n\n    Args:\n      text: The input string to be processed.\n      maxlen: The target maximum length of each group, default is 15.\n\n    Returns:\n      A processed, newline-concatenated string.'
    # List of punctuation and spaces
    flag = [
        ",", ".", "?", "!", ";",
        "，", "。", "？", "；", "！", " "
    ]

    # 1. Remove all newlines
    text_string = text.strip() #replace('\n', ' ').replace('\r', ' ').strip()

    # 0. If the text length is less than or equal to maxlen, return directly
    if len(text_string) <= maxlen:
        return text_string

    groups = []
    # Keep original newlines
    for text in re.split(r'\n|\\n',text_string):
        text=text.strip()
        if not text:
            continue
        cursor = 0
        text_len = len(text)
        if text_len<=maxlen:
            groups.append(text)
            continue

        while cursor < text_len:
            # If the remaining text is less than maxlen, all of them will be used as the last group.
            if text_len - cursor <= maxlen:
                groups.append(text[cursor:])
                break

            # 2. Intelligent grouping logic
            break_point = -1

            # Determine the range of searching for punctuation, starting from the maxlen position and looking back up to 4 characters
            # For example, maxlen=15, cursor=0, then search for characters with indexes 15, 16, 17
            search_range = range(max(cursor + maxlen-3,0), min(cursor + maxlen + 2, text_len))

            found_flag = False
            for i in search_range:
                if text[i] in flag:
                    # Find the punctuation point and set the breakpoint after the punctuation point
                    break_point = i + 1
                    found_flag = True
                    break

            # If no punctuation is found within the search range, hard split at maxlen
            if not found_flag:
                break_point = cursor + maxlen

            groups.append(text[cursor:break_point])
            cursor = break_point

    # 3. If the group is greater than 1 and the length of the last group is less than 3, merge
    if len(groups) > 1 and len(groups[-1]) < 3:
        groups[-2] += groups[-1]
        groups.pop()

    return ("\n".join(groups)).strip()

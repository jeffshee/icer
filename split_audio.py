import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')

from os.path import join, basename
from pydub import AudioSegment

from edit_audio import trim_audio


def optimized_segment_audio(input_path, output_dir, max_duration_sec=60):
    from inaSpeechSegmenter import Segmenter
    from pydub.silence import detect_silence
    max_duration = max_duration_sec * 1000

    def silence_based_split(audio_segment, min_silence_len=500, silence_thresh=-16, seek_step=1, result_offset=0):
        # slightly drop the threshold until silence is detected
        silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
        delta = 0
        while len(silent_ranges) == 0 and delta < 30:
            delta += 5
            silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh - delta, seek_step)

        len_seg = len(audio_segment)

        # if there is no silence, the whole thing is nonsilent
        if not silent_ranges:
            return [('nonsilence', result_offset, result_offset + len_seg)]

        # short circuit when the whole audio segment is silent
        if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
            return [('silence', result_offset, result_offset + len_seg)]

        prev_end_i = 0
        nonsilent_ranges = []
        for start_i, end_i in silent_ranges:
            nonsilent_ranges.append([prev_end_i, start_i])
            prev_end_i = end_i

        if end_i != len_seg:
            nonsilent_ranges.append([prev_end_i, len_seg])

        if nonsilent_ranges[0] == [0, 0]:
            nonsilent_ranges.pop(0)

        result = []
        for start, end in silent_ranges:
            result += [('silence', result_offset + start, result_offset + end)]
        for start, end in nonsilent_ranges:
            result += [('nonsilence', result_offset + start, result_offset + end)]
        return sorted(result, key=lambda x: x[1])

    def check_overflow(split_list):
        total_overflow = 0
        count = 0
        for start, end in split_list:
            if end - start > max_duration:
                total_overflow += (end - start) - max_duration
                count += 1
        return count, total_overflow

    def get_init_split():
        seg = Segmenter(vad_engine='smn', detect_gender=False)
        segmentation = seg(input_path)

        # convert sec to msec
        temp = []
        for result, start, end in segmentation:
            temp += [(result, start * 1000, end * 1000)]
        segmentation = temp

        # check whether if any segment is over max_duration,
        # if yes then split that segment further by using silence_based_split
        temp = []
        for i, (result, start, end) in enumerate(segmentation):
            if end - start > max_duration:
                overflow_part = AudioSegment.from_wav(input_path)[start:end]
                temp += silence_based_split(overflow_part, silence_thresh=overflow_part.dBFS - 10, result_offset=start)
            else:
                temp += [(result, start, end)]
        segmentation = temp

        split_list = []
        prev_point = 0
        for result, start, end in segmentation:
            if result != 'speech' and result != 'nonsilence':
                mid_point = start + (end - start) / 2
                split_list += [(prev_point, mid_point)]
                prev_point = mid_point
        split_list += [(prev_point, segmentation[-1][2])]

        return split_list

    def remove_split_at(split_list, index):
        front = split_list[:index] if index > 0 else []
        back = split_list[index + 2:] if index + 2 < len(split_list) else []
        new_split = front + [(split_list[index][0], split_list[index + 1][1])] + back
        return new_split

    def optimized_split():
        split = get_init_split()
        init_count, init_total_overflow = check_overflow(split)
        i = 0
        while i < len(split) - 1:
            new_split = remove_split_at(split, i)
            new_count, new_total_overflow = check_overflow(new_split)
            if new_count == init_count and not new_total_overflow > init_total_overflow:
                split = new_split
                i = 0
            i += 1
        return split

    output_path_list = []
    origin_filename = basename(input_path)[:-4]  # remove extension
    for i, (start, end) in enumerate(optimized_split()):
        trim_ms_range = (start, end)
        output_path = join(output_dir, '{}_seg_{:03d}.wav'.format(origin_filename, i))
        output_path_list.append(output_path)
        trim_audio(input_path, output_path, trim_ms_range)
        i += 1
    return output_path_list

# class Counter:
#     def __init__(self, low, high):
#         self.low = low
#         self.current = low - 1
#         self.high = high
#
#     def __len__(self):
#         return self.high - self.low
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):  # Python 2: def next(self)
#         self.current += 1
#         if self.current < self.high:
#             return self.current
#         raise StopIteration

import numpy as np
import cv2
import math

from utils.video_utils import read_video_capture

class VideoIterator:
    """
    A class for getting video frames using cv2
    """

    def __init__(self, video_path: str, start_frame=0, end_frame=-1):
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.current = start_frame - 1
        if end_frame == -1:
            self.end_frame = self.frame_count
        assert self.start_frame >= 0
        assert self.end_frame <= self.frame_count
        assert self.start_frame < self.end_frame

    def __len__(self):
        frame_num = math.ceil((self.end_frame - self.start_frame) / (self.skip_frame + 1))
        if self.drop_last:
            return frame_num // self.batch_size
        else:
            return math.ceil(frame_num / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        batch = []
        length = self.__len__()
        if self.current < length:
            size_of_batch = (self.end_frame - self.start_frame) // length
            low = self.current * size_of_batch
            high = (self.current + 1) * size_of_batch + 1
            if high > self.frame_count:
                high = self.frame_count
            for position in range(low, high):
                # Set current position
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ret, frame = read_video_capture(self.video_capture, self.offset)
                assert ret
                batch.append(frame)
            return np.array(batch)
        raise StopIteration

# class VideoIterator:
#     """
#     A class for getting video frames using cv2
#     """
#
#     def __init__(self, video_path: str, start_frame=0, end_frame=-1, k_resolution=-1, skip_frame=0, batch_size=8,
#                  drop_last=False, roi=None, offset=0):
#         self.video_capture = cv2.VideoCapture(video_path)
#         self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.start_frame = start_frame
#         self.end_frame = end_frame
#         self.skip_frame = skip_frame
#         self.drop_last = drop_last
#         self.batch_size = batch_size
#         self.offset = offset
#         self.current = start_frame - 1
#         if end_frame == -1:
#             self.end_frame = self.frame_count
#         assert self.start_frame >= 0
#         assert self.end_frame <= self.frame_count
#         assert self.start_frame < self.end_frame
#         assert skip_frame >= 0
#
#     def __len__(self):
#         frame_num = math.ceil((self.end_frame - self.start_frame) / (self.skip_frame + 1))
#         if self.drop_last:
#             return frame_num // self.batch_size
#         else:
#             return math.ceil(frame_num / self.batch_size)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         self.current += 1
#         batch = []
#         length = self.__len__()
#         if self.current < length:
#             size_of_batch = (self.end_frame - self.start_frame) // length
#             low = self.current * size_of_batch
#             high = (self.current + 1) * size_of_batch + 1
#             if high > self.frame_count:
#                 high = self.frame_count
#             for position in range(low, high):
#                 # Set current position
#                 self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
#                 ret, frame = read_video_capture(self.video_capture, self.offset)
#                 assert ret
#                 batch.append(frame)
#             return np.array(batch)
#         raise StopIteration


# c = Counter(3, 9)
# print(len(c))
# for i in c:
#     print(i)
video_path = "/home/jeffshee/Developer/icer/datasets/200225_expt22/video.mp4"
print(len(VideoIterator(video_path)))

for frame in VideoIterator(video_path):
    print(frame.shape)

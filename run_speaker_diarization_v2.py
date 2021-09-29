from speaker_diarization_v2.speakerDiarization import main
import time
import datetime


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if "log_time" in kwargs:
            name = kwargs.get("log_name", method.__name__.upper())
            kwargs["log_time"][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__} elapsed time: {datetime.timedelta(seconds=(te - ts))} (H:MM:SS)")
        return result

    return timed


@timeit
def run_speaker_diarization(path_audio, sound_name, path_diarization_result, people_num, embedding_per_second=0.5,
                            overlap_rate=0.5, noise_class_num=3, use_spectral_cluster=False):
    """
    The final result is influenced by the size of each window and the overlap rate.
    When the overlap is too large, the uis-rnn perhaps generates fewer speakers since the speaker embeddings changed smoothly, otherwise will generate more speakers.
    And also, the window size cannot be too short, it must contain enough information to generate more discrimitive speaker embeddings.
    """
    path_sound_file = path_audio + sound_name + ".wav"
    main(path_sound_file, path_diarization_result, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate,
         use_spectral_cluster=use_spectral_cluster, min_clusters=people_num, max_clusters=people_num + noise_class_num)


if __name__ == '__main__':
    run_speaker_diarization("audio/", "Take01", "diarization/{}_{}_{}/".format("Take01", 0.5, 0.5), 5)

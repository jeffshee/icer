from speaker_diarization_v2.speakerDiarization import main


def run_speaker_diarization(path_audio, sound_name, path_diarization_result, people_num, embedding_per_second=0.5, overlap_rate=0.5, noise_class_num=3):
    """
    The final result is influenced by the size of each window and the overlap rate.
    When the overlap is too large, the uis-rnn perhaps generates fewer speakers since the speaker embeddings changed smoothly, otherwise will generate more speakers.
    And also, the window size cannot be too short, it must contain enough information to generate more discrimitive speaker embeddings.
    """
    path_sound_file = path_audio + sound_name + ".wav"
    main(path_sound_file, path_diarization_result, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate,
         min_clusters=people_num, max_clusters=people_num + noise_class_num)


if __name__ == '__main__':
    path_diarization_result = "diarization/{}_{}_{}/".format("Take01", 0.5, 0.5)
    run_speaker_diarization("audio/", "Take01", path_diarization_result, 5)

"""
Registry of dataset mixtures for multi-dataset training.

Each mixture maps a name to a list of (dataset_name, sampling_weight) tuples.
The dataset_name is joined with data_root_dir at runtime to form the full path.
"""

DATASET_MIXTURES: dict[str, list[tuple[str, float]]] = {
    "yupu_test": [
        ("0_30_s_academic_v0_1", 1.0),
    ],
    "ego1_multiimage": [
        ("agibot_world", 1.0),
        ("EgoExoLearn", 1.0),
        ("GenEgoData", 1.0),
        ("something_something_v2", 1.0),
        ("hoi4d", 1.0),
        ("em_interaction", 1.0),
        ("ego4d", 1.0),
        ("epic_kitchens", 1.0),
        ("egocentric10k", 1.0),
        ("egodex", 1.0),
        ("kinetics", 1.0),
        ("llava-178k", 1.0),
        ("pe_video_extended", 1.0),
        ("pe_video_traintest", 1.0),
        ("psi_ego", 1.0),
    ],
    "ego1": [
        ("EgoExoLearn_1fps", 1.0),
        ("GenEgoData", 1.0),
        ("0_30_s_nextqa", 1.0),
        ("0_30_s_academic_v0_1", 1.0),
        ("0_30_s_activitynetqa", 1.0),
        ("0_30_s_youtube_v0_1", 1.0),
        ("1_2_m_academic_v0_1", 1.0),
        ("1_2_m_activitynetqa", 1.0),
        ("1_2_m_nextqa", 1.0),
        ("1_2_m_youtube_v0_1", 1.0),
        ("2_3_m_academic_v0_1", 1.0),
        ("2_3_m_activitynetqa", 1.0),
        ("2_3_m_nextqa", 1.0),
        ("2_3_m_youtube_v0_1", 1.0),
        ("30_60_s_academic_v0_1", 1.0),
        ("30_60_s_activitynetqa", 1.0),
        ("30_60_s_nextqa", 1.0),
        ("30_60_s_youtube_v0_1", 1.0),
        ("egodex_1fps_refreshed", 1.0),
        ("something_something_v2", 1.0)
    ],
}

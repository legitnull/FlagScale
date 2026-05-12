import os
import re
import json

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

vlm_data_root = os.environ.get("VLM_DATA_ROOT", "/share/project/wangyihao/starVLA/playground/Datasets/VLM_DATA")
video_data_root = os.environ.get("VLM_VIDEO_ROOT", "/share/project/mycao/web_video_train")
image_root = os.environ.get("VLM_IMAGE_ROOT", "/share/project/mycao/robobrain_v2_5_sz")

HONEYDATA = {
    "annotation_path": f"{vlm_data_root}/honey_data_rel_1071666_qwen35_tokenlen1024.json",
    "data_path": f"{image_root}/",
}

LLAVAV15_LRV = {
    "annotation_path": f"{vlm_data_root}/llava_v1_5_lrv_mix873k_wo_vg_full.json",
    "data_path": f"{image_root}/",
}

ACTIVITYNETQA_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_activitynetqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_activitynetqa/",
}

NEXTQA_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_nextqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_nextqa/",
}

PERCEPTIONTEST_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_perceptiontest_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_perceptiontest/",
}

YOUTUBE_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_youtube_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_youtube_v0_1/",
}

ACADEMIC_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_academic_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_academic_v0_1/",
}

ACTIVITYNETQA_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_activitynetqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_activitynetqa/",
}

NEXTQA_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_nextqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_nextqa/",
}

YOUTUBE_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_youtube_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_youtube_v0_1/",
}

ACADEMIC_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_academic_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_academic_v0_1/",
}

ACTIVITYNETQA_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_activitynetqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_activitynetqa/",
}

NEXTQA_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_nextqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_nextqa/",
}

YOUTUBE_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_youtube_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_youtube_v0_1/",
}

ACADEMIC_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_academic_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_academic_v0_1/",
}

ACTIVITYNETQA_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_activitynetqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_activitynetqa/",
}

NEXTQA_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_nextqa_oe_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_nextqa/",
}

PERCEPTIONTEST_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_perceptiontest_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_perceptiontest/",
}

YOUTUBE_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_youtube_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_youtube_v0_1/",
}
###############################################################################
# 新增：0_30_s_academic (oe) + 所有 mc 数据集
###############################################################################

ACADEMIC_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_academic_oe_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_academic_v0_1/",
}

ACADEMIC_MC_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_academic_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_academic_v0_1/",
}

NEXTQA_MC_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_nextqa_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_nextqa/",
}

YOUTUBE_MC_0_30_S = {
    "annotation_path": f"{vlm_data_root}/0_30_s_youtube_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/0_30_s_youtube_v0_1/",
}

ACADEMIC_MC_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_academic_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_academic_v0_1/",
}

NEXTQA_MC_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_nextqa_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_nextqa/",
}

YOUTUBE_MC_1_2_M = {
    "annotation_path": f"{vlm_data_root}/1_2_m_youtube_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/1_2_m_youtube_v0_1/",
}

ACADEMIC_MC_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_academic_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_academic_v0_1/",
}

NEXTQA_MC_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_nextqa_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_nextqa/",
}

YOUTUBE_MC_2_3_M = {
    "annotation_path": f"{vlm_data_root}/2_3_m_youtube_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/2_3_m_youtube_v0_1/",
}

ACADEMIC_MC_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_academic_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_academic_v0_1/",
}

NEXTQA_MC_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_nextqa_mc_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_nextqa/",
}

YOUTUBE_MC_30_60_S = {
    "annotation_path": f"{vlm_data_root}/30_60_s_youtube_mc_v0_1_qa_processed.filtered.json",
    "data_path": f"{video_data_root}/30_60_s_youtube_v0_1/",
}

CAMBRIAN_S_3M = {
    "annotation_path": f"{vlm_data_root}/cambrian_s_3m_dir.filtered.jsonl",
    "data_path": f"{video_data_root}/CambrianS-3M/",
}

CAMBRIAN_10M = {
    "annotation_path": f"{vlm_data_root}/Cambrian7M_prompt_unique.filtered.jsonl",
    "data_path": f"{video_data_root}/Cambrian-10M/",
}

VSI_590K = {
    "annotation_path": f"{vlm_data_root}/vsi_590k_video_374148.filtered.jsonl",
    "data_path": f"{video_data_root}/VSI_590K/",
}

VSI_590K_SPATIAL = {
    "annotation_path": f"{vlm_data_root}/vsi_590k_image_216519.filtered.jsonl",
    "data_path": f"{video_data_root}/VSI_590K/",
}

LLAVA_HOUND = {
    "annotation_path": f"{vlm_data_root}/llavahound.filtered.jsonl",
    "data_path": f"{video_data_root}/llava-hound/",
}

data_dict = {
    "honeydata": HONEYDATA,
    "llavav15_lrv": LLAVAV15_LRV,
    "activitynetqa_0_30_s": ACTIVITYNETQA_0_30_S,
    "nextqa_0_30_s": NEXTQA_0_30_S,
    "perceptiontest_0_30_s": PERCEPTIONTEST_0_30_S,
    "youtube_0_30_s": YOUTUBE_0_30_S,
    "academic_1_2_m": ACADEMIC_1_2_M,
    "activitynetqa_1_2_m": ACTIVITYNETQA_1_2_M,
    "nextqa_1_2_m": NEXTQA_1_2_M,
    "youtube_1_2_m": YOUTUBE_1_2_M,
    "academic_2_3_m": ACADEMIC_2_3_M,
    "activitynetqa_2_3_m": ACTIVITYNETQA_2_3_M,
    "nextqa_2_3_m": NEXTQA_2_3_M,
    "youtube_2_3_m": YOUTUBE_2_3_M,
    "academic_30_60_s": ACADEMIC_30_60_S,
    "activitynetqa_30_60_s": ACTIVITYNETQA_30_60_S,
    "nextqa_30_60_s": NEXTQA_30_60_S,
    "perceptiontest_30_60_s": PERCEPTIONTEST_30_60_S,
    "youtube_30_60_s": YOUTUBE_30_60_S,
    "academic_0_30_s": ACADEMIC_0_30_S,
    "academic_mc_0_30_s": ACADEMIC_MC_0_30_S,
    "nextqa_mc_0_30_s": NEXTQA_MC_0_30_S,
    "youtube_mc_0_30_s": YOUTUBE_MC_0_30_S,
    "academic_mc_1_2_m": ACADEMIC_MC_1_2_M,
    "nextqa_mc_1_2_m": NEXTQA_MC_1_2_M,
    "youtube_mc_1_2_m": YOUTUBE_MC_1_2_M,
    "academic_mc_2_3_m": ACADEMIC_MC_2_3_M,
    "nextqa_mc_2_3_m": NEXTQA_MC_2_3_M,
    "youtube_mc_2_3_m": YOUTUBE_MC_2_3_M,
    "academic_mc_30_60_s": ACADEMIC_MC_30_60_S,
    "nextqa_mc_30_60_s": NEXTQA_MC_30_60_S,
    "youtube_mc_30_60_s": YOUTUBE_MC_30_60_S,
    "cambrian_s_3m": CAMBRIAN_S_3M,
    "cambrian_10m": CAMBRIAN_10M,
    "vsi_590k": VSI_590K,
    "vsi_590k_spatial": VSI_590K_SPATIAL,
    "llava_hound": LLAVA_HOUND,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def _prefer_clean_annotation(annotation_path: str) -> str:
    """如果存在对应的 .clean.json 文件，则优先使用它（由 filter_bad_videos.py 生成）。"""
    clean_path = annotation_path.replace(".json", ".clean.json").replace(".jsonl", ".clean.jsonl")
    if os.path.exists(clean_path):
        return clean_path
    return annotation_path


def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["annotation_path"] = _prefer_clean_annotation(config["annotation_path"])
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    def _count_annotation_items(annotation_path: str) -> int:
        suffix = Path(annotation_path).suffix.lower()
        if suffix == ".jsonl":
            n = 0
            with open(annotation_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
            return n
        if suffix == ".json":
            with open(annotation_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return len(obj)
            if isinstance(obj, dict):
                # tolerate common formats:
                # - {"data": [...]}
                # - {"annotations": [...]}
                for k in ("data", "annotations", "items"):
                    if k in obj and isinstance(obj[k], list):
                        return len(obj[k])
                return len(obj)
            raise TypeError(f"Unsupported json root type: {type(obj)} in {annotation_path}")
        raise ValueError(f"Unsupported annotation file suffix: {suffix} ({annotation_path})")

    total = 0
    per_dataset = {}
    for name, cfg in data_dict.items():
        ann = cfg.get("annotation_path", "")
        if not ann:
            per_dataset[name] = 0
            continue
        try:
            n = _count_annotation_items(ann)
        except Exception as e:
            print(f"[CountError] dataset={name} annotation_path={ann} error={e}")
            n = 0
        per_dataset[name] = n
        total += n

    print("=== Annotation counts (data_dict) ===")
    for name in sorted(per_dataset.keys()):
        print(f"{name}\t{per_dataset[name]}")
    print(f"TOTAL\t{total}")
    

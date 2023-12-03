# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

import csv
import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import get_file_name, get_file_name_with_ext, get_file_size
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "CelebFaces Attributes"
    images_path = "/mnt/d/datasetninja-raw/img_celeba/img_celeba.7z/img_celeba"
    ds_name = "ds"
    batch_size = 30

    identities_data = "/mnt/d/datasetninja-raw/img_celeba/identity_CelebA.txt"
    tags_data = "/mnt/d/datasetninja-raw/img_celeba/list_attr_celeba.txt"
    bboxes_data = "/mnt/d/datasetninja-raw/img_celeba/list_bbox_celeba.txt"
    landmarks_data = "/mnt/d/datasetninja-raw/img_celeba/list_landmarks_celeba.txt"

    def create_ann(image_path):
        labels = []
        tags = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name_with_ext(image_path)

        tags_data = name_to_tags[image_name]
        for idx, tag_val in enumerate(tags_data):
            if tag_val == "1":
                curr_meta = meta.get_tag_meta(tags_names[idx])
                tag = sly.Tag(curr_meta)
                tags.append(tag)

        if name_to_identity.get(get_file_name_with_ext(image_path)) is not None:
            val = name_to_identity[get_file_name_with_ext(image_path)]
            curr_meta = meta.get_tag_meta("identity")
            tags.append(sly.Tag(curr_meta, int(val)))

        bbox_data = name_to_bbox[image_name]
        left = int(bbox_data[0])
        top = int(bbox_data[1])
        right = left + int(bbox_data[2])
        bottom = top + int(bbox_data[3])
        rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
        label = sly.Label(rect, face)
        labels.append(label)

        landmarks_data = name_to_landmarks[image_name]
        point_l_eye = sly.Point(int(landmarks_data[1]), int(landmarks_data[0]))
        label = sly.Label(point_l_eye, l_eye)
        labels.append(label)
        point_r_eye = sly.Point(int(landmarks_data[3]), int(landmarks_data[2]))
        label = sly.Label(point_r_eye, r_eye)
        labels.append(label)
        point_nose = sly.Point(int(landmarks_data[5]), int(landmarks_data[4]))
        label = sly.Label(point_nose, nose)
        labels.append(label)
        point_l_mouse = sly.Point(int(landmarks_data[7]), int(landmarks_data[6]))
        label = sly.Label(point_l_mouse, l_mouth)
        labels.append(label)
        point_r_mouse = sly.Point(int(landmarks_data[9]), int(landmarks_data[8]))
        label = sly.Label(point_r_mouse, r_mouth)
        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    face = sly.ObjClass("face", sly.Rectangle)
    l_eye = sly.ObjClass("left eye", sly.Point)
    r_eye = sly.ObjClass("right eye", sly.Point)
    nose = sly.ObjClass("nose", sly.Point)
    l_mouth = sly.ObjClass("left mouth", sly.Point)
    r_mouth = sly.ObjClass("right mouth", sly.Point)

    tags_names = [
        "5_o_clock_shadow",
        "arched_eyebrows",
        "attractive",
        "bags_under_eyes",
        "bald",
        "bangs",
        "big_lips",
        "big_nose",
        "black_hair",
        "blond_hair",
        "blurry",
        "brown_hair",
        "bushy_eyebrows",
        "chubby",
        "double_chin",
        "eyeglasses",
        "goatee",
        "gray_hair",
        "heavy_makeup",
        "high_cheekbones",
        "male",
        "mouth_slightly_open",
        "mustache",
        "narrow_eyes",
        "no_beard",
        "oval_face",
        "pale_skin",
        "pointy_nose",
        "receding_hairline",
        "rosy_cheeks",
        "sideburns",
        "smiling",
        "straight_hair",
        "wavy_hair",
        "wearing_earrings",
        "wearing_hat",
        "wearing_lipstick",
        "wearing_necklace",
        "wearing_necktie",
        "young",
    ]

    tag_identity = sly.TagMeta("id", sly.TagValueType.ANY_NUMBER)

    tag_metas = [sly.TagMeta(tag_name, sly.TagValueType.NONE) for tag_name in tags_names]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[face, l_eye, r_eye, nose, l_mouth, r_mouth],
        tag_metas=[tag_identity] + tag_metas,
    )
    api.project.update_meta(project.id, meta.to_json())

    name_to_identity = {}
    name_to_tags = {}
    name_to_bbox = {}
    name_to_landmarks = {}

    with open(identities_data) as f:
        content = f.read().split("\n")
        for idx, row in enumerate(content):
            if idx in [202599]:
                continue
            name, identity = row.split(" ")
            name_to_identity[name] = identity

    with open(tags_data) as f:
        content = f.read().split("\n")
        for idx, row in enumerate(content):
            if idx in [0, 1]:
                continue
            if len(row) == 0:
                continue
            row = row.split(" ")
            row = list(filter(None, row))
            name_to_tags[row[0].lower()] = row[1:]

    with open(bboxes_data) as f:
        content = f.read().split("\n")
        for idx, row in enumerate(content):
            if idx in [0, 1]:
                continue
            if len(row) == 0:
                continue
            row = row.split(" ")
            row = list(filter(None, row))
            name_to_bbox[row[0]] = row[1:]

    with open(landmarks_data) as f:
        content = f.read().split("\n")
        for idx, row in enumerate(content):
            if idx in [0, 1]:
                continue
            if len(row) == 0:
                continue
            row = row.split(" ")
            row = list(filter(None, row))
            name_to_landmarks[row[0]] = row[1:]

    # with open(tags_data, "r") as file:
    #     csvreader = csv.reader(file)
    #     for idx, row in enumerate(csvreader):
    #         if idx == 0:
    #             continue
    #         name_to_tags[row[0]] = row[1:]

    # with open(bboxes_data, "r") as file:
    #     csvreader = csv.reader(file)
    #     for idx, row in enumerate(csvreader):
    #         if idx == 0:
    #             continue
    #         name_to_bbox[row[0]] = row[1:]

    # with open(landmarks_data, "r") as file:
    #     csvreader = csv.reader(file)
    #     for idx, row in enumerate(csvreader):
    #         if idx == 0:
    #             continue
    #         name_to_landmarks[row[0]] = row[1:]

    all_images_names = list(name_to_tags.keys())
    train_names = all_images_names[:162771]
    val_names = all_images_names[162771:182637]
    test_names = all_images_names[182638:]

    ds_name_to_names = {"train": train_names, "val": val_names, "test": test_names}

    for ds_name, images_names in ds_name_to_names.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project

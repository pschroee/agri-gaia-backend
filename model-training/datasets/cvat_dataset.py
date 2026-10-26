#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
# SPDX-FileContributor: Philipp Schröer
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import os
import shutil
import logging
import xmltodict

from pathlib import Path
from operator import itemgetter
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from config import load_config
from dataset import (
    DATASETS_ROOT_PATH,
    DATA_DIRS,
    get_dataset_path,
    load_dataset_filepaths,
)
from collections import defaultdict
from shapely import geometry

ANNOTATIONS_SUBFILEPATH = "annotations/annotations.xml"


def _get_annotations_filepath() -> Path:
    return get_dataset_path().joinpath(ANNOTATIONS_SUBFILEPATH)


def _get_tasks(annotation_file) -> List[Dict]:
    meta = annotation_file["meta"]
    assert "task" in meta, "No tasks present in annotations file."
    tasks = meta["task"]
    if type(tasks) != list:
        return [tasks]
    return tasks


def create_images(images_path: Path, files_with_annotations: Dict[Path, Dict]) -> None:
    shutil.rmtree(images_path)
    os.makedirs(images_path)
    for filepath in files_with_annotations.keys():
        images_path.joinpath(filepath.name).symlink_to(filepath)


def create_train_images(train_data: Dict[Path, Dict]) -> None:
    create_images(DATA_DIRS["images_train"], files_with_annotations=train_data)


def create_test_images(test_data: Dict[Path, Dict]) -> None:
    create_images(DATA_DIRS["images_val"], files_with_annotations=test_data)


def create_train_labels(
    create_labels_fn, train_data: Dict[Path, Dict], label_names: List[str]
) -> None:
    create_labels_fn(
        DATA_DIRS["labels_train"], label_names, files_with_annotations=train_data
    )


def create_test_labels(
    create_labels_fn, test_data: Dict[Path, Dict], label_names: List[str]
) -> None:
    create_labels_fn(
        DATA_DIRS["labels_val"], label_names, files_with_annotations=test_data
    )


def create_directory_structure() -> None:
    for dir_name in ["images", "labels"]:
        for subdir_name in ["train", "val"]:
            path = Path(os.path.join(DATASETS_ROOT_PATH, dir_name, subdir_name))
            path.mkdir(parents=True, exist_ok=True)
            DATA_DIRS[f"{dir_name}_{subdir_name}"] = path


def load_annotations_file() -> Dict:
    annotations_filepath = _get_annotations_filepath()
    with open(annotations_filepath, "r") as fh:
        annotations_file = xmltodict.parse(fh.read())
        assert (
            "annotations" in annotations_file
        ), "No annotations present. Likely incompatible annotations file."
        return annotations_file["annotations"]


def image_has_annotations(image: Dict) -> bool:
    # Supported CVAT annotation types.
    # See: https://opencv.github.io/cvat/docs/manual/advanced/xml_format/#annotation
    annotation_types = {"box", "polygon", "polyline", "points", "skeleton", "tag"}
    return any(annotation_type in image for annotation_type in annotation_types)


def find_annotations_for_files(
    filepaths: List[Path], annotations: List[Dict], strict: bool = False
) -> Dict[Path, Dict]:
    files_with_annotations = {}
    for filepath in filepaths:
        for image in annotations:
            if filepath.name == Path(image["@name"]).name:
                if image_has_annotations(image):
                    files_with_annotations[filepath] = image
                else:
                    print(f"Skipping file '{filepath}' without annotations.")
    if strict:
        assert len(filepaths) == len(
            files_with_annotations
        ), "Missing annotations for files."
    return files_with_annotations


def get_label_names(annotation_file: Dict) -> List[str]:
    tasks: List[Dict] = _get_tasks(annotation_file)
    label_names = []
    for task in tasks:
        if "labels" not in task or "label" not in task["labels"]:
            continue
        labels = task["labels"]["label"]
        if type(labels) != list:
            label_names.append(labels["name"])
        else:
            label_names += [label["name"] for label in labels]
    assert label_names, "No labels present in annotations file."
    # Keep the order of the annotations file: the class index of every template is
    # the position in this list, and a set has no stable order across processes.
    return list(dict.fromkeys(label_names))


def get_annotations(annotation_file: Dict) -> List[Dict]:
    assert (
        "image" in annotation_file
    ), "No image annotations present in annotations file."
    images = annotation_file["image"]
    if type(images) != list:
        return [images]
    return images


def get_image_annotations(image: Dict, annotation_type: str) -> List[Dict]:
    if annotation_type not in image:
        return []
    annotations = image[annotation_type]
    if type(annotations) != list:
        return [annotations]
    return annotations


def split_dataset(
    files_with_annotations: Dict[Path, Dict], train_size: float, test_size: float
) -> Tuple[Dict[Path, Dict], Dict[Path, Dict]]:
    train_filepaths, test_filepaths = train_test_split(
        list(files_with_annotations.keys()),
        train_size=train_size,
        test_size=test_size,
        random_state=42,
    )

    [train, test] = map(
        lambda keys: {x: files_with_annotations[x] for x in keys},
        [train_filepaths, test_filepaths],
    )

    return train, test


"""
See:
    https://cocodataset.org/#format-data
    https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#cocodetectiondataset
"""


def as_coco(
    label_names: List[str],
    files_with_annotations: Dict[Path, Dict],
    dataset_type="segmentation",
) -> Dict[str, List]:

    assert dataset_type in {
        "segmentation",
        "classification",
    }, f"Unknown COCO dataset type '{dataset_type}'."

    labels = defaultdict(list)

    # categories
    for label_id, label_name in enumerate(label_names, start=1):
        labels["categories"].append({"id": label_id, "name": label_name})

    annotation_id = 1
    annotation_type = (
        "polygon" if dataset_type == "segmentation"
        else "tag" if dataset_type == "classification"
        else "box"
    )

    # images
    for image_id, filepath in enumerate(files_with_annotations, start=1):
        image = files_with_annotations[filepath]

        # annotations
        annotations = get_image_annotations(image, annotation_type=annotation_type)

        # Do not include images without annotations of specific type
        if not annotations:
            print(
                f"Skipping image '{filepath}' without annotations of type '{annotation_type}'."
            )
            continue

        image_width, image_height = int(image["@width"]), int(image["@height"])

        labels["images"].append(
            {
                "id": image_id,
                "file_name": filepath.name,
                "height": image_height,
                "width": image_width,
            }
        )

        for annotation in annotations:
            try:
                category_id = next(
                    filter(
                        lambda c: c["name"] == annotation["@label"],
                        labels["categories"],
                    )
                )["id"]

                id_annotations = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                }

                if annotation_type == "polygon":
                    segmentation = [
                        list(map(float, point.split(",")))
                        for point in annotation["@points"].split(";")
                    ]
                    polygon = geometry.Polygon(segmentation)
                    minx, miny, maxx, maxy = polygon.bounds

                    bbox_width, bbox_height = maxx - minx, maxy - miny
                    bbox = [minx, miny, bbox_width, bbox_height]

                    if not (
                        (0 < bbox_width <= image_width)
                        and (0 < bbox_height <= image_height)
                    ):
                        raise RuntimeError(
                            f"An invalid bounding box {bbox} was derived from a segmentation mask {segmentation} of image '{filepath.name}'."
                        )

                    extra_annotations = {
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "area": polygon.area,
                        "iscrowd": 0,
                    }
                elif annotation_type == "box":
                    minx, miny, maxx, maxy = map(
                        float,
                        [
                            annotation["@xtl"],
                            annotation["@ytl"],
                            annotation["@xbr"],
                            annotation["@ybr"],
                        ],
                    )
                    extra_annotations = {"bbox": [minx, miny, maxx - minx, maxy - miny]}
                else:
                    # annotation_type == "tag": whole-image classification, no bbox
                    extra_annotations = {}

                labels["annotations"].append({**id_annotations, **extra_annotations})

                annotation_id += 1
            except Exception as e:
                logging.warning(f"Skipping annotation of image '{filepath}': {e}")

    return labels


def as_yolo(
    label_names: List[str],
    files_with_annotations: Dict[Path, Dict],
) -> Dict[Path, List[str]]:
    def _calc_annotation_entry(
        annotation: Dict, label_names: List[str], height: int, width: int
    ) -> List:
        label_name = annotation["@label"]
        label_index = label_names.index(label_name)

        xbr, xtl = float(annotation["@xbr"]), float(annotation["@xtl"])
        ybr, ytl = float(annotation["@ybr"]), float(annotation["@ytl"])

        obj_width = xbr - xtl
        obj_height = ybr - ytl

        x_center = (xtl + (obj_width / 2)) / width
        y_center = (ytl + (obj_height / 2)) / height

        norm_width = obj_width / width
        norm_height = obj_height / height

        return [label_index, x_center, y_center, norm_width, norm_height]

    labels = defaultdict(list)
    for filepath, image in files_with_annotations.items():
        height, width = int(image["@height"]), int(image["@width"])
        annotations = get_image_annotations(image, annotation_type="box")
        for annotation in annotations:
            annotation_entry = _calc_annotation_entry(
                annotation, label_names, height, width
            )
            annotation_entry = " ".join(map(str, annotation_entry))
            labels[filepath].append(annotation_entry)
    return labels


def get_cvat_dataset() -> Tuple[Dict, List[str], Dict[Path, Dict], Dict[Path, Dict]]:
    config: Dict = load_config()

    try:
        train_split, test_split = itemgetter("train-split", "test-split")(config)
    except KeyError:
        train_split, test_split = 0.8, 0.2

    filepaths: List[Path] = load_dataset_filepaths()

    annotation_file: Dict = load_annotations_file()
    annotations: List[Dict] = get_annotations(annotation_file)

    files_with_annotations: Dict[Path, Dict] = find_annotations_for_files(
        filepaths, annotations, strict=False
    )

    train_data, val_data = split_dataset(
        files_with_annotations, train_size=train_split, test_size=test_split
    )

    label_names = get_label_names(annotation_file)

    return config, label_names, train_data, val_data

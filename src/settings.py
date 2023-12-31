from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "CelebA"
PROJECT_NAME_FULL: str = "Large-scale CelebFaces Attributes (CelebA) Dataset"
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Custom(redistributable=False)
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Domain.General()]
CATEGORY: Category = Category.Entertainment()

CV_TASKS: List[CVTask] = [
    CVTask.ObjectDetection(),
    CVTask.Classification(),
    CVTask.Identification(),
    CVTask.WeaklySupervisedLearning(),
]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.ObjectDetection()]

RELEASE_DATE: Optional[str] = None  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = 2015

HOMEPAGE_URL: str = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 9228692
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/celeb-faces-attributes"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[
    Union[str, dict]
] = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = None
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = "https://arxiv.org/abs/1411.7766"
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = None

CITATION_URL: Optional[str] = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
AUTHORS: Optional[List[str]] = ["Ziwei Liu", "Ping Luo", "Xiaogang Wang", "Xiaoou Tang"]
AUTHORS_CONTACTS: Optional[List[str]] = [
    "lz013@ie.cuhk.edu.hk",
    "pluo@ie.cuhk.edu.hk",
    "xtang@ie.cuhk.edu.hk",
    "xgwang@ee.cuhk.edu.hk",
]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "The Chinese University of Hong Kong"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://www.cuhk.edu.hk/"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Additionally, every person has their own ***id*** tag",
    "attributes": [
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
    ],
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings

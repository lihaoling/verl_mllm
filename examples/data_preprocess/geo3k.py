# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import posixpath
import shutil
from urllib.parse import urlparse

import datasets  # type: ignore

try:
    import pyarrow.fs as pa_fs  # type: ignore
except Exception:
    pa_fs = None


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "data", "geo3k")


def _is_hdfs_path(path: str) -> bool:
    return path is not None and path.startswith("hdfs://")


def _get_hdfs_fs(path: str):
    if pa_fs is None:
        raise ImportError("需要写入 HDFS 时请安装 pyarrow，并确保启用了 HDFS 支持。")
    parsed = urlparse(path)
    host, port = parsed.hostname, parsed.port or 8020
    return pa_fs.HadoopFileSystem(host=host, port=port), parsed.path


def makedirs(path: str):
    """创建本地或 HDFS 目录，替代 verl.utils.hdfs_io.makedirs。"""
    if _is_hdfs_path(path):
        fs, hdfs_path = _get_hdfs_fs(path)
        fs.create_dir(hdfs_path, recursive=True)
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, dst: str):
    """复制目录到本地或 HDFS，替代 verl.utils.hdfs_io.copy。"""
    if _is_hdfs_path(dst):
        fs, hdfs_root = _get_hdfs_fs(dst)
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            target_dir = hdfs_root if rel == "." else posixpath.join(
                hdfs_root, rel)
            fs.create_dir(target_dir, recursive=True)
            for fname in files:
                local_path = os.path.join(root, fname)
                remote_path = posixpath.join(target_dir, fname)
                with open(local_path, "rb") as fsrc, fs.open_output_stream(remote_path) as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    else:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None,
                        help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--local_save_dir", default=DEFAULT_SAVE_DIR,
                        help="The save directory for the preprocessed dataset.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn(
        "test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

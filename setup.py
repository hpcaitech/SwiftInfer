import os
import subprocess
from typing import List

from setuptools import find_packages, setup


def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_version() -> str:
    """
    This function reads the version.txt and generates the colossalai/version.py file.

    Returns:
        The library version stored in version.txt.
    """

    setup_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(setup_file_path)
    version_txt_path = os.path.join(project_path, "version.txt")
    version_py_path = os.path.join(project_path, "swiftinfer/version.py")

    with open(version_txt_path) as f:
        version = f.read().strip()

    # write version into version.py
    with open(version_py_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

    return version


def build_trt_llm() -> None:
    try:
        import tensorrt_llm # noqa
    except ImportError:
        print("TensorRT-LLM is not installed. Building it now...")
        script_path = os.path.join(os.path.dirname(__file__), "scripts/build_trt_llm.sh")
        subprocess.run(f"bash {script_path}", shell=True, check=True)


build_trt_llm()

version = get_version()
package_name = "streamingllm-trt"

setup(
    name=package_name,
    version=version,
    packages=find_packages(
        exclude=(
            "tests",
            "build",
            "dist",
            "scripts",
            "requirements",
            "docs",
            "examples",
            "*.egg-info",
            "docker",
        )
    ),
    description="An efficient inference system based on TensorRT.",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://www.colossalai.org",
    project_urls={
        "Github": "https://github.com/hpcaitech/ColossalAI-Inference",
    },
    install_requires=fetch_requirements("requirements/requirements.txt"),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)

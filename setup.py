import os
from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    INSTALL_REQUIRES = f.read().splitlines()

EXTRAS_REQUIRE = {
    "dev": INSTALL_REQUIRES + [
        "pre-commit",
        "ruff==0.0.270",
        "black==23.3.0",
    ],
    "test": INSTALL_REQUIRES + [
        "pytest",
    ],
    "docs": INSTALL_REQUIRES + [
        "mkdocs-material",
    ],
}

setup(
    name="llmfinetune",
    version="0.0.1",
    description="A framework to finetune LLMs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include="llmadmin*"),
    keywords=["ChatGLM", "BaiChuan", "LLaMA", "BLOOM", "Falcon",
              "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],
    include_package_data=True,
    package_data={"llmadmin": ["models/*"]},
    entry_points={
        "console_scripts": [
            "llmfinetune=llmadmin.api.cli:app",
        ]
    },
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)

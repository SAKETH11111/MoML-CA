from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="moml-ca-mgnn",
    version="1.0.0",
    author="MoML-CA Team",
    author_email="your.email@example.com",
    description="Molecular Graph Neural Networks for PFAS Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MoML-CA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pfas-pipeline=code.MGNN.pfas_pipeline:main",
            "pfas-graph=code.MGNN.examples.molecular_graph_gen:main",
        ],
    },
) 
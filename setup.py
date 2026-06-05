from setuptools import setup, find_packages

setup(
    name="orchestrion",
    version="0.1.0",
    description="A framework for orchestrating robot tasks and managing function calls with synchronization options.",
    author="DexForce Technology Co., Ltd.",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

from setuptools import setup

# This file will be configured appropriately once the
# open source approval has been granted. Then, we
# will make this package available within the
# conda package manager

setup(
    name="solvmate",
    version="",
    packages=[
        "solvmate",
        "solvmate.ccryst",
        "solvmate.ccryst.web_app",
        "solvmate.ccryst.solvent_group",
        "solvmate.ranksolv",
    ],
    package_dir={"": "src"},
    url="",
    license="",
    author="Jan Wollschläger and Floriane Montanari",
    author_email="",
    description="",
)

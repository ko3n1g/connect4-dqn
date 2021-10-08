import setuptools

import connect4Agent

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="connect4agent",
    version=connect4Agent.__version__,
    author="Oliver Koenig",
    author_email="oliver.koenig@dpdhl.com",
    description="",
    install_requires=[

    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "pre-commit",
            "pytest",
            "gitlint",
        ],
        # "doc": ["sphinx", "sphinx-rtd-theme", "sphinxcontrib-confluencebuilder"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ko3n1g/connect4agent/",
    packages=setuptools.find_packages(exclude="tests"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.9",
)
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
        name = "debut",
        version = "0.0.1",
        author = "Boris Burle and Laure Spieser",
        author_email = "boris.burle@univ-amu.fr",
        description = "Debut is a package for detecting EMG burst onset.",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        url = "",
        packages = setuptools.find_packages(),
        classifiers = [
                "Programming Language :: Python :: 3",
                "Licence :: ",
                "Operating System :: OS Independent",
                ],
        python_requires = '>=3.6',
        )



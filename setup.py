import setuptools

#######################
# Taken from: 
# https://packaging.python.org/tutorials/packaging-projects/
#######################

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SRLPackage",
    version="0.0.1",
    scripts=['SRLPackage'],
    author="Andre Markard",
    author_email="andre@markard.eu",
    description="A summarization of the SRL process",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Yet to be published",
    packages=setuptools.find_packages(),
    install_requires=[
        'pyfn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
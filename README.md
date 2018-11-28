# SRLPackage

A summarization of the SRL process
- Based on (and using) pyfn (https://pypi.org/project/pyfn/)

## Preparation
- Clone repository or download files
- Enter the directory
- Run pip3 install . 


## Usage
- After installation run "SRLPackage download" to acquire all required data
- Then run "SRLPackage convert" to create the CoNLL dataset
- Now "SRLPackage f_id" can be used for Frame Identification

Alternatively conversion.sh provides a also the ability to convert FN data to CoNLL using pyfn.
In this case, manually download and extract the FrameNet dataset (https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z)
and adjust the path inside the script.

For further information run SRLPackage --help
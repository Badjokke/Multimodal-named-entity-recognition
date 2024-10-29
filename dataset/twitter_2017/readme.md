# Acquiring dataset
Please refer to [google drive](https://drive.google.com/drive/u/0/folders/1Yk5pTei9vVjkKpoHxEfs-5DP48LmRR1Z) to download data.  
Relevant files are:
- text
- images.zip

Extract the files and move them to this folder.

# Issue with images
It is possible that data processor for images may rise following warning:
``
libpng warning: iCCP: known incorrect sRGB profile
`` which is treated as error by python opencv lib.  
To fix this please follow [stackoverflow](https://stackoverflow.com/questions/22745076/libpng-warning-iccp-known-incorrect-srgb-profile?rq=4).
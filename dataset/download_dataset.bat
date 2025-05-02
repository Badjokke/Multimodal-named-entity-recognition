@echo off
set twitter15=twitter_2015.zip
set twitter17=twitter_2017.zip
echo downloading %twitter15% dataset from gdrive
curl "https://drive.usercontent.google.com/download?id=1vVlV3pXRJSkkuhNL1KgdkcUq2eburAix&confirm=xxx" -o %twitter15%
echo downloading %twitter17% dataset from gdrive
curl "https://drive.usercontent.google.com/download?id=1-HIsdNKtncinP2GKaa4YF0pe3NnG4K_t&confirm=xxx" -o %twitter17%
echo "unpacking"
tar -xf %twitter15% twitter_2015
tar -xf %twitter17% twitter_2017
del %twitter15%
del %twitter17%
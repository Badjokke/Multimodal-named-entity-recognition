#!/bin/bash

twitter15="twitter_2015.zip"
twitter17="twitter_2017.zip"

echo "Downloading $twitter15 dataset from GDrive"
curl -L "https://drive.usercontent.google.com/download?id=1vVlV3pXRJSkkuhNL1KgdkcUq2eburAix&confirm=xxx" -o "$twitter15"

echo "Downloading $twitter17 dataset from GDrive"
curl -L "https://drive.usercontent.google.com/download?id=1-HIsdNKtncinP2GKaa4YF0pe3NnG4K_t&confirm=xxx" -o "$twitter17"

echo "Unpacking"
unzip "$twitter15"
unzip "$twitter17"

rm "$twitter15"
rm "$twitter17"

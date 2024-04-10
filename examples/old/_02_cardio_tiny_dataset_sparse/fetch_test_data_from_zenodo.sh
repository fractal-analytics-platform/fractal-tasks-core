#!/bin/bash

DOI="10.5281/zenodo.7059515"
CLEAN_DOI=${DOI/\//_}
zenodo_get $DOI -o ../images/${CLEAN_DOI}

rm -r ../images/${CLEAN_DOI}_sparse
cp -r ../images/${CLEAN_DOI} ../images/${CLEAN_DOI}_sparse

MLF=../images/${CLEAN_DOI}_sparse/MeasurementData.mlf
sed -i '3,4s/1517/3517/g' $MLF
sed -i 's/1448/3448/g' $MLF
cat $MLF

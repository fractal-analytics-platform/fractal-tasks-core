#!/bin/bash

DOIs="10.5281/zenodo.7057076"

for DOI in $DOIs; do
    CLEAN_DOI=${DOI/\//_}
    echo $CLEAN_DOI
    zenodo_get $DOI -o data/${CLEAN_DOI}
done

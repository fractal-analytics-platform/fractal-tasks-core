#!/bin/bash

LIST_RECORD_ID="7059515 8091756"

for RECORD_ID in $LIST_RECORD_ID; do
    echo "****************************"
    echo "START RECORD_ID=$RECORD_ID"
    OUTPUT_FOLDER=tests/data/10_5281_zenodo_$RECORD_ID
    echo "OUTPUT_FOLDER: $OUTPUT_FOLDER"

    if [ -d $OUTPUT_FOLDER ]; then
        echo "OUTPUT_FOLDER already exists. Exit."
    else
        mkdir $OUTPUT_FOLDER
        FILES=`curl https://zenodo.org/api/records/$RECORD_ID | jq -r ".files[].links.download"`
        echo "curl exit code: $?"
        for FILE in $FILES; do
            FILEPATH=${FILE%"/content"}
            FILENAME=`basename $FILEPATH`
            echo "FILE:     $FILE"
            echo "FILEPATH: $FILEPATH"
            echo "FILENAME: $FILENAME"
            echo
            wget --no-verbose $FILE --output-document=${OUTPUT_FOLDER}/${FILENAME}
            echo
        done
    fi

    echo "END RECORD_ID=$RECORD_ID"
    echo "****************************"
    echo
done

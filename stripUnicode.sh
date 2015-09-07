#!/bin/bash

#usage: <scriptName> -i INPUT -o OUTPUT
while [[ $# > 1 ]]
do
key="$1"

case $key in
    -i|--input)
    INPUT="$2"
    shift
    ;;
    -o|--output)
    OUTPUT="$2"
    shift
    ;;
    *)
            # unknown option
    ;;
esac
shift
done
echo INPUT FILE  = "${INPUT}"
echo SEARCH PATH     = "${OUTPUT}"

if [[ -n ${INPUT} ]] && [[ -n ${OUTPUT} ]]; then
    tempf=$(mktemp); iconv -c -f utf-8 -t ascii ${INPUT} > $tempf; iconv -f ascii -t utf-8 $tempf > ${OUTPUT}
fi


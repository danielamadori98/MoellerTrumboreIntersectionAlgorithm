#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/src/main.cu -I"$DIR"/include -I"$DIR"/src -o moeller -allow-unsupported-compiler


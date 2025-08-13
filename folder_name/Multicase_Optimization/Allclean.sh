#!/bin/bash

# clean everyting
echo "Cleaning..."
cd Forward
rm -rf postProcessing
rm -rf __pycache__
rm -rf 0.*
rm -rf processor*
rm -rf reports
rm -rf *.bin *.info *.txt *.html *.hst
cd ../Hover
rm -rf postProcessing
rm -rf __pycache__
rm -rf 0.*
rm -rf processor*
rm -rf reports
rm -rf *.bin *.info *.txt *.html *.hst
cd ../
rm -rf reports
rm -rf __pycache__
rm -rf reports
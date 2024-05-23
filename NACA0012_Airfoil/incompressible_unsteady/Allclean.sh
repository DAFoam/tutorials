#!/bin/bash

while true 
do
    read -p "Delete everything and resume to the default setup (y/n)?" yn
    case $yn in
        [Yy]* ) 
            # clean everyting
            echo "Cleaning..."
            rm -rf 0
            rm -rf postProcessing
            rm -rf constant/extendedFeatureEdgeMesh
            rm -rf constant/triSurface
            rm -rf constant/polyMesh/
            rm -rf *.bin *.info *.dat *.xyz *.stl
            rm -rf processor* 0.0000*
            rm -rf {1..9}*
            exit
            ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


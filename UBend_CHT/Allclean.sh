#!/bin/bash

echo "Cleaning..."

rm -rf log-*
rm -rf opt_SNOPT_*
rm -rf OptView.hst
rm -rf n2.html
rm -rf OptView.hst
rm -rf Opt.txt
rm -rf log.meshGeneration

cd aero
rm -rf 0
rm -rf postProcessing
rm -rf constant/extendedFeatureEdgeMesh
rm -rf constant/triSurface
rm -rf *.bin *.info *.dat *.xyz *.stl
rm -rf processor* 0.00*
rm -rf {1..9}*

cd ../thermal
rm -rf 0
rm -rf postProcessing
rm -rf constant/extendedFeatureEdgeMesh
rm -rf constant/triSurface
rm -rf *.bin *.info *.dat *.xyz *.stl
rm -rf processor* 0.00*
rm -rf {1..9}*

echo "Done!"
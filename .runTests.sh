#!/usr/bin/env bash
  
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Argument not found. Options: incompressible, compressible, and solid"
  exit 1
else
  argm="$1"
fi

case $argm in
  "v2")
    echo "Running incompressible tests"
    find */system/controlDict -type f -exec sed -i '/endTime/c\endTime 10;' {} \;
    find */*/system/controlDict -type f -exec sed -i '/endTime/c\endTime 10;' {} \;
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    #cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd NACA0012_Airfoil/multipoint && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd JBC_Hull && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd UBend_Channel && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd pitzDaily && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI --opt=ipopt && cd - || exit 1
    ;;
  "v3")
    echo "Running compressible tests"
    find */system/controlDict -type f -exec sed -i '/endTime/c\endTime 10;' {} \;
    find */*/system/controlDict -type f -exec sed -i '/endTime/c\endTime 10;' {} \;
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    #cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd 30N30P_MultiElement_Airfoil && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd Onera_M6_Wing && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd Onera_M6_Wing && echo y | ./Allclean.sh && ./preProcessing_snappyHexMesh.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd CRM_Wing && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd Rotor37_Compressor && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd NREL6_Wind_Turbine && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd Prowim_Wing_Propeller && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd DPW4_Aircraft && sed -i 's/snappyHexMesh -overwrite >> log.meshGeneration/snappyHexMesh -overwrite/g' preProcessing.sh && ./preProcessing.sh && mpirun -np 2 python runScript.py --task=testAPI && cd - || exit 1
    #cd DPW4_Aircraft && echo y | ./Allclean.sh && ./preProcessing.sh && mpirun -np 2 python runScript2FFDs.py --task=testAPI && cd - || exit 1
    ;;
  *)
    echo "Argument not valid! Options are: v2, v3"
    echo "Example: . .runTests.sh v2"
    exit 1
    ;;
esac

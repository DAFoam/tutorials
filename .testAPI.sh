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
    find */runScript* -type f -exec sed -i '/"primalMinResTol":/c\    "primalMinResTol": 0.9,' {} \;
    find */*/runScript* -type f -exec sed -i '/"primalMinResTol":/c\    "primalMinResTol": 0.9,' {} \;
    cd 30N30P_MultiElement_Airfoil && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd CRM_Wing && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd DPW4_Aircraft && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd JBC_Hull && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/multipoint && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd NREL6_Wind_Turbine && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd Onera_M6_Wing && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd PeriodicHill_FieldInversion && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd pitzDaily && ./preProcessing.sh && python runScript_v2.py --task=runPrimal --opt=ipopt && cd - || exit 1
    cd PlateHole_Structure && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd Prowim_Wing_Propeller && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd Rotor37_Compressor && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    cd UBend_Channel && ./preProcessing.sh && python runScript_v2.py --task=runPrimal && cd - || exit 1
    ;;
  "v3")
    echo "Running compressible tests"
    find */runScript* -type f -exec sed -i '/"primalMinResTol":/c\    "primalMinResTol": 0.9,' {} \;
    find */*/runScript* -type f -exec sed -i '/"primalMinResTol":/c\    "primalMinResTol": 0.9,' {} \;
    cd 30N30P_MultiElement_Airfoil && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd ADODG3_Wing && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd CRM_Wing && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd MACH_Tutorial_Wing && ./preProcessing.sh && python runScript_Aero.py -task=runPrimal && cd - || exit 1
    cd MACH_Tutorial_Wing && python runScript_AeroStruct.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/incompressible && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/SpalartAllmaras/kOmegaSST/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/subsonic && echo y | ./Allclean.sh && sed -i 's/kOmegaSST/kEpsilon/g' constant/turbulenceProperties && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/multipoint && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd NACA0012_Airfoil/multicase && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd PeriodicHill_FieldInversion && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    cd Rotor37_Compressor && ./preProcessing.sh && python runScript.py -task=runPrimal && cd - || exit 1
    ;;
  *)
    echo "Argument not valid! Options are: v2, v3"
    echo "Example: . .testAPI.sh v2"
    exit 1
    ;;
esac

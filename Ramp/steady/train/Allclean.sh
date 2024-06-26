#!/bin/bash

while true 
do
    read -p "Delete everything and resume to the default setup (y/n)?" yn
    case $yn in
        [Yy]* ) 
            # clean everyting
            echo "Cleaning..."
            rm -rf */0
            rm -rf */*.bin */*.info */*.xyz */*.txt
            rm -rf */processor* */0.00*
            rm -rf */{1..9}*
            exit
            ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


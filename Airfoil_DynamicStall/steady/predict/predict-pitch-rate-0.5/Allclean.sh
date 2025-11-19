#!/bin/bash

while true 
do
    read -p "Delete everything and resume to the default setup (y/n)?" yn
    case $yn in
        [Yy]* ) 
            # clean everyting
            echo "Cleaning..."
            rm -rf */processor*
            exit
            ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


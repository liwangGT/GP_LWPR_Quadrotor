echo "Running batch GP regression"
## declare an array variable
declare -a arr=("element1" "element2" "element3")

declare -a filelist=("cf1_3rd_data_09202017_GPOFFwind_v7_partial_XY"
"cf1_3rd_data_09202017_PIDdrift_v0GPWind_XY"
"cf1_3rd_data_09202017_PIDdrift_v0Wind_XY"
"cf1_3rd_data_09202017_PIDdrift_v1GPWind_XY"
"cf1_3rd_data_12052017_wind1off_XY"
"cf1_3rd_data_12052017_wind1on_XY"
"cf1_3rd_data_12052017_wind2off_XY"
"cf1_3rd_data_12052017_wind2on_XY"
)

for filename in "${filelist[@]}"
do 
    echo "Processing" "$filename"
    python ExpCompare.py $filename
    echo "Finished generating GP prediction data for " "$filename"
done





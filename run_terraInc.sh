for dataset in 'TerraInc'
do
    for target in 'location_38' 'location_43' 'location_46' 'location_100'; do
        for rec_weight in 0.1; do
	        sbatch --job-name=$dataset-$target batch.sh $dataset $target $rec_weight
        done
    done
done
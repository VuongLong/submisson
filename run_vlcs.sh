for dataset in 'VLCS'
do
    for target in 'Caltech' 'LabelMe' 'SUN' 'VOC'; do
        for rec_weight in 0.1; do
	        sbatch --job-name=$dataset-$target batch.sh $dataset $target $rec_weight
        done
    done
done
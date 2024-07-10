for dataset in 'OfficeHome'
do
    for target in 'art' 'clipart' 'real' 'product'; do
        for rec_weight in 0.1; do
	        sbatch --job-name=$dataset-$target batch.sh $dataset $target $rec_weight
        done
    done
done
for dataset in 'DomainNet'
do
    for target in 'clipart' 'infograph' 'painting' 'real' 'quickdraw' 'sketch'; do
        for rec_weight in 0.1; do
	        sbatch --job-name=$dataset-$target batch.sh $dataset $target $rec_weight
        done
    done
done
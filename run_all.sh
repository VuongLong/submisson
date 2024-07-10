for dataset in 'PACS' 'TerraIncognita' 'VLCS' 'OfficeHome'; do
    for seed in 0 1 2; do
	    sbatch --job-name=$dataset-$target sbatch.sh $dataset $seed
    done
done
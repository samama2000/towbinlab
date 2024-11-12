#!/bin/bash


EXPERIMENT_DIR="/mnt/towbin.data/shared/smarin/analysis/lifespan_20240404-20240605"

FILEMAP_CSV="$EXPERIMENT_DIR/analysis/report/analysis_filemap.csv"



summary_script=$(cat <<END
import pandas as pd
from pathlib import Path

csv_dir = Path('$FILEMAP_CSV').parent / 'stardist_seg_features'
csv_files = list(csv_dir.glob('*.csv'))

summary_file = csv_dir.parent / 'stardist_report.csv'
print(summary_file)
summary_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
summary_df = (
    summary_df
        .groupby(['Time', 'Point'])
        .size()
        .reset_index(name='egg-count')
        .sort_values(by=['Point', 'Time'])
)
summary_df.to_csv(summary_file, index=False)

END
)

sbatch \
    --cpus-per-task=4 \
    --time='1:00:00' \
    --mem='32GB' \
    --wrap="~/.local/bin/micromamba run -n stardist_eggseg_env python3 -c \"$summary_script\"" \


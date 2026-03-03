## How to run

## Mode 1

## CSV-based (N=155)

python .\bot_bayes_cefr_unified_csv_only.py 
`
  --out_dir out_csv `
  
  --bank_csv .\question_bank_mo_with_bands_range_utf8sig.csv `
  
  --export_review_lists

  
## Simulation 180

python .\bot_bayes_cefr_unified_csv_only.py --out_dir out_sim --export_review_lists


## Mode 2

1) Run with your uploaded CSV (has band_low/band_high):

   python bot_bayes_mode2_run_csv_no_bands.py `
   
     --bank_csv question_bank_mo_with_bands_range_utf8sig.csv `
   
     --id_col question_id --truth_col cefr_level `
   
     --p_from_range band_low band_high `
   
     --out_dir out_mode2_T200 --T_max 200 --seed 42

3) Run with a CSV that already has true_p_mid:

   python bot_bayes_mode2_run_csv_no_bands.py `
   
     --bank_csv question_bank_mo_with_bands_range_utf8sig.csv --id_col item_id --p_col true_p_mid `
   
     --out_dir out_mode2_T200 --T_max 200

5) Run synthetic 180-item bank (no CSV):

   python bot_bayes_mode2_run_csv_no_bands.py --out_dir out_mode2_T200_sim --T_max 200

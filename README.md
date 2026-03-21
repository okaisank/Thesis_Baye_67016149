## How to run

## Mode 1

## CSV-based 

1) Run with your uploaded CSV (has band_low/band_high):
      
python .\bot_bayes_cefr_mode_1.py `

  --out_dir .\out_mode1_csv `
  
  --bank_csv .\question_bank_180_balanced_with_true_p_mid_stratified.csv `
  
  --T_max 200 `
  
  --step 10 `
  
  --train_frac 0.70 `
  
  --seed_bank 42 `
  
  --seed_resp 123 `
  
  --prob_model logit_sigmoid `
  
  --mod_mag 0.15 `
  
  --alpha0 1.0 `
  
  --beta0 1.0 `
  
  --export_review_lists
  
## Simulation 

2) Run synthetic 180-item bank (no CSV):

python .\bot_bayes_cefr_mode_1.py `

  --out_dir .\out_mode1_sim `
  
  --T_max 200 `
  
  --step 10 `
  
  --train_frac 0.70 `
  
  --seed_bank 42 `
  
  --seed_resp 123 `
  
  --prob_model logit_sigmoid `
  
  --mod_mag 0.15 `
  
  --alpha0 1.0 `
  
  --beta0 1.0 `
  
  --export_review_lists


## Mode 2

## CSV-based 

1) Run with your uploaded CSV (has band_low/band_high):

python .\bot_bayes_mode_2.py `

  --bank_csv .\question_bank_180_balanced_with_true_p_mid_stratified.csv `
  
  --id_col question_id `
  
  --truth_col cefr_level `
  
  --p_col true_p_mid `
  
  --out_dir .\out_mode2_csv `
  
  --T_max 200 `
  
  --step 10 `
  
  --seed 42

## Simulation 

2) Run synthetic 180-item bank (no CSV):

python .\bot_bayes_mode_2.py `

  --out_dir .\out_mode2_sim `
  
  --T_max 200 `
  
  --step 10 `
  
  --seed 42

  

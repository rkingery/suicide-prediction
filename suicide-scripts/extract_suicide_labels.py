### ICD9 codes possibly containing suicidal tendencies
#E950 Suicide and self-inflicted poisoning by solid or liquid substances
#E951 Suicide and self-inflicted poisoning by gases in domestic use
#E952 Suicide and self-inflicted poisoning by other gases and vapors
#E953 Suicide and self-inflicted injury by hanging, strangulation, and suffocation
#E954 Suicide and self-inflicted injury by submersion [drowning]
#E955 Suicide and self-inflicted injury by firearms, air guns and explosives
#E956 Suicide and self-inflicted injury by cutting and piercing instrument
#E957 Suicide and self-inflicted injuries by jumping from high place
#E958 Suicide and self-inflicted injury by other and unspecified means
#E959 Late effects of self-inflicted injury
#V6284 Suicidal ideation
#V1559 Personal history of self-harm

### other catch-all codes considered - not currently using
#V5889 Other specified aftercare
#9089 Late effect of unspecified injury

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = '/Users/ryankingery/desktop/suicides/data/'

suicide_codes = ['E950','E951','E952','E953','E954','E955','E956','E957','E958',
                 'E959','V6284','V1559']

def extract_suicides(row):
    label = 0
    for code in suicide_codes:
        if code in row.icd9_codes:
            label = 1
    return label

def main():
    df = pd.read_csv(path+'hadm_id_icd9_codes.csv',sep='|')
    df.columns = ['hadm_id', 'icd9_codes']
    
    df['labels'] = df.apply(extract_suicides,axis=1)
    
    df[['hadm_id','labels']].to_csv(path+'labels.csv',index=False)
    
    df_text = pd.read_csv(path+'std_format_raw_data.csv',index_col=0)
    df_text = df_text.drop('labels',axis=1)
    
    df_new = pd.merge(df,df_text,on='hadm_id',how='inner')
    df_new = df_new[['subject_id', 'hadm_id', 'labels', 'text']]
    df_new.to_csv(path+'text_with_labels.csv',index=False)
    

if __name__ == '__main__':
    main()



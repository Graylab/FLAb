import pandas as pd
from abnumber import Chain

# original trastuzumab sequences
heavy = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS'
light = 'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK'

# save cdr regions in variables, to be replaced
chain = Chain(heavy, scheme='imgt')

h1 = chain.cdr1_seq
h2 = chain.cdr2_seq
h3 = chain.cdr3_seq

# load original datasets
zero = pd.read_csv('zero-shot-binders.csv')
multi =pd.read_csv('multi-step-binders.csv')

# light chains are unchanged so make list with no changes
l_zero = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']*422
l_multi = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']*24

# heavy chains need to be remade
# zeros
designed_h_zero = []

for i in range(422):
    h_new = heavy.replace(h1, zero['HCDR3'][i])
    designed_h_zero.append(h_new)

# multi
designed_h_multi = []

for i in range(24):
    h_h1_new = heavy.replace(h1, multi['HCDR1'][i])
    h_h2_new = h_h1_new.replace(h2, multi['HCDR2'][i])
    h_h3_new = h_h2_new.replace(h3, multi['HCDR3'][i])
    
    designed_h_multi.append(h_h3_new)

zero_new = pd.DataFrame({'heavy':designed_h_zero, 'light':l_zero, 'fitness':zero['KD (nM)']})
zero_new.to_csv('Shanehsazzadeh2023_trastuzumab_zero_kd.csv')

multi_new = pd.DataFrame({'heavy':designed_h_multi, 'light':l_multi, 'fitness':multi['KD (nM)']})
multi_new.to_csv('Shanehsazzadeh2023_trastuzumab_multi_kd.csv')

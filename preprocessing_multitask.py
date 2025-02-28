import pandas as pd
from Bio.Seq import Seq

exp_data = pd.read_csv('/extdata2/YMC/AR/data/AR_Dependency_Result3_syn5.csv')
clinvar = pd.read_csv('/extdata2/YMC/AR/data/Clinvar_AR_LBD.csv').loc[:, ['AA_Change', 'Germline classification']]
clinvar_e13 = pd.read_csv('/extdata2/YMC/AR/data/AR_E13_Processed.csv').loc[:, ['AA_Change', 'Germline classification']]
am_data = pd.read_csv('/extdata2/YMC/AR/data/external/AlphaMissense-Search-P10275.csv')
phylop = pd.read_csv('/extdata2/YMC/AR/data/external/PhyloP_codon_score.csv')

refseq = 'atggaagtgcagttagggctgggaagggtctaccctcggccgccgtccaagacctaccgaggagctttccagaatctgttccagagcgtgcgcgaagtgatccagaacccgggccccaggcacccagaggccgcgagcgcagcacctcccggcgccagtttgctgctgctgcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcaagagactagccccaggcagcagcagcagcagcagggtgaggatggttctccccaagcccatcgtagaggccccacaggctacctggtcctggatgaggaacagcaaccttcacagccgcagtcggccctggagtgccaccccgagagaggttgcgtcccagagcctggagccgccgtggccgccagcaaggggctgccgcagcagctgccagcacctccggacgaggatgactcagctgccccatccacgttgtccctgctgggccccactttccccggcttaagcagctgctccgctgaccttaaagacatcctgagcgaggccagcaccatgcaactccttcagcaacagcagcaggaagcagtatccgaaggcagcagcagcgggagagcgagggaggcctcgggggctcccacttcctccaaggacaattacttagggggcacttcgaccatttctgacaacgccaaggagttgtgtaaggcagtgtcggtgtccatgggcctgggtgtggaggcgttggagcatctgagtccaggggaacagcttcggggggattgcatgtacgccccacttttgggagttccacccgctgtgcgtcccactccttgtgccccattggccgaatgcaaaggttctctgctagacgacagcgcaggcaagagcactgaagatactgctgagtattcccctttcaagggaggttacaccaaagggctagaaggcgagagcctaggctgctctggcagcgctgcagcagggagctccgggacacttgaactgccgtctaccctgtctctctacaagtccggagcactggacgaggcagctgcgtaccagagtcgcgactactacaactttccactggctctggccggaccgccgccccctccgccgcctccccatccccacgctcgcatcaagctggagaacccgctggactacggcagcgcctgggcggctgcggcggcgcagtgccgctatggggacctggcgagcctgcatggcgcgggtgcagcgggacccggttctgggtcaccctcagccgccgcttcctcatcctggcacactctcttcacagccgaagaaggccagttgtatggaccgtgtggtggtggtgggggtggtggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcgaggcgggagctgtagccccctacggctacactcggccccctcaggggctggcgggccaggaaagcgacttcaccgcacctgatgtgtggtaccctggcggcatggtgagcagagtgccctatcccagtcccacttgtgtcaaaagcgaaatgggcccctggatggatagctactccggaccttacggggacatgcgtttggagactgccagggaccatgttttgcccattgactattactttccaccccagaagacctgcctgatctgtggagatgaagcttctgggtgtcactatggagctctcacatgtggaagctgcaaggtcttcttcaaaagagccgctgaagggaaacagaagtacctgtgcgccagcagaaatgattgcactattgataaattccgaaggaaaaattgtccatcttgtcgtcttcggaaatgttatgaagcagggatgactctgggagcccggaagctgaagaaacttggtaatctgaaactacaggaggaaggagaggcttccagcaccaccagccccactgaggagacaacccagaagctgacagtgtcacacattgaaggctatgaatgtcagcccatctttctgaatgtcctggaagccattgagccaggtgtagtgtgtgctggacacgacaacaaccagcccgactcctttgcagccttgctctctagcctcaatgaactgggagagagacagcttgtacacgtggtcaagtgggccaaggccttgcctggcttccgcaacttacacgtggacgaccagatggctgtcattcagtactcctggatggggctcatggtgtttgccatgggctggcgatccttcaccaatgtcaactccaggatgctctacttcgcccctgatctggttttcaatgagtaccgcatgcacaagtcccggatgtacagccagtgtgtccgaatgaggcacctctctcaagagtttggatggctccaaatcaccccccaggaattcctgtgcatgaaagcactgctactcttcagcattattccagtggatgggctgaaaaatcaaaaattctttgatgaacttcgaatgaactacatcaaggaactcgatcgtatcattgcatgcaaaagaaaaaatcccacatcctgctcaagacgcttctaccagctcaccaagctcctggactccgtgcagcctattgcgagagagctgcatcagttcGcttttgacctgctaatcaagtcacacatggtgagcgtggactttccggaaatgatggcagagatcatctctgtgcaagtgcccaagatcctttctgggaaagtcaagcccatctatttccacacccagtga'
refseq_dna = Seq(refseq)
refseq_protein = refseq_dna.translate()

def encode_aachange(row):
    pos = row['position']
    ori = row['a.a.1']
    mut = row['a.a.2']
    
    encoded = ori + str(pos) + mut
    
    return encoded

def apply_aa_change(ref_protein, aa_change):
    mutated_seq = list(str(ref_protein))
    try:
        pos = int(aa_change[1:-1]) - 1 # Position (Zero-indexed)
        if mutated_seq[pos] == aa_change[0]:
            if aa_change[-1] != '*':
                mutated_seq[pos] = aa_change[-1]
            else:
                return ''.join(mutated_seq[:pos])
        return ''.join(mutated_seq[:-1])
    except:
        return None
    
am_data['AA_Change'] = am_data.apply(encode_aachange, axis=1)
am_data['pos'] = am_data['position']
am_data['alphamissense'] = am_data['pathogenicity score']
am_data = am_data.loc[:, ['is_snv', 'AA_Change', 'pos', 'alphamissense']]

exp_data = exp_data[~exp_data['AA_Change'].str.contains('_', na=False)]
exp_data = exp_data[~exp_data['AA_Change'].str.startswith('*', na=False)]
exp_data = exp_data[~exp_data['AA_Change'].str.endswith('*', na=False)]

exp_data['label'] = exp_data['Fitness_Score'] / 10
exp_data['classification'] = exp_data['Classification'].apply(lambda x: 1 if x == 'Functional' else 0 if x == 'Intermediate' else -1)
exp_data['pos'] = exp_data['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)
exp_data = exp_data[['AA_Change', 'pos', 'classification', 'label']]

merged_data = pd.merge(am_data, exp_data, on='AA_Change', how='outer', suffixes=('_am', ''))
merged_data.loc[merged_data['alphamissense'].isna(), 'alphamissense'] = 0.05
merged_data.loc[merged_data['classification'].isna(), 'classification'] = 0

# Handle duplicate pos columns
if 'pos_am' in merged_data.columns:
    merged_data['pos'] = merged_data['pos'].fillna(merged_data['pos_am'])
    merged_data.drop('pos_am', axis=1, inplace=True)

clinvar_combined = pd.concat([clinvar, clinvar_e13], ignore_index=True)
clinvar_combined['pos'] = clinvar_combined['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)

clinvar_aachange = clinvar_combined.loc[clinvar_combined['Germline classification']!='Uncertain significance', ['AA_Change', 'pos', 'Germline classification']]
clinvar_aachange['Germline classification'] = clinvar_aachange['Germline classification'].apply(lambda x: 0 if x in ['Benign', 'Likely benign', 'Benign/Likely benign'] else 1)
clinvar_dict = clinvar_aachange.drop_duplicates('AA_Change').set_index('AA_Change')['Germline classification'].to_dict()

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

data = merged_data.copy()

data['Fold'] = data['AA_Change'].apply(lambda x: 'Test_ClinVar' if x in clinvar_aachange['AA_Change'].values else None)

non_clinvar_data = data[(data['Fold'].isna()) & (~data['label'].isna())].copy()
non_clinvar_indices = non_clinvar_data.index.tolist()

splitter = GroupShuffleSplit(test_size=0.05, n_splits=1, random_state=216)
split = splitter.split(non_clinvar_data, groups=non_clinvar_data['pos'])
train_indices, test_indices = next(split)

train_idx = [non_clinvar_indices[i] for i in train_indices]
test_idx = [non_clinvar_indices[i] for i in test_indices]

data.loc[test_idx, 'Fold'] = 'Test_internal'

train_data_with_label = data.loc[train_idx + data[(data['Fold'].isna()) & (~data['label'].isna())].index.tolist()].copy()
train_data_indices_with_label = train_data_with_label.index.tolist()

if len(train_data_with_label) > 0:
    gkf_with_label = GroupKFold(n_splits=5)
    for fold, (_, fold_indices) in enumerate(gkf_with_label.split(train_data_with_label, groups=train_data_with_label['pos'])):
        # Convert back to original DataFrame indices
        original_indices = [train_data_indices_with_label[i] for i in fold_indices]
        data.loc[original_indices, 'Fold'] = f'Train_{fold+1}'

train_data_without_label = data[data['Fold'].isna() & (data['label'].isna())].copy()
train_data_indices_without_label = train_data_without_label.index.tolist()

if len(train_data_without_label) > 0:
    gkf_without_label = GroupKFold(n_splits=5)
    for fold, (_, fold_indices) in enumerate(gkf_without_label.split(train_data_without_label, groups=train_data_without_label['pos'])):
        original_indices = [train_data_indices_without_label[i] for i in fold_indices]
        data.loc[original_indices, 'Fold'] = f'Train_{fold+1}'

data['clinvar classification'] = data.apply(
    lambda row: clinvar_dict[row['AA_Change']] if row['AA_Change'] in clinvar_dict else None, 
    axis=1
)

data['wt'] = str(refseq_protein[:-1])
data['sequence'] = data['AA_Change'].apply(lambda x: apply_aa_change(refseq_protein, x))

data.to_csv('/extdata2/YMC/AR/data/all_dependency_merged.csv', index=False)
print("All dependency data merged and saved successfully.")
import pandas as pd
from Bio.Seq import Seq

data = pd.read_csv('/extdata2/YMC/AR/data/AR_Dependency_Result3_syn5.csv')
clinvar = pd.read_csv('/extdata2/YMC/AR/data/Clinvar_AR_LBD.csv')
alphamissense = pd.read_csv('/extdata2/YMC/AR/data/external/AlphaMissense-Search-P10275.csv')
phylop = pd.read_csv('/extdata2/YMC/AR/data/external/PhyloP_codon_score.csv')

refseq = 'atggaagtgcagttagggctgggaagggtctaccctcggccgccgtccaagacctaccgaggagctttccagaatctgttccagagcgtgcgcgaagtgatccagaacccgggccccaggcacccagaggccgcgagcgcagcacctcccggcgccagtttgctgctgctgcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcaagagactagccccaggcagcagcagcagcagcagggtgaggatggttctccccaagcccatcgtagaggccccacaggctacctggtcctggatgaggaacagcaaccttcacagccgcagtcggccctggagtgccaccccgagagaggttgcgtcccagagcctggagccgccgtggccgccagcaaggggctgccgcagcagctgccagcacctccggacgaggatgactcagctgccccatccacgttgtccctgctgggccccactttccccggcttaagcagctgctccgctgaccttaaagacatcctgagcgaggccagcaccatgcaactccttcagcaacagcagcaggaagcagtatccgaaggcagcagcagcgggagagcgagggaggcctcgggggctcccacttcctccaaggacaattacttagggggcacttcgaccatttctgacaacgccaaggagttgtgtaaggcagtgtcggtgtccatgggcctgggtgtggaggcgttggagcatctgagtccaggggaacagcttcggggggattgcatgtacgccccacttttgggagttccacccgctgtgcgtcccactccttgtgccccattggccgaatgcaaaggttctctgctagacgacagcgcaggcaagagcactgaagatactgctgagtattcccctttcaagggaggttacaccaaagggctagaaggcgagagcctaggctgctctggcagcgctgcagcagggagctccgggacacttgaactgccgtctaccctgtctctctacaagtccggagcactggacgaggcagctgcgtaccagagtcgcgactactacaactttccactggctctggccggaccgccgccccctccgccgcctccccatccccacgctcgcatcaagctggagaacccgctggactacggcagcgcctgggcggctgcggcggcgcagtgccgctatggggacctggcgagcctgcatggcgcgggtgcagcgggacccggttctgggtcaccctcagccgccgcttcctcatcctggcacactctcttcacagccgaagaaggccagttgtatggaccgtgtggtggtggtgggggtggtggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcgaggcgggagctgtagccccctacggctacactcggccccctcaggggctggcgggccaggaaagcgacttcaccgcacctgatgtgtggtaccctggcggcatggtgagcagagtgccctatcccagtcccacttgtgtcaaaagcgaaatgggcccctggatggatagctactccggaccttacggggacatgcgtttggagactgccagggaccatgttttgcccattgactattactttccaccccagaagacctgcctgatctgtggagatgaagcttctgggtgtcactatggagctctcacatgtggaagctgcaaggtcttcttcaaaagagccgctgaagggaaacagaagtacctgtgcgccagcagaaatgattgcactattgataaattccgaaggaaaaattgtccatcttgtcgtcttcggaaatgttatgaagcagggatgactctgggagcccggaagctgaagaaacttggtaatctgaaactacaggaggaaggagaggcttccagcaccaccagccccactgaggagacaacccagaagctgacagtgtcacacattgaaggctatgaatgtcagcccatctttctgaatgtcctggaagccattgagccaggtgtagtgtgtgctggacacgacaacaaccagcccgactcctttgcagccttgctctctagcctcaatgaactgggagagagacagcttgtacacgtggtcaagtgggccaaggccttgcctggcttccgcaacttacacgtggacgaccagatggctgtcattcagtactcctggatggggctcatggtgtttgccatgggctggcgatccttcaccaatgtcaactccaggatgctctacttcgcccctgatctggttttcaatgagtaccgcatgcacaagtcccggatgtacagccagtgtgtccgaatgaggcacctctctcaagagtttggatggctccaaatcaccccccaggaattcctgtgcatgaaagcactgctactcttcagcattattccagtggatgggctgaaaaatcaaaaattctttgatgaacttcgaatgaactacatcaaggaactcgatcgtatcattgcatgcaaaagaaaaaatcccacatcctgctcaagacgcttctaccagctcaccaagctcctggactccgtgcagcctattgcgagagagctgcatcagttcGcttttgacctgctaatcaagtcacacatggtgagcgtggactttccggaaatgatggcagagatcatctctgtgcaagtgcccaagatcctttctgggaaagtcaagcccatctatttccacacccagtga'

clinvar['pos'] = clinvar['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)

clinvar_aachange = clinvar.loc[clinvar['Germline classification']!='Uncertain significance', ['AA_Change', 'pos', 'Germline classification']]
clinvar_aachange['Germline classification'] = clinvar_aachange['Germline classification'].apply(lambda x: 1 if x in ['Benign', 'Likely benign', 'Benign/Likely benign'] else 0)

data = data[~data['AA_Change'].str.contains('_', na=False)]

# Filtering: stop codons
data = data[~data['AA_Change'].str.startswith('*', na=False)]
data = data[~data['AA_Change'].str.endswith('*', na=False)]

data['label'] = data['Fitness_Score'] / 10
data['classification'] = data['Classification'].apply(lambda x: 1 if x == 'Functional' else 0 if x == 'Intermediate' else -1)
data = data[['AA_Change', 'classification', 'label']]

print(f'Remaining data count: {len(data)}')

# RefSeq into AA seq
refseq_dna = Seq(refseq)
refseq_protein = refseq_dna.translate()

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

data['wt'] = str(refseq_protein[:-1])
data['sequence'] = data['AA_Change'].apply(lambda x: apply_aa_change(refseq_protein, x)) # add position
data['pos'] = data['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)

def get_alphamissense_score(aa_change, df):
    try:
        pos = int(aa_change[1:-1])
        ori = aa_change[0]
        mut = aa_change[-1]

        # Filter the DataFrame
        result = df[
            (df['position'] == pos) &
            (df['a.a.1'] == ori) &
            (df['a.a.2'] == mut)
        ]

        if not result.empty:
            return result['pathogenicity score'].values[0]
        else:
            return 0.05
    except:
        return None

def get_phylop_score(aa_change, df):    
    try:
        pos = int(aa_change[1:-1]) - 1

        # Filter the DataFrame
        result = df[
            (df['Location'] == pos * 3 + 1)
        ]

        if not result.empty:
            return result['PhyloP'].values[0] / 5
        return None
    except:
        return None

data['alphamissense'] = data['AA_Change'].apply(lambda x: get_alphamissense_score(x, alphamissense))
data['phylop'] = data['AA_Change'].apply(lambda x: get_phylop_score(x, phylop))

# data['Fold'] = data['pos'].apply(lambda x: 'Test' if x in clinvar_aachange['pos'].values else 'Train')

# Step 1: Select ClinVar variants
data['Fold'] = data['AA_Change'].apply(lambda x: 'Test_ClinVar' if x in clinvar_aachange['AA_Change'].values else 'Train')

clinvar_dict = clinvar_aachange.drop_duplicates('AA_Change').set_index('AA_Change')['Germline classification'].to_dict()

data['clinvar classification'] = data.apply(
    lambda row: clinvar_dict[row['AA_Change']] if row['AA_Change'] in clinvar_dict else None, 
    axis=1
)

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Step 2: Split remaining data into internal test (5%) and training (95%)
non_clinvar_data = data[data['Fold'] != 'Test_ClinVar'].copy()
non_clinvar_indices = non_clinvar_data.index.tolist()

splitter = GroupShuffleSplit(train_size=0.95, n_splits=1, random_state=216)
split = splitter.split(non_clinvar_data, groups=non_clinvar_data['pos'])
train_indices, test_indices = next(split)

# Convert back to original DataFrame indices
train_idx = [non_clinvar_indices[i] for i in train_indices]
test_idx = [non_clinvar_indices[i] for i in test_indices]

# Assign test set
data.loc[test_idx, 'Fold'] = 'Test_internal'

# Step 3: Split training data into 5 folds
train_data = data.loc[train_idx].copy()
train_data_indices = train_data.index.tolist()

gkf = GroupKFold(n_splits=5)
for fold, (_, fold_indices) in enumerate(gkf.split(train_data, groups=train_data['pos'])):
    # Convert back to original DataFrame indices
    original_indices = [train_data_indices[i] for i in fold_indices]
    data.loc[original_indices, 'Fold'] = f'Train_{fold+1}'

# Save to CSV
data.to_csv('/extdata2/YMC/AR/data/filtered_dependency.csv', index=False)
print("Filtered data with Fold column saved successfully.")
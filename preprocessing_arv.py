import pandas as pd
from Bio.Seq import Seq

# Mutations to exclude: F877L, L702H
external_aachange = ['F877L', 'L702H']

data = pd.read_csv('/extdata2/YMC/AR/data/AR_ARV_Result2_3sd.csv')
alphamissense = pd.read_csv('/extdata2/YMC/AR/data/external/AlphaMissense-Search-P10275.csv')
phylop = pd.read_csv('/extdata2/YMC/AR/data/external/PhyloP_codon_score.csv')

refseq = 'atggaagtgcagttagggctgggaagggtctaccctcggccgccgtccaagacctaccgaggagctttccagaatctgttccagagcgtgcgcgaagtgatccagaacccgggccccaggcacccagaggccgcgagcgcagcacctcccggcgccagtttgctgctgctgcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcaagagactagccccaggcagcagcagcagcagcagggtgaggatggttctccccaagcccatcgtagaggccccacaggctacctggtcctggatgaggaacagcaaccttcacagccgcagtcggccctggagtgccaccccgagagaggttgcgtcccagagcctggagccgccgtggccgccagcaaggggctgccgcagcagctgccagcacctccggacgaggatgactcagctgccccatccacgttgtccctgctgggccccactttccccggcttaagcagctgctccgctgaccttaaagacatcctgagcgaggccagcaccatgcaactccttcagcaacagcagcaggaagcagtatccgaaggcagcagcagcgggagagcgagggaggcctcgggggctcccacttcctccaaggacaattacttagggggcacttcgaccatttctgacaacgccaaggagttgtgtaaggcagtgtcggtgtccatgggcctgggtgtggaggcgttggagcatctgagtccaggggaacagcttcggggggattgcatgtacgccccacttttgggagttccacccgctgtgcgtcccactccttgtgccccattggccgaatgcaaaggttctctgctagacgacagcgcaggcaagagcactgaagatactgctgagtattcccctttcaagggaggttacaccaaagggctagaaggcgagagcctaggctgctctggcagcgctgcagcagggagctccgggacacttgaactgccgtctaccctgtctctctacaagtccggagcactggacgaggcagctgcgtaccagagtcgcgactactacaactttccactggctctggccggaccgccgccccctccgccgcctccccatccccacgctcgcatcaagctggagaacccgctggactacggcagcgcctgggcggctgcggcggcgcagtgccgctatggggacctggcgagcctgcatggcgcgggtgcagcgggacccggttctgggtcaccctcagccgccgcttcctcatcctggcacactctcttcacagccgaagaaggccagttgtatggaccgtgtggtggtggtgggggtggtggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcgaggcgggagctgtagccccctacggctacactcggccccctcaggggctggcgggccaggaaagcgacttcaccgcacctgatgtgtggtaccctggcggcatggtgagcagagtgccctatcccagtcccacttgtgtcaaaagcgaaatgggcccctggatggatagctactccggaccttacggggacatgcgtttggagactgccagggaccatgttttgcccattgactattactttccaccccagaagacctgcctgatctgtggagatgaagcttctgggtgtcactatggagctctcacatgtggaagctgcaaggtcttcttcaaaagagccgctgaagggaaacagaagtacctgtgcgccagcagaaatgattgcactattgataaattccgaaggaaaaattgtccatcttgtcgtcttcggaaatgttatgaagcagggatgactctgggagcccggaagctgaagaaacttggtaatctgaaactacaggaggaaggagaggcttccagcaccaccagccccactgaggagacaacccagaagctgacagtgtcacacattgaaggctatgaatgtcagcccatctttctgaatgtcctggaagccattgagccaggtgtagtgtgtgctggacacgacaacaaccagcccgactcctttgcagccttgctctctagcctcaatgaactgggagagagacagcttgtacacgtggtcaagtgggccaaggccttgcctggcttccgcaacttacacgtggacgaccagatggctgtcattcagtactcctggatggggctcatggtgtttgccatgggctggcgatccttcaccaatgtcaactccaggatgctctacttcgcccctgatctggttttcaatgagtaccgcatgcacaagtcccggatgtacagccagtgtgtccgaatgaggcacctctctcaagagtttggatggctccaaatcaccccccaggaattcctgtgcatgaaagcactgctactcttcagcattattccagtggatgggctgaaaaatcaaaaattctttgatgaacttcgaatgaactacatcaaggaactcgatcgtatcattgcatgcaaaagaaaaaatcccacatcctgctcaagacgcttctaccagctcaccaagctcctggactccgtgcagcctattgcgagagagctgcatcagttcGcttttgacctgctaatcaagtcacacatggtgagcgtggactttccggaaatgatggcagagatcatctctgtgcaagtgcccaagatcctttctgggaaagtcaagcccatctatttccacacccagtga'

data = data[~data['AA_Change'].str.contains('_', na=False)]

# Filtering: stop codons
data = data[~data['AA_Change'].str.startswith('*', na=False)]
data = data[~data['AA_Change'].str.endswith('*', na=False)]

data['label'] = data['Resistance_Score'] / 4
data['classification'] = data['Classification'].apply(lambda x: 1 if x == 'Resistance' else 0 if x == 'Intermediate' else -1)
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
data['Fold'] = data['AA_Change'].apply(lambda x: 'Test_external' if x in external_aachange else 'Train')

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Step 2: Split remaining data into internal test (5%) and training (95%)
internal_data = data[data['Fold'] != 'Test_external'].copy()
internal_indices = internal_data.index.tolist()

splitter = GroupShuffleSplit(train_size=0.95, n_splits=1, random_state=608)
split = splitter.split(internal_data, groups=internal_data['pos'])
train_indices, test_indices = next(split)

# Convert back to original DataFrame indices
train_idx = [internal_indices[i] for i in train_indices]
test_idx = [internal_indices[i] for i in test_indices]

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
data.to_csv('/extdata2/YMC/AR/data/filtered_arv.csv', index=False)
print("Filtered data with Fold column saved successfully.")
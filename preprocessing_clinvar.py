import pandas as pd
from Bio.Seq import Seq

data = pd.read_csv('/extdata2/YMC/AR/data/AR_Dependency_Result3_syn5.csv')
refseq = 'atggaagtgcagttagggctgggaagggtctaccctcggccgccgtccaagacctaccgaggagctttccagaatctgttccagagcgtgcgcgaagtgatccagaacccgggccccaggcacccagaggccgcgagcgcagcacctcccggcgccagtttgctgctgctgcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcaagagactagccccaggcagcagcagcagcagcagggtgaggatggttctccccaagcccatcgtagaggccccacaggctacctggtcctggatgaggaacagcaaccttcacagccgcagtcggccctggagtgccaccccgagagaggttgcgtcccagagcctggagccgccgtggccgccagcaaggggctgccgcagcagctgccagcacctccggacgaggatgactcagctgccccatccacgttgtccctgctgggccccactttccccggcttaagcagctgctccgctgaccttaaagacatcctgagcgaggccagcaccatgcaactccttcagcaacagcagcaggaagcagtatccgaaggcagcagcagcgggagagcgagggaggcctcgggggctcccacttcctccaaggacaattacttagggggcacttcgaccatttctgacaacgccaaggagttgtgtaaggcagtgtcggtgtccatgggcctgggtgtggaggcgttggagcatctgagtccaggggaacagcttcggggggattgcatgtacgccccacttttgggagttccacccgctgtgcgtcccactccttgtgccccattggccgaatgcaaaggttctctgctagacgacagcgcaggcaagagcactgaagatactgctgagtattcccctttcaagggaggttacaccaaagggctagaaggcgagagcctaggctgctctggcagcgctgcagcagggagctccgggacacttgaactgccgtctaccctgtctctctacaagtccggagcactggacgaggcagctgcgtaccagagtcgcgactactacaactttccactggctctggccggaccgccgccccctccgccgcctccccatccccacgctcgcatcaagctggagaacccgctggactacggcagcgcctgggcggctgcggcggcgcagtgccgctatggggacctggcgagcctgcatggcgcgggtgcagcgggacccggttctgggtcaccctcagccgccgcttcctcatcctggcacactctcttcacagccgaagaaggccagttgtatggaccgtgtggtggtggtgggggtggtggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcgaggcgggagctgtagccccctacggctacactcggccccctcaggggctggcgggccaggaaagcgacttcaccgcacctgatgtgtggtaccctggcggcatggtgagcagagtgccctatcccagtcccacttgtgtcaaaagcgaaatgggcccctggatggatagctactccggaccttacggggacatgcgtttggagactgccagggaccatgttttgcccattgactattactttccaccccagaagacctgcctgatctgtggagatgaagcttctgggtgtcactatggagctctcacatgtggaagctgcaaggtcttcttcaaaagagccgctgaagggaaacagaagtacctgtgcgccagcagaaatgattgcactattgataaattccgaaggaaaaattgtccatcttgtcgtcttcggaaatgttatgaagcagggatgactctgggagcccggaagctgaagaaacttggtaatctgaaactacaggaggaaggagaggcttccagcaccaccagccccactgaggagacaacccagaagctgacagtgtcacacattgaaggctatgaatgtcagcccatctttctgaatgtcctggaagccattgagccaggtgtagtgtgtgctggacacgacaacaaccagcccgactcctttgcagccttgctctctagcctcaatgaactgggagagagacagcttgtacacgtggtcaagtgggccaaggccttgcctggcttccgcaacttacacgtggacgaccagatggctgtcattcagtactcctggatggggctcatggtgtttgccatgggctggcgatccttcaccaatgtcaactccaggatgctctacttcgcccctgatctggttttcaatgagtaccgcatgcacaagtcccggatgtacagccagtgtgtccgaatgaggcacctctctcaagagtttggatggctccaaatcaccccccaggaattcctgtgcatgaaagcactgctactcttcagcattattccagtggatgggctgaaaaatcaaaaattctttgatgaacttcgaatgaactacatcaaggaactcgatcgtatcattgcatgcaaaagaaaaaatcccacatcctgctcaagacgcttctaccagctcaccaagctcctggactccgtgcagcctattgcgagagagctgcatcagttcGcttttgacctgctaatcaagtcacacatggtgagcgtggactttccggaaatgatggcagagatcatctctgtgcaagtgcccaagatcctttctgggaaagtcaagcccatctatttccacacccagtga'
clinvar = pd.read_csv('/extdata2/YMC/AR/data/Clinvar_AR_LBD.csv')
clinvar['pos'] = clinvar['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)

clinvar_aachange = clinvar.loc[clinvar['Germline classification']!='Uncertain significance', ['AA_Change', 'pos']]
# clinvar_aachange['Germline classification'] = clinvar_aachange['Germline classification'].apply(lambda x: 1 if x in ['Benign', 'Likely benign', 'Benign/Likely benign'] else -1)

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

data = data.dropna()

# 'Fold' 컬럼 추가
data['Fold'] = data['pos'].apply(lambda x: 'Test' if x in clinvar_aachange['pos'].values else 'Train')
# data['Fold'] = data['AA_Change'].apply(lambda x: 'Test' if x in clinvar_aachange['AA_Change'].values else 'Train')

# CSV 저장
data.to_csv('/extdata2/YMC/AR/data/filtered_dependency.csv', index=False)
print("Filtered data with Fold column saved successfully.")
import pandas as pd
from Bio.Seq import Seq

clinvar = pd.read_csv('/extdata2/YMC/AR/data/Clinvar_AR_LBD.csv')
data = pd.read_csv('/extdata2/YMC/AR/data/AR_Dependency_Result3_syn5.csv')

clinvar = clinvar[clinvar['Germline classification'] != 'Uncertain significance']
clinvar = clinvar[~clinvar['AA_Change'].str.contains('_', na=False)]
clinvar = clinvar[~clinvar['AA_Change'].str.startswith('*', na=False)]
clinvar = clinvar[~clinvar['AA_Change'].str.endswith('*', na=False)]

clinvar['classification'] = clinvar['Germline classification'].apply(lambda x: 1 if x in ['Benign', 'Likely benign', 'Benign/Likely benign'] else -1)
clinvar = clinvar[['AA_Change', 'classification']]

print(f'Remaining data count: {len(clinvar)}')

# RefSeq into AA seq
refseq = 'atggaagtgcagttagggctgggaagggtctaccctcggccgccgtccaagacctaccgaggagctttccagaatctgttccagagcgtgcgcgaagtgatccagaacccgggccccaggcacccagaggccgcgagcgcagcacctcccggcgccagtttgctgctgctgcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcaagagactagccccaggcagcagcagcagcagcagggtgaggatggttctccccaagcccatcgtagaggccccacaggctacctggtcctggatgaggaacagcaaccttcacagccgcagtcggccctggagtgccaccccgagagaggttgcgtcccagagcctggagccgccgtggccgccagcaaggggctgccgcagcagctgccagcacctccggacgaggatgactcagctgccccatccacgttgtccctgctgggccccactttccccggcttaagcagctgctccgctgaccttaaagacatcctgagcgaggccagcaccatgcaactccttcagcaacagcagcaggaagcagtatccgaaggcagcagcagcgggagagcgagggaggcctcgggggctcccacttcctccaaggacaattacttagggggcacttcgaccatttctgacaacgccaaggagttgtgtaaggcagtgtcggtgtccatgggcctgggtgtggaggcgttggagcatctgagtccaggggaacagcttcggggggattgcatgtacgccccacttttgggagttccacccgctgtgcgtcccactccttgtgccccattggccgaatgcaaaggttctctgctagacgacagcgcaggcaagagcactgaagatactgctgagtattcccctttcaagggaggttacaccaaagggctagaaggcgagagcctaggctgctctggcagcgctgcagcagggagctccgggacacttgaactgccgtctaccctgtctctctacaagtccggagcactggacgaggcagctgcgtaccagagtcgcgactactacaactttccactggctctggccggaccgccgccccctccgccgcctccccatccccacgctcgcatcaagctggagaacccgctggactacggcagcgcctgggcggctgcggcggcgcagtgccgctatggggacctggcgagcctgcatggcgcgggtgcagcgggacccggttctgggtcaccctcagccgccgcttcctcatcctggcacactctcttcacagccgaagaaggccagttgtatggaccgtgtggtggtggtgggggtggtggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcggcgaggcgggagctgtagccccctacggctacactcggccccctcaggggctggcgggccaggaaagcgacttcaccgcacctgatgtgtggtaccctggcggcatggtgagcagagtgccctatcccagtcccacttgtgtcaaaagcgaaatgggcccctggatggatagctactccggaccttacggggacatgcgtttggagactgccagggaccatgttttgcccattgactattactttccaccccagaagacctgcctgatctgtggagatgaagcttctgggtgtcactatggagctctcacatgtggaagctgcaaggtcttcttcaaaagagccgctgaagggaaacagaagtacctgtgcgccagcagaaatgattgcactattgataaattccgaaggaaaaattgtccatcttgtcgtcttcggaaatgttatgaagcagggatgactctgggagcccggaagctgaagaaacttggtaatctgaaactacaggaggaaggagaggcttccagcaccaccagccccactgaggagacaacccagaagctgacagtgtcacacattgaaggctatgaatgtcagcccatctttctgaatgtcctggaagccattgagccaggtgtagtgtgtgctggacacgacaacaaccagcccgactcctttgcagccttgctctctagcctcaatgaactgggagagagacagcttgtacacgtggtcaagtgggccaaggccttgcctggcttccgcaacttacacgtggacgaccagatggctgtcattcagtactcctggatggggctcatggtgtttgccatgggctggcgatccttcaccaatgtcaactccaggatgctctacttcgcccctgatctggttttcaatgagtaccgcatgcacaagtcccggatgtacagccagtgtgtccgaatgaggcacctctctcaagagtttggatggctccaaatcaccccccaggaattcctgtgcatgaaagcactgctactcttcagcattattccagtggatgggctgaaaaatcaaaaattctttgatgaacttcgaatgaactacatcaaggaactcgatcgtatcattgcatgcaaaagaaaaaatcccacatcctgctcaagacgcttctaccagctcaccaagctcctggactccgtgcagcctattgcgagagagctgcatcagttcGcttttgacctgctaatcaagtcacacatggtgagcgtggactttccggaaatgatggcagagatcatctctgtgcaagtgcccaagatcctttctgggaaagtcaagcccatctatttccacacccagtga'
refseq_protein = Seq(refseq).translate()

def get_fitness_score(aa_change):
    matching_data = data[data['AA_Change'] == aa_change]
    if not matching_data.empty:
        return matching_data['Fitness_Score'].mean() / 10
    return None

def apply_aa_change(ref_protein, aa_change):
    mutated_seq = list(str(ref_protein))
    try:
        pos = int(aa_change[1:-1]) - 1 # Position (Zero-indexed)
        if mutated_seq[pos] == aa_change[0]:
            mutated_seq[pos] = aa_change[-1]
        return ''.join(mutated_seq[:-1])
    except:
        return None

clinvar['label'] = clinvar['AA_Change'].apply(get_fitness_score)

clinvar['label'].fillna('NA', inplace=True)
clinvar['wt'] = str(refseq_protein[:-1])
clinvar['sequence'] = clinvar['AA_Change'].apply(lambda x: apply_aa_change(refseq_protein, x)) # add position
clinvar['pos'] = clinvar['AA_Change'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else None)

# data = data.dropna()

clinvar.to_csv('/extdata2/YMC/AR/data/clinvar_test.csv', index=False)
print("Updated clinvar with label column successfully.")

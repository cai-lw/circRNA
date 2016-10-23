from Bio import SeqIO
import csv
from itertools import chain

# {chromosome:[(start, end, '+'/'-', group)]}
pos_loci = {}
neg_loci = {}
alu_loci = {}

group = 0
with open('raw_data/hsa_hg19_Rybak2015.bed', 'rb') as csvfile:
    bed = csv.reader(csvfile, delimiter="\t")
    for rec in bed:
        t = (int(rec[1]), int(rec[2]), rec[5], group)
        if t[1] <= t[0]:
            continue
        group = (group + 1) % 10
        try:
            pos_loci[rec[0]].append(t)
        except KeyError:
            pos_loci[rec[0]] = [t]

with open('raw_data/all_exons.bed', 'rb') as csvfile:
    bed = csv.reader(csvfile, delimiter=" ")
    for rec in bed:
        t = (int(rec[2]), int(rec[4]), rec[6], group)
        if t[1] <= t[0]:
            continue
        group = (group + 1) % 10
        try:
            neg_loci[rec[0]].append(t)
        except KeyError:
            neg_loci[rec[0]] = [t]

with open('raw_data/hg19_Alu.bed', 'rb') as csvfile:
    bed = csv.reader(csvfile, delimiter='\t')
    for rec in bed:
        t = (int(rec[1]), int(rec[2]), rec[5])
        if t[1] <= t[0]:
            continue
        try:
            alu_loci[rec[0]].append(t)
        except KeyError:
            alu_loci[rec[0]] = [t]

print("Pos Num: %d" % sum(map(len, pos_loci.values())))
print("Pos Max Len: %d" % max(map(lambda t:t[1] - t[0], chain.from_iterable(pos_loci.values()))))
print("Neg Num: %d" % sum(map(len, neg_loci.values())))
print("Neg Max Len: %d" % max(map(lambda t:t[1] - t[0], chain.from_iterable(neg_loci.values()))))
print("Alu Num: %d" % sum(map(len, alu_loci.values())))

f_pos = [open("clean_data/pos%d.txt" % i, "w") for i in range(10)]
f_neg = [open("clean_data/neg%d.txt" % i, "w") for i in range(10)]

for chromo in SeqIO.parse("raw_data/hg19.fa", "fasta"):
    pseq = chromo.seq.upper().tomutable()
    nseq = chromo.seq.upper().tomutable()
    pos_recs = pos_loci.get(chromo.id, [])
    neg_recs = neg_loci.get(chromo.id, [])
    alu_recs = alu_loci.get(chromo.id, [])
    for rec in alu_recs:
        if rec[2] == '+':
            pseq[rec[0]:rec[1]] = str(pseq[rec[0]:rec[1]]).lower()
        else:
            nseq[rec[0]:rec[1]] = str(nseq[rec[0]:rec[1]]).lower()
    for rec in pos_recs:
        if rec[2] == '+':
            dna = str(pseq[rec[0]:rec[1]])
        else:
            dna = str(nseq[rec[0]:rec[1]].toseq().reverse_complement())
        f_pos[rec[3]].write(dna + '\n')
    for rec in neg_recs:
        if rec[2] == '+':
            dna = str(pseq[rec[0]:rec[1]])
        else:
            dna = str(nseq[rec[0]:rec[1]].toseq().reverse_complement())
        f_neg[rec[3]].write(dna + '\n')

for i in range(10):
    f_pos[i].close()
    f_neg[i].close()

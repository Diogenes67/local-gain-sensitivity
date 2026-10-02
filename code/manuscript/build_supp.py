"""Assemble Document S1 (supplement_patterns.md) for the Patterns submission from the NMI/JMLR markdown sources in src/.

Cell Press order: supplemental figures (S1-S11, each with title, "related to" and legend), supplemental tables (S1-S7),
supplemental notes (1-5; note 5 is the experimental record), supplemental methods (the full methods sections whose
condensed forms are in the Experimental procedures), supplemental references ([S1], [S2], ... in order of first citation).
Cross-references are renamed from the Nature forms (Extended Data Fig. N, Supplementary Note N, Methods) to the Cell forms.

Run: python build_supp.py  -> src/supplement_patterns.md, src/main_figs.md (figure pages appended to the main PDF)
"""
import os, re, sys
H = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(H, "src")
sys.path.insert(0, SRC)
from refs import REFS

FIGMAP = {"1": "S1", "2": "S2", "3": "S3", "4": "S4", "5": "S5", "6": "S6", "7": "S8", "8": "S9", "9": "S10", "10": "S11"}
TABMAP = {"1": "S1", "2": "S2", "3": "S7", "4": "S3", "5": "S4"}
FIG_FILES = {"S1": "edfig1_rerun", "S2": "edfig2_new", "S3": "edfig3_new", "S4": "edfig4_nmi", "S5": "edfig5_new", "S6": "edfig6_new",
             "S7": "edfig6b_new", "S8": "edfig7_new", "S9": "edfig8_new", "S10": "edfig9_arch", "S11": "edfig10_kgram"}
FIG_WIDTH = {"S1": "95%", "S2": "66%", "S3": "85%", "S4": "78%", "S5": "72%", "S6": "78%", "S7": "78%", "S8": "68%", "S9": "85%", "S10": "80%", "S11": "85%"}
FIG_RELATED = {"S1": "Figure 1", "S2": "Figure 1 and Table 1", "S3": "Figure 5", "S4": "Figures 4 and 5", "S5": "Figure 4", "S6": "Figure 5",
               "S7": "Figure 5", "S8": "Figures 2 and 3", "S9": "Figure 4", "S10": "Figure 4", "S11": "Figure 4"}
TAB_RELATED = {"S1": "Figure 5", "S2": "Figure 5", "S3": "Figure 2", "S4": "Figure 4", "S5": "Figure 4", "S6": "Figure 4", "S7": "Figures 1–5"}
TAB_FILES = {"S1": "ed_table1_nmi.md", "S2": "ed_table2.md", "S3": "ed_table4.md", "S4": "ed_table5.md", "S7": "ed_table3_nmi.md"}
TAB_FONT = {"S1": "scriptsize", "S2": "tiny", "S3": "scriptsize", "S4": "scriptsize", "S5": "footnotesize", "S6": "footnotesize", "S7": "tiny"}
LAND = {"S2", "S3", "S7"}


def read(name):
    return open(os.path.join(SRC, name), encoding="utf-8").read()


def rename(t):
    t = t.replace("Extended Data Fig. 6 (continued)", "Figure S7")
    t = t.replace("Extended Data Figs. 2 to 7, 9 and 10, and Extended Data Tables 1 to 5", "Figures S2 to S8, S10 and S11, and Tables S1 to S4 and S7")
    t = t.replace("Extended Data Figs. 2–7, 9 and 10 and Extended Data Tables 1–5", "Figures S2–S8, S10 and S11 and Tables S1–S4 and S7")
    t = re.sub(r"Extended Data Figs?\. (\d+)", lambda m: "Figure " + FIGMAP[m.group(1)], t)
    t = re.sub(r"\bED Fig\. (\d+)", lambda m: "Figure " + FIGMAP[m.group(1)], t)
    t = re.sub(r"\bED Tables? (\d)", lambda m: "Table " + TABMAP[m.group(1)], t)
    t = re.sub(r"Extended Data Tables? (\d)", lambda m: "Table " + TABMAP[m.group(1)], t)
    t = re.sub(r"Supplementary Notes? (\d)", r"Supplemental note \1", t)
    t = t.replace("Supplementary Notes", "Supplemental notes")
    t = re.sub(r"Supplementary Tables? (\d)", lambda m: "Table S" + str(int(m.group(1)) + 4), t)
    t = t.replace("Supplementary Information", "Supplemental information").replace("Supplementary Tables", "Tables S5 and S6")
    t = re.sub(r"\bFigs\.? (\d)", r"Figures \1", t)
    t = re.sub(r"\bFig\. (\d)", r"Figure \1", t)
    t = re.sub(r"\bMethods\b", "Experimental procedures", t)
    t = t.replace("Extended Data", "supplemental information")
    return t


# ---------------------------------------------------------------- supplemental references: {{key}} -> [S#]
cited = []


def cite(t):
    def rep(m):
        k = m.group(1)
        if k not in cited: cited.append(k)
        return f"^[S{cited.index(k) + 1}]^"
    return re.sub(r"\{\{([a-z0-9]+)\}\}", rep, t)


# ---------------------------------------------------------------- figures
leg = read("legends_nmi.md")
figs = []
for m in re.finditer(r"^\*\*Extended Data Fig\. (\d+)( \(continued\))? \| (.+?)\*\* (.*)$", leg, re.M):
    n, cont, title, body = m.groups()
    s = "S7" if cont else FIGMAP[n]
    figs.append((s, title.rstrip("."), body))
assert [f[0] for f in figs] == [f"S{i}" for i in range(1, 12)], [f[0] for f in figs]
out = ["# Supplemental information", "", "**When local amplification predicts downstream sensitivity in neural networks**", "",
       "Anthony Porter, Nadhir Hassen, Hemanth Saratchandran, Zhibin Liao, Johan Verjans and Anton van den Hengel", "",
       "Document S1. Figures S1–S11, Tables S1–S7, Supplemental notes 1–5, supplemental methods and supplemental references.", "",
       "Supplemental notes 1 to 4 report retained analyses from the May 2026 runs that the methods cite for design details and controls; the main results do not rest on them, and the provenance of each is stated at its head. Supplemental note 5 records which run each reported result comes from, what it replaced, what can be recomputed from the deposited files and what has been withdrawn. The supplemental methods give in full the protocols that the Experimental procedures summarise.", "",
       "\\newpage", "", "# Supplemental figures", ""]
for s, title, body in figs:
    out += [f"![](fig/{FIG_FILES[s]}.png){{width={FIG_WIDTH[s]}}}", "",
            f"**Figure {s}. {rename(title)}, related to {FIG_RELATED[s]}.** {cite(rename(body))}", "", "\\newpage", ""]

# ---------------------------------------------------------------- tables
tab_leg = {}
for m in re.finditer(r"^\*\*Extended Data Table (\d) \| (.+?)\*\* (.*)$", leg, re.M):
    tab_leg[TABMAP[m.group(1)]] = (m.group(2).rstrip("."), m.group(3))
sup = read("supplement_nmi.md")
for m in re.finditer(r"^\*\*Supplementary Table (\d) \| (.+?)\*\* (.*)$", sup, re.M):
    tab_leg["S" + str(int(m.group(1)) + 4)] = (m.group(2).rstrip("."), m.group(3))


def table_body(md):
    """the pipe table and any footnote paragraphs of a table file (title lines removed)"""
    lines = md.strip().split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("|"))
    return "\n".join(lines[i:]).strip()


def sup_table(n):
    m = re.search(r"^\*\*Supplementary Table %d \|.*?\n\n(\|.*?)(?:\n\n|\Z)" % n, sup, re.S | re.M)
    return m.group(1).strip()


out += ["# Supplemental tables", ""]
for s in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
    title, body = tab_leg[s]
    if s in TAB_FILES:
        tb = table_body(read(TAB_FILES[s]))
    else:
        tb = sup_table(int(s[1:]) - 4)
    land_pre = ["```{=latex}", "\\begin{landscape}", "```", ""] if s in LAND else []
    land_post = ["```{=latex}", "\\end{landscape}", "```", ""] if s in LAND else []
    out += land_pre + [f"**Table {s}. {rename(title)}, related to {TAB_RELATED[s]}.** {cite(rename(body))}", "", "```{=latex}", "\\begingroup\\%s" % TAB_FONT[s], "```", "",
            cite(rename(tb)), "", "```{=latex}", "\\endgroup", "```", ""] + land_post + ["\\newpage", ""]

# ---------------------------------------------------------------- notes 1-5
notes_block = sup[sup.index("# Supplementary Notes"):sup.index("# Supplementary Tables")]
notes_block = notes_block.replace("# Supplementary Notes", "# Supplemental notes")
# note 5 of the NMI file (provenance) becomes the second half of the Patterns note 5, after the record sections
i5 = notes_block.index("**Supplementary Note 5:")
notes14 = notes_block[:i5]
note5_prov = notes_block[i5:]
note5_prov = note5_prov.replace("**Supplementary Note 5: Provenance of every reported result**\n\nThis note is the single record of which run each number in the paper comes from, which earlier runs it replaces, and what can be recomputed from the deposited files. The Methods describe only the protocol in force; the history is here.",
                                "**Provenance of every reported result.** The paragraphs below record, result by result, which run each number in the paper comes from, which earlier runs it replaces, and what can be recomputed from the deposited files (Table S7).")
assert "Provenance of every reported result." in note5_prov
record = read("record_patterns.md")
out += [cite(rename(notes14)).strip(), "", cite(rename(record)).strip(), "", cite(rename(note5_prov)).strip(), "", "\\newpage", ""]

# ---------------------------------------------------------------- supplemental methods
met = read("methods_nmi.md")
parts = re.split(r"^### (.+?)\s*$", met, flags=re.M)
secs = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}
SKIP = {"Hook-based spectral estimation", "Pretrained model panel", "Statistics", "Software", "Data availability", "Code availability"}
# the joint baseline is added to the held-out-prediction paragraph
secs["Singular-direction perturbation and residual-gain stress"] = secs["Singular-direction perturbation and residual-gain stress"].replace(
    "A coefficient per block in place of the linear depth term changes the σ~1~ set by at most 0.04.",
    "A joint baseline of depth with the norm and the random-direction gain, and the same set with σ~1~ added, were run on the same folds (26 September 2026): per unit displacement the joint baseline gives 0.42 with one model held out and 0.55 with σ~1~ (paired difference 0.14, 95% CI 0.07 to 0.21); 0.18 and 0.13 with one condition held out (−0.05, −0.40 to 0.22; the four text conditions pooled 0.25 and 0.19, per held-out condition real text −0.25 and 0.48, shuffled text 0.61 and 0.90, random labels −0.76 and −3.00, bidirectional 0.87 and 0.85; the six k-gram orders pooled 0.04 and 0.01); 0.17 and 0.08 with the text models predicting the k-gram models (−0.09, −0.28 to 0.05); and −6.61 and −8.46 with the k-gram models predicting the text models (−1.85, −3.49 to −0.80), every set containing the norm failing in that direction. On the same-position readout the joint baseline gives 0.52 and 0.64 with one model held out (0.12, 0.05 to 0.19) and 0.14 and 0.53 with one condition held out (0.38, 0.19 to 0.66). Table S3 lists every value. A coefficient per block in place of the linear depth term changes the σ~1~ set by at most 0.04.")
assert "joint baseline" in secs["Singular-direction perturbation and residual-gain stress"]
secs["Matched assay in pretrained decoders"] = secs["Matched assay in pretrained decoders"].replace(
    "and over the model folds 0.07, 0.06, 0.07 and 0.24 (0.14 to 0.30).",
    "and over the model folds 0.07, 0.06, 0.07 and 0.24 (0.14 to 0.30); the joint baseline of depth with the norm and the random-direction gain gives −0.00 over the family folds and 0.22 with σ~1~ added (paired difference over 2,000 resamples of held-out models 0.22, −0.01 to 0.42), and 0.05 and 0.11 over the model folds (0.06, −0.29 to 0.23).")
assert "joint baseline" in secs["Matched assay in pretrained decoders"]
secs["Estimator validation"] = "The token-local validation is reported in the Experimental procedures; the comparisons below give the full-scan check for Mamba-130M and the estimator comparisons in detail.\n\n" + secs["Estimator validation"]
out += ["# Supplemental methods", "",
        "These sections give in full the protocols that the Experimental procedures summarise, in the order of the Experimental procedures where one exists. Every profile in this paper is a token-local J^T^J power iteration in float32 unless a legend states otherwise (Table S7).", ""]
for title, body in secs.items():
    if title in SKIP: continue
    out += [f"## {rename(title)}", "", cite(rename(body)), ""]

# ---------------------------------------------------------------- supplemental references
out += ["# Supplemental references", ""]
for i, k in enumerate(cited):
    out.append(f"S{i + 1}. {REFS[k]}")
    out.append("")
text = "\n".join(out)
open(os.path.join(SRC, "supplement_patterns.md"), "w", encoding="utf-8").write(text)
left = sorted(set(re.findall(r"Extended Data[^.;)]{0,20}|Supplementary [A-Z][a-z]+|\bFig\. |\bMethods\b", text)))
print("supplement written:", len(text.split()), "words; unrenamed forms:", left)
print("supplemental references:", cited)

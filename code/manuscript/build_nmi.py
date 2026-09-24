"""Build the NMI manuscript docx from nmi/main_nmi.md, nmi/methods_nmi.md and nmi/legends_nmi.md (run assemble_nmi.py first).
Same citation, figure and table machinery as ../build.py. Figures not yet rebuilt for the NMI layout are embedded from the
NCS set and marked as interim beneath the image."""
import re, subprocess, os, sys, copy
D = os.path.dirname(os.path.abspath(__file__)); NCS = os.path.dirname(D)
sys.path.insert(0, NCS)
from refs import REFS
os.chdir(NCS)

main = open(os.path.join(D, 'main_nmi.md')).read()
methods = open(os.path.join(D, 'methods_nmi.md')).read()
legends = open(os.path.join(D, 'legends_nmi.md')).read()
body = main + "\n\n" + methods

order = []
def key_num(k):
    if k not in order:
        order.append(k)
    return order.index(k) + 1

def fmt_nums(nums):
    nums = sorted(set(nums)); out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j > i else f"{nums[i]}"))
        i = j + 1
    return ",".join(out)

def repl(m):
    keys = re.findall(r'\{\{([a-z0-9]+)\}\}', m.group(0))
    for k in keys:
        if k not in REFS:
            raise SystemExit(f"unknown ref key {k}")
    return "^" + fmt_nums([key_num(k) for k in keys]) + "^"

body = re.sub(r'(\{\{[a-z0-9]+\}\})+', repl, body)
refs_md = "## References\n\n" + "\n\n".join(f"{i+1}. {REFS[k]}" for i, k in enumerate(order))

def img(name, w="6.5in"):
    return f'![](fig/{name}.png){{width="{w}"}}\n\n'
INTERIM = '*[Interim figure from the NCS draft; to be rebuilt for the layout described in this legend.]*\n\n'
fig_blocks = {
    'Fig. 1': img('fig1_nmi', '6.5in'),
    'Fig. 2': img('fig2_scale_nmi', '6.5in'),
    'Fig. 3': img('fig3_direction_nmi', '6.5in'),
    'Fig. 4': img('fig4_data_nmi', '6.5in'),
    'Fig. 5': img('fig5_pretrained_nmi', '6.5in'),
    'Extended Data Fig. 1': img('edfig1_mamba'), 'Extended Data Fig. 2': img('edfig2_new', '6.2in'), 'Extended Data Fig. 3': img('edfig3_new'),
    'Extended Data Fig. 4': img('edfig4_nmi'), 'Extended Data Fig. 5': img('edfig5_new'), 'Extended Data Fig. 6': img('edfig6_new', '6.2in'),
    'Extended Data Fig. 6 (continued)': img('edfig6b_new', '6.2in'), 'Extended Data Fig. 7': img('edfig7_new'), 'Extended Data Fig. 8': img('edfig8_new'),
    'Extended Data Fig. 9': img('edfig9_arch'), 'Extended Data Fig. 10': img('edfig10_kgram'),
}
for blk in fig_blocks.values():
    for n in re.findall(r'fig/([a-z0-9_]+)\.png', blk):
        if not os.path.exists(f'fig/{n}.png'):
            raise SystemExit(f'missing fig/{n}.png')
TABLES = {'Extended Data Table 1': 'tables/ed_table1_nmi.md', 'Extended Data Table 2': 'tables/ed_table2.md', 'Extended Data Table 3': 'tables/ed_table3_nmi.md',
          'Extended Data Table 4': 'tables/ed_table4.md', 'Extended Data Table 5': 'tables/ed_table5.md'}   # 3-5 from make_ed_tables_nmi.py
out = []
for p in legends.split('\n\n'):
    m = re.match(r'\*\*((?:Extended Data )?Fig\. \d+(?: \(continued\))?) \|', p)
    if m and m.group(1) in fig_blocks:
        out.append(fig_blocks[m.group(1)].strip())
    out.append(p)
    t = re.match(r'\*\*(Extended Data Table \d) \|', p)
    if t and t.group(1) in TABLES:
        out.append(open(TABLES[t.group(1)]).read().strip())
legends_with_figs = "\n\n".join(out)
# Table 1 heading is a level-2 heading in the source; keep it, it sits after the main legends
full = body + "\n\n" + refs_md + "\n\n" + legends_with_figs
sys.path.insert(0, D)
import copyedit_patch
full = copyedit_patch.apply(full, 'full')   # Nature copy-edit, 24 Sept 2026 (nmi/copyedit_patch.json)
open(os.path.join(D, 'hourglass_NMI_draft.md'), 'w').write(full)

def wc(s):
    s = re.sub(r'\^[\d,–]+\^', '', s); s = re.sub(r'\[NEW EXPERIMENT.*?\]', '', s, flags=re.S); s = re.sub(r'[#*]', '', s); return len(s.split())
print("abstract words:", wc(full.split('## Abstract')[1].split('## Main')[0]))
print("main text words (after copy-edit):", wc(full.split('## Main')[1].split('## Methods')[0]))
print("references:", len(order))

docx_path = os.path.join(D, 'hourglass_NMI_draft.docx')
subprocess.run(['pandoc', os.path.join(D, 'hourglass_NMI_draft.md'), '-o', docx_path, '--from', 'markdown+superscript+subscript+pipe_tables', '--resource-path', NCS], check=True)

from docx import Document
from docx.shared import Pt, Inches, Emu
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
doc = Document(docx_path)
paras = doc.paragraphs
i_t1 = next(i for i, p in enumerate(paras) if p.text.startswith('Extended Data Table 1 |'))
body_sectPr = doc.element.body.find(qn('w:sectPr'))
pPr = paras[i_t1 - 1]._p.get_or_add_pPr(); pPr.append(copy.deepcopy(body_sectPr))
for s_ in doc.sections[:-1]:
    s_.page_width, s_.page_height = Inches(8.27), Inches(11.69)
    for m_ in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(s_, m_, Inches(0.9))
sec = doc.sections[-1]; sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Inches(11.69), Inches(8.27)
for m_ in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
    setattr(sec, m_, Inches(0.6))
land = sec.page_width - sec.left_margin - sec.right_margin
port = Inches(8.27) - Inches(1.8)
WIDTHS = {10: [1.6, 1.3, 0.6, 0.8, 0.6, 0.9, 0.9, 0.9, 0.8, 0.9],
          16: [1.7, 1.2, 0.5, 1.4, 0.6, 0.5, 0.7, 0.6, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.8, 0.6],
          9: [1.5, 1.4, 1.3, 1.3, 0.7, 1.4, 1.6, 0.9, 2.4],
          5: [0.9, 1.3, 1.4, 2.6, 0.8],   # Table 1 (portrait)
          6: [1.3, 2.0, 1.9, 1.9, 1.9, 1.8],   # ED Table 4
          8: [1.3, 1.9, 0.3, 1.1, 0.9, 0.9, 1.2, 1.2]}   # ED Table 5
for t in doc.tables:
    ncol = len(t.columns); rel = WIDTHS.get(ncol, [1.0] * ncol); tot = sum(rel)
    usable = port if ncol == 5 else land
    t.autofit = False
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); t._tbl.tblPr.append(lay)
    for row in t.rows:
        for j, cell in enumerate(row.cells):
            cell.width = Emu(int(usable * rel[j] / tot))
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(7.5 if ncol == 5 else 7)
    grid = t._tbl.find(qn('w:tblGrid'))
    if grid is not None:
        for j, gc in enumerate(grid.findall(qn('w:gridCol'))):
            gc.set(qn('w:w'), str(int(usable * rel[j] / tot / 635)))   # EMU -> twips
    tw = t._tbl.tblPr.find(qn('w:tblW'))
    if tw is not None:
        tw.set(qn('w:type'), 'dxa'); tw.set(qn('w:w'), str(int(usable / 635)))
    trPr = t.rows[0]._tr.get_or_add_trPr(); hdr = OxmlElement('w:tblHeader'); hdr.set(qn('w:val'), 'true'); trPr.append(hdr)
n_fig = 0; prev_img = False
for p in doc.paragraphs:
    has = bool(p._p.findall('.//' + qn('w:drawing')))
    if has:
        if not prev_img:
            p.paragraph_format.page_break_before = True
        p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(6); n_fig += 1
    prev_img = has
doc.save(docx_path)
print(f"built {docx_path}; {n_fig} figure images")

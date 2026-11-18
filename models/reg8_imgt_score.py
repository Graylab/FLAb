"""Sequence-only regional biophysical + IMGT germline Z-score baseline (reg8_bp_imgt_z).

Computes an equal-weight (1/8 each) average of eight sequence descriptors standardized
solely against the external human IMGT VH x VL germline repertoire (24 VH x 18 VL = 432
reference pairs, zero assay-label fitting):
  1. Framework Vihinen flexibility (+fwr_flex)
  2. Framework hydrophilicity (-fwr_gravy)
  3. Framework Guruprasad stability (-fwr_instability_index)
  4. CDR low aromaticity (-cdr_aromaticity)
  5. CDR low isoelectric point (-cdr_pI)
  6. CDR low net positive charge at pH 7.4 (-cdr_charge_at_7_4)
  7. Chain molecular weight (+mw)
  8. Anchor-aligned human IMGT V+J germline sequence identity (+germ)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from biopython_score import (
        bp_aromaticity,
        bp_average_flexibility,
        bp_charge_at_7_4,
        bp_gravy,
        bp_instability_index,
        bp_isoelectric_point,
        bp_molecular_weight,
    )
except ImportError:
    # Pure-Python exact ProtParam fallback when Biopython is not installed
    _KD = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
        "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
        "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    _FLEX = {
        "A": 0.984, "C": 0.906, "E": 1.094, "D": 1.068, "G": 1.031, "F": 0.915,
        "I": 0.927, "H": 0.95, "K": 1.102, "M": 0.952, "L": 0.935, "N": 1.048,
        "Q": 1.037, "P": 1.049, "S": 1.046, "R": 1.008, "T": 0.997, "W": 0.904,
        "V": 0.931, "Y": 0.929,
    }
    _MW = {
        "A": 89.0932, "C": 121.1582, "D": 133.1027, "E": 147.1293, "F": 165.1891,
        "G": 75.0666, "H": 155.1546, "I": 131.1729, "K": 146.1876, "L": 131.1729,
        "M": 149.2113, "N": 132.1179, "P": 115.1305, "Q": 146.1445, "R": 174.201,
        "S": 105.0926, "T": 119.1192, "V": 117.1463, "W": 204.2252, "Y": 181.1885,
    }
    _POS_PK = {"Nterm": 7.5, "K": 10.0, "R": 12.0, "H": 5.98}
    _NEG_PK = {"Cterm": 3.55, "D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
    _PK_CTERM = {"D": 4.55, "E": 4.75}
    _PK_NTERM = {"A": 7.59, "M": 7.0, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.7}
    _DIWV = {
        "A": {"A": 1.0, "C": 44.94, "E": 1.0, "D": -7.49, "G": 1.0, "F": 1.0, "I": 1.0, "H": -7.49, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},
        "C": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 20.26, "G": 1.0, "F": 1.0, "I": 1.0, "H": 33.6, "K": 1.0, "M": 33.6, "L": 20.26, "N": 1.0, "Q": -6.54, "P": 20.26, "S": 1.0, "R": 1.0, "T": 33.6, "W": 24.68, "V": -6.54, "Y": 1.0},
        "E": {"A": 1.0, "C": 44.94, "E": 33.6, "D": 20.26, "G": 1.0, "F": 1.0, "I": 20.26, "H": -6.54, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},
        "D": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 1.0, "S": 20.26, "R": -6.54, "T": -14.03, "W": 1.0, "V": 1.0, "Y": 1.0},
        "G": {"A": -7.49, "C": 1.0, "E": -6.54, "D": 1.0, "G": 13.34, "F": 1.0, "I": -7.49, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": -7.49, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -7.49, "W": 13.34, "V": 1.0, "Y": -7.49},
        "F": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 13.34, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -14.03, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 33.601},
        "I": {"A": 1.0, "C": 1.0, "E": 44.94, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 13.34, "K": -7.49, "M": 1.0, "L": 20.26, "N": 1.0, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},
        "H": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": -9.37, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 24.68, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": -6.54, "W": -1.88, "V": 1.0, "Y": 44.94},
        "K": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": -7.49, "H": 1.0, "K": 1.0, "M": 33.6, "L": -7.49, "N": 1.0, "Q": 24.64, "P": -6.54, "S": 1.0, "R": 33.6, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},
        "M": {"A": 13.34, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 58.28, "K": 1.0, "M": -1.88, "L": 1.0, "N": 1.0, "Q": -6.54, "P": 44.94, "S": 44.94, "R": -6.54, "T": -1.88, "W": 1.0, "V": 1.0, "Y": 24.68},
        "L": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 33.6, "P": 20.26, "S": 1.0, "R": 20.26, "T": 1.0, "W": 24.68, "V": 1.0, "Y": 1.0},
        "N": {"A": 1.0, "C": -1.88, "E": 1.0, "D": 1.0, "G": -14.03, "F": -14.03, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 1.0, "Q": -6.54, "P": -1.88, "S": 1.0, "R": 1.0, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 1.0},
        "Q": {"A": 1.0, "C": -6.54, "E": 20.26, "D": 20.26, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 1.0, "T": 1.0, "W": 1.0, "V": -6.54, "Y": -6.54},
        "P": {"A": 20.26, "C": -6.54, "E": 18.38, "D": -6.54, "G": 1.0, "F": 20.26, "I": 1.0, "H": 1.0, "K": 1.0, "M": -6.54, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": -6.54, "T": 1.0, "W": -1.88, "V": 20.26, "Y": 1.0},
        "S": {"A": 1.0, "C": 33.6, "E": 20.26, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 44.94, "S": 20.26, "R": 20.26, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},
        "R": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": 1.0, "H": 20.26, "K": 1.0, "M": 1.0, "L": 1.0, "N": 13.34, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 58.28, "T": 1.0, "W": 58.28, "V": 1.0, "Y": -6.54},
        "T": {"A": 1.0, "C": 1.0, "E": 20.26, "D": 1.0, "G": -7.49, "F": 13.34, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": -14.03, "Q": -6.54, "P": 1.0, "S": 1.0, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},
        "W": {"A": -14.03, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": 1.0, "I": 1.0, "H": 24.68, "K": 1.0, "M": 24.68, "L": 13.34, "N": 13.34, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -14.03, "W": 1.0, "V": -7.49, "Y": 1.0},
        "V": {"A": 1.0, "C": 1.0, "E": 1.0, "D": -14.03, "G": -7.49, "F": 1.0, "I": 1.0, "H": 1.0, "K": -1.88, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": -7.49, "W": 1.0, "V": 1.0, "Y": -6.54},
        "Y": {"A": 24.68, "C": 1.0, "E": -6.54, "D": 24.68, "G": -7.49, "F": 1.0, "I": 1.0, "H": 13.34, "K": 1.0, "M": 44.94, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 13.34, "S": 1.0, "R": -15.91, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 13.34},
    }

    def bp_gravy(s: str) -> float:
        return sum(_KD[a] for a in s) / len(s)

    def bp_aromaticity(s: str) -> float:
        return (s.count("Y") + s.count("W") + s.count("F")) / len(s)

    def bp_molecular_weight(s: str) -> float:
        return sum(_MW[a] for a in s) - (len(s) - 1) * 18.0153

    def bp_instability_index(s: str) -> float:
        return (10.0 / len(s)) * sum(_DIWV[s[i]][s[i + 1]] for i in range(len(s) - 1))

    def bp_average_flexibility(s: str) -> float:
        w = (0.25, 0.4375, 0.625, 0.8125, 1.0)
        n_win = len(s) - 9
        tot = 0.0
        for i in range(n_win):
            sub = s[i : i + 9]
            sc = sum((_FLEX[sub[j]] + _FLEX[sub[8 - j]]) * w[j] for j in range(4)) + _FLEX[sub[5]]
            tot += sc / 5.25
        return tot / n_win

    def _chg(s: str, pH: float) -> float:
        ch = {aa: float(s.count(aa)) for aa in ("K", "R", "H", "D", "E", "C", "Y")}
        ch["Nterm"] = 1.0
        ch["Cterm"] = 1.0
        pos_p = _POS_PK.copy()
        neg_p = _NEG_PK.copy()
        if s[0] in _PK_NTERM:
            pos_p["Nterm"] = _PK_NTERM[s[0]]
        if s[-1] in _PK_CTERM:
            neg_p["Cterm"] = _PK_CTERM[s[-1]]
        return sum(ch[a] / (10.0 ** (pH - pK) + 1.0) for a, pK in pos_p.items()) - sum(
            ch[a] / (10.0 ** (pK - pH) + 1.0) for a, pK in neg_p.items()
        )

    def bp_charge_at_7_4(s: str) -> float:
        return _chg(s, 7.4)

    def bp_isoelectric_point(s: str) -> float:
        lo, hi, pH = 4.05, 12.0, 7.775
        while hi - lo > 0.0001:
            if _chg(s, pH) > 0.0:
                lo = pH
            else:
                hi = pH
            pH = 0.5 * (lo + hi)
        return pH

HUMAN_VH_GERMLINES = [
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR",
    "EVQLVESGGGLVKPGGSLRLSCAASGFTFSNAWMSWVRQAPGKGLEWVGRIKSKTDGGTTDYAAPVKGRFTISRDDSKNTLYLQMNSLKTEDTAVYYCTT",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSYISSSSSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR",
    "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    "QVQLVESGGGLVKPGGSLRLSCAASGFTFSDYYMSWVRQAPGKGLEWVSYISSSGSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR",
    "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR",
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGRINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQATGQGLEWMGWMNPNSGNTGYAQKFQGRVTMTRNTSISTAYMELSSLRSEDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKVSGYTLTELSMHWVRQAPGKGLEWMGGFDPEDGETIYAQKFQGRVTMTEDTSTDTAYMELSSLRSEDTAVYYCAT",
    "QVQLQQWGAGLLKPSETLSLTCAVYGGSFSGYYWSWIRQPPGKGLEWIGEINHSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR",
    "QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR",
    "QVQLQESGPGLVKPSGTLSLTCAVSGYSISSSNWWGWVRQPPGKGLEWIGEIYHSGSTNYNPSLKSRVTISVDKSKNQFSLKLSSVTAADTAVYYCAR",
    "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR",
    "QITLKESGPTLVKPTQTLTLTCTFSGFSLSTSGVGVGWIRQPPGKALEWLALIYWDDDKRYSPSLKSRLTITKDTSKNQVVLTMTNMDPVDTATYYCAH",
    "QVTLKESGPVLVKPTETLTLTCTVSGFSLSNARMGVSWIRQPPGKALEWLAHIFSNDEKSYSTSLKSRLTISKDTSKSQVVLTMTNMDPVDTATYYCAR",
    "QVQLVQSGSELKKPGASVKVSCKASGYTFTSYAMNWVRQAPGQGLEWMGWINTNTGNPTYAQGFTGRFVFSLDTSVSTAYLQICSLKAEDTAVYYCAR",
]
HUMAN_JH_GERMLINES = ["DYWGQGTLVTVSS", "DIWGQGTMVTVSS", "DVWGQGTTVTVSS", "DPWGQGTLVTVSS", "DLWGRGTLVTVSS"]

HUMAN_VL_GERMLINES = [
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLT",
    "DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLP",
    "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYS",
    "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSFP",
    "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQKYNSAP",
    "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTP",
    "DVVMTQSPLSLPVTLGQPASISCRSSQSLVYSDGNTYLNWFQQRPGQSPRRLIYKVSNRDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQGTHWP",
    "EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWP",
    "EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISSLQSEDFAVYYCQQYNNWP",
    "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSP",
    "DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTP",
    "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYGNSNRPSGVPDRFSGSKSGTSASLAITGLQAEDEADYYCQSYDSSLSG",
    "QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNTVNWYQQLPGTAPKLLIYSNNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEADYYCAAWDDSLNG",
    "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYEVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTL",
    "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSYTF",
    "QSALTQPRSVSGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVPDRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSYTF",
    "SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAEDEADYYCQAWDSSTA",
    "SSELTQDPAVSVALGQTVRITCQGDSLRSYYASWYQQKPGQAPVLVIYGKNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEADYYCNSRDSSGNH",
]
HUMAN_JL_GERMLINES = ["FGQGTKVEIK", "FGQGTKLEIK", "FGPGTKVDIK", "FGGGTKVEIK", "FGQGTRLEIK", "FGGGTKLTVL", "FGTGTKVTVL"]

REG8_KEYS = (
    "fwr_flex",
    "neg_fwr_gravy",
    "neg_fwr_inst",
    "neg_cdr_arom",
    "neg_cdr_pI",
    "neg_cdr_chg",
    "mw",
    "germ",
)


def _clean_aa(seq: str) -> str:
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(seq).upper())
    if len(s) < 10:
        raise KeyError(f"Sequence too short after cleaning (len={len(s)})")
    return s


def _get_anchors(seq: str) -> tuple[str, int, int, int, int]:
    n = len(seq)
    if n < 70:
        return seq, min(21, n - 1), min(35, n - 1), max(0, n - 15), max(0, n - 10)
    c1 = 21
    for i in range(15, min(30, n)):
        if seq[i] == "C":
            c1 = i
            break
    w = c1 + 13
    for i in range(c1 + 8, min(c1 + 24, n - 2)):
        if seq[i] == "W" and seq[i + 1] in "VIYFLM" and seq[i + 2] in "RKQ":
            w = i
            break
    else:
        for i in range(c1 + 9, min(c1 + 22, n)):
            if seq[i] == "W":
                w = i
                break
    c2 = max(w + 40, n - 22)
    for i in range(w + 36, min(n - 8, w + 70)):
        if seq[i] == "C" and (
            seq[i - 2 : i] in ("YY", "YF", "YI", "VY", "VF", "TY", "FC", "TH") or seq[i - 1] in "YF"
        ):
            c2 = i
            break
    j = min(n, c2 + 12)
    for i in range(c2 + 3, min(n - 3, c2 + 32)):
        if seq[i] in "WF" and seq[i + 1] == "G" and seq[i + 3] == "G":
            j = i
            break
    return seq, c1, w, c2, j


_VG_ANCHORS_H = [(idx, vg, *_get_anchors(vg)[1:]) for idx, vg in enumerate(HUMAN_VH_GERMLINES)]
_VG_ANCHORS_L = [(idx, vg, *_get_anchors(vg)[1:]) for idx, vg in enumerate(HUMAN_VL_GERMLINES)]
_CHAIN_CACHE: dict[tuple[str, bool], tuple[str, str, float, float]] = {}
_FWR_CACHE: dict[str, tuple[float, float, float]] = {}
_CDR_CACHE: dict[str, tuple[float, float, float]] = {}


def _germline_identity(seq: str, is_heavy: bool) -> float:
    _, c1, w, c2, j = _get_anchors(seq)
    l3 = 38 if is_heavy else 34
    offset = -2 if is_heavy else 0
    vg_list = _VG_ANCHORS_H if is_heavy else _VG_ANCHORS_L
    j_list = HUMAN_JH_GERMLINES if is_heavy else HUMAN_JL_GERMLINES
    best_v = 0.0
    for _, vg_s, vc1, vw, vc2, _ in vg_list:
        matches = total = 0
        for d in range(-min(c1, vc1), 9):
            if 0 <= c1 + d < len(seq) and 0 <= vc1 + d < len(vg_s):
                total += 1
                matches += int(seq[c1 + d] == vg_s[vc1 + d])
        for d in range(-5, 23):
            if 0 <= w + d < len(seq) and 0 <= vw + d < len(vg_s):
                total += 1
                matches += int(seq[w + d] == vg_s[vw + d])
        for d in range(-l3, 8):
            if 0 <= c2 + d < len(seq) and 0 <= vc2 + d < len(vg_s):
                total += 1
                matches += int(seq[c2 + d] == vg_s[vc2 + d])
        best_v = max(best_v, matches / max(total, 1))
    best_j = 0.0
    for jg in j_list:
        m4 = sum(
            1 for k in range(len(jg)) if 0 <= j + offset + k < len(seq) and seq[j + offset + k] == jg[k]
        )
        best_j = max(best_j, m4 / len(jg))
    return (best_v * 95.0 + best_j * 12.0) / 107.0


def _split_fwr_cdr(seq: str) -> tuple[str, str]:
    _, c1, w, c2, j = _get_anchors(seq)
    fwr = seq[: c1 + 4] + seq[w - 1 : w + 14] + seq[w + 24 : c2 + 1] + seq[j:]
    cdr = seq[c1 + 4 : w - 1] + seq[w + 14 : w + 24] + seq[c2 + 1 : j]
    return fwr, cdr


def _chain_info(seq: str, is_heavy: bool) -> tuple[str, str, float, float]:
    key = (seq, is_heavy)
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    fwr, cdr = _split_fwr_cdr(seq)
    val = (fwr, cdr, _germline_identity(seq, is_heavy), bp_molecular_weight(seq))
    _CHAIN_CACHE[key] = val
    return val


def reg8_descriptors(heavy: str, light: str | None = None) -> dict[str, float]:
    sh = _clean_aa(heavy)
    has_l = light is not None and pd.notna(light) and len(str(light).strip()) >= 40
    fwr_h, cdr_h, gh, mw_h = _chain_info(sh, True)
    if has_l:
        sl = _clean_aa(light)
        fwr_l, cdr_l, gl, mw_l = _chain_info(sl, False)
        mw = 0.5 * (mw_h + mw_l)
        germ = 0.5 * (gh + gl)
        fwr_seq = fwr_h + fwr_l
        cdr_seq = cdr_h + cdr_l
    else:
        mw = mw_h
        germ = gh
        fwr_seq = fwr_h
        cdr_seq = cdr_h

    if fwr_seq not in _FWR_CACHE:
        _FWR_CACHE[fwr_seq] = (
            bp_average_flexibility(fwr_seq),
            -bp_gravy(fwr_seq),
            -bp_instability_index(fwr_seq),
        )
    fwr_flex, neg_fwr_gravy, neg_fwr_inst = _FWR_CACHE[fwr_seq]

    cdr_eval = cdr_seq if len(cdr_seq) >= 10 else sh
    if cdr_eval not in _CDR_CACHE:
        _CDR_CACHE[cdr_eval] = (
            -bp_aromaticity(cdr_eval),
            -bp_isoelectric_point(cdr_eval),
            -bp_charge_at_7_4(cdr_eval),
        )
    neg_cdr_arom, neg_cdr_pI, neg_cdr_chg = _CDR_CACHE[cdr_eval]

    return {
        "fwr_flex": fwr_flex,
        "neg_fwr_gravy": neg_fwr_gravy,
        "neg_fwr_inst": neg_fwr_inst,
        "neg_cdr_arom": neg_cdr_arom,
        "neg_cdr_pI": neg_cdr_pI,
        "neg_cdr_chg": neg_cdr_chg,
        "mw": mw,
        "germ": germ,
    }


_IMGT_REF_DF = pd.DataFrame(
    [
        reg8_descriptors(vh + "DYWGQGTLVTVSS", vl + "FGQGTKVEIK")
        for vh in HUMAN_VH_GERMLINES
        for vl in HUMAN_VL_GERMLINES
    ]
)
IMGT_MEAN = {k: float(_IMGT_REF_DF[k].mean()) for k in REG8_KEYS}
IMGT_STD = {k: float(_IMGT_REF_DF[k].std()) for k in REG8_KEYS}


def reg8_bp_imgt_z_pair(heavy: str, light: str | None = None) -> float:
    d = reg8_descriptors(heavy, light)
    return float(sum((d[k] - IMGT_MEAN[k]) / IMGT_STD[k] for k in REG8_KEYS) / len(REG8_KEYS))

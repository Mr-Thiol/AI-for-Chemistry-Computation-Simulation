import os
import sys
import json
import math
import traceback
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


# -------------------------
# Paths and environment
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.getenv("PROJECT_DIR", os.path.abspath(os.path.join(BASE_DIR, "..", "..")))  # HW4/
CHEMFORMER_DIR = os.path.join(PROJECT_DIR, "Chemformer")
DATASET_CSV = os.getenv("USPTO_CSV_PATH", os.path.join(PROJECT_DIR, "USPTO-50K.csv"))
CKPT_PATH = os.getenv("CHEMFORMER_CKPT", os.path.join(PROJECT_DIR, "fine_tune_upsto_50_last_v2.ckpt"))
VOCAB_PATH = os.getenv("CHEMFORMER_VOCAB", os.path.join(CHEMFORMER_DIR, "bart_vocab_downstream.json"))

if CHEMFORMER_DIR not in sys.path:
    sys.path.insert(0, CHEMFORMER_DIR)


# -------------------------
# Lazy model holders
# -------------------------
_ESTIMATOR = None
_CHEMFORMER = None


# -------------------------
# Request / Response models
# -------------------------
class PreviewRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=600)


class RetrosynthesisRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=600)
    n_beams: int = Field(default=5, ge=1, le=15)
    optimize_confs: int = Field(default=30, ge=5, le=80)


class MoleculeCard(BaseModel):
    name: str
    smiles: str
    descriptors: Dict[str, float]
    molblock_3d: str
    energy_kcal: Optional[float]


# -------------------------
# Helper functions
# -------------------------
def canon_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol


def canon_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def make_2d_svg(mol: Chem.Mol, w: int = 340, h: int = 260, legend: str = "") -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.padding = 0.12
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def compute_desc(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "mw": round(float(Descriptors.MolWt(mol)), 2),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": round(float(Descriptors.TPSA(mol)), 2),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "heavy_atoms": int(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
    }


def optimize_3d(mol: Chem.Mol, n_confs: int = 30) -> Tuple[Chem.Mol, Optional[float]]:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params))
    if not conf_ids:
        # fallback single conformer
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == -1:
            raise ValueError("3D embedding failed")
        if AllChem.UFFHasAllMoleculeParams(mol_h):
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
            ff = AllChem.UFFGetMoleculeForceField(mol_h)
            e = float(ff.CalcEnergy()) if ff else None
            return mol_h, e
        return mol_h, None

    ff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
    energies: List[Tuple[float, int]] = []

    for cid in conf_ids:
        if ff_props:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, ff_props, confId=cid)
            if ff:
                ff.Minimize(maxIts=2000)
                energies.append((float(ff.CalcEnergy()), int(cid)))
            else:
                energies.append((float("inf"), int(cid)))
        else:
            if AllChem.UFFHasAllMoleculeParams(mol_h):
                AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=2000)
                ff_u = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
                e = float(ff_u.CalcEnergy()) if ff_u else float("inf")
                energies.append((e, int(cid)))
            else:
                energies.append((float("inf"), int(cid)))

    energies.sort(key=lambda x: x[0])
    best_energy, best_cid = energies[0]

    # Keep only best conformer
    out = Chem.Mol(mol_h)
    for cid in [x[1] for x in energies if x[1] != best_cid]:
        try:
            out.RemoveConformer(int(cid))
        except Exception:
            pass

    if math.isinf(best_energy):
        return out, None
    return out, round(best_energy, 4)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -------------------------
# Estimator from USPTO stats
# -------------------------
class SuccessEstimator:
    def __init__(self, csv_path: str):
        self.stats = {
            "mw_mean": 250.0,
            "mw_std": 90.0,
            "heavy_mean": 18.0,
            "heavy_std": 7.0,
            "rings_mean": 1.8,
            "rings_std": 1.2,
        }
        self._fit(csv_path)

    def _fit(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            return
        try:
            df = pd.read_csv(csv_path)
            if "input" not in df.columns:
                return

            vals = []
            sample = df["input"].dropna().astype(str).head(5000)
            for smi in sample:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                vals.append(
                    (
                        float(Descriptors.MolWt(m)),
                        float(rdMolDescriptors.CalcNumHeavyAtoms(m)),
                        float(rdMolDescriptors.CalcNumRings(m)),
                    )
                )

            if len(vals) < 100:
                return

            arr = pd.DataFrame(vals, columns=["mw", "heavy", "rings"])
            self.stats = {
                "mw_mean": float(arr["mw"].mean()),
                "mw_std": float(arr["mw"].std() + 1e-6),
                "heavy_mean": float(arr["heavy"].mean()),
                "heavy_std": float(arr["heavy"].std() + 1e-6),
                "rings_mean": float(arr["rings"].mean()),
                "rings_std": float(arr["rings"].std() + 1e-6),
            }
        except Exception:
            # Keep fallback stats
            pass

    def estimate(self, mol: Chem.Mol) -> Dict[str, object]:
        d = compute_desc(mol)

        z_mw = abs((d["mw"] - self.stats["mw_mean"]) / self.stats["mw_std"])
        z_hv = abs((d["heavy_atoms"] - self.stats["heavy_mean"]) / self.stats["heavy_std"])
        z_rg = abs((d["rings"] - self.stats["rings_mean"]) / self.stats["rings_std"])

        # Base confidence from distance to training-set center
        score = 0.9 - 0.12 * z_mw - 0.10 * z_hv - 0.08 * z_rg

        reasons: List[str] = []

        # Practical heuristics for retrosynthesis quality
        if d["mw"] < 130:
            score -= 0.22
            reasons.append("Molecule is very small; model often returns trivial variants.")
        if d["heavy_atoms"] < 10:
            score -= 0.18
            reasons.append("Too few heavy atoms for rich disconnection patterns.")
        if d["rings"] >= 4:
            score -= 0.08
            reasons.append("Many ring systems can reduce one-step prediction reliability.")
        if d["aromatic_rings"] >= 3:
            score -= 0.06
            reasons.append("Highly aromatic systems may produce substitution-like pseudo-retrosynthesis.")
        if d["mw"] > 650:
            score -= 0.15
            reasons.append("Large molecule may exceed model comfort zone in sequence modeling.")

        prob = clamp(score, 0.03, 0.98)

        if prob >= 0.75:
            level = "high"
            msg = "Likely suitable for Chemformer single-step retrosynthesis."
        elif prob >= 0.45:
            level = "medium"
            msg = "Partially suitable; inspect beams carefully."
        else:
            level = "low"
            msg = "High failure risk; output may be trivial or invalid."

        return {
            "success_probability": round(prob, 3),
            "risk_level": level,
            "message": msg,
            "reasons": reasons,
            "training_stats": self.stats,
            "descriptors": d,
        }


# -------------------------
# Chemformer runner
# -------------------------
def get_estimator() -> SuccessEstimator:
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = SuccessEstimator(DATASET_CSV)
    return _ESTIMATOR


def get_chemformer():
    global _CHEMFORMER
    if _CHEMFORMER is not None:
        return _CHEMFORMER

    try:
        import molbart.utils.data_utils as util  # noqa: F401
        from omegaconf import OmegaConf
        from molbart.models import Chemformer

        cfg = OmegaConf.create(
            {
                "train_mode": "eval",
                "batch_size": 1,
                "n_gpus": 0,
                "n_beams": 5,
                "n_unique_beams": None,
                "vocabulary_path": VOCAB_PATH,
                "model_path": CKPT_PATH,
                "model_type": "bart",
                "task": "backward_prediction",
                "data_path": None,
                "dataset_part": "full",
                "i_chunk": 0,
                "n_chunks": 1,
                "datamodule": None,
                "scorers": None,
                "output_sampled_smiles": None,
            }
        )
        _CHEMFORMER = Chemformer(cfg)
        return _CHEMFORMER
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chemformer: {e}")


def run_retrosynthesis(smiles: str, n_beams: int) -> List[Dict[str, object]]:
    import molbart.utils.data_utils as util
    from molbart.data import SynthesisDataModule

    chemformer = get_chemformer()

    # Update beam size dynamically
    chemformer.sampler.sample_unique = False
    if hasattr(chemformer.sampler, "beam_size"):
        chemformer.sampler.beam_size = int(n_beams)

    dm = SynthesisDataModule(
        reactants=[smiles],
        products=[smiles],
        tokenizer=chemformer.tokenizer,
        batch_size=1,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
        dataset_path="",
    )
    dm.setup()

    beams, lhs, _ = chemformer.predict(dataloader=dm.full_dataloader())

    out = []
    for i, (smi, ll) in enumerate(zip(beams[0], lhs[0])):
        parts = [x for x in str(smi).split(".") if x.strip()]
        valid = all(Chem.MolFromSmiles(x) is not None for x in parts)
        out.append(
            {
                "beam": i + 1,
                "reactants_smiles": str(smi),
                "log_likelihood": round(float(ll), 6),
                "n_reactants": len(parts),
                "reactants": parts,
                "valid_smiles": bool(valid),
            }
        )
    return out


def build_molecule_cards(
    target_smiles: str, beam_rows: List[Dict[str, object]], optimize_confs: int
) -> List[Dict[str, object]]:
    registry: Dict[str, str] = {}

    target_m = canon_mol(target_smiles)
    target_c = canon_smiles(target_m)
    registry[target_c] = "Target"

    for row in beam_rows:
        b = int(row["beam"])
        for j, smi in enumerate(row["reactants"], start=1):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if c not in registry:
                registry[c] = f"Beam {b} Reactant {j}"

    cards: List[Dict[str, object]] = []
    for c_smi, name in registry.items():
        m = Chem.MolFromSmiles(c_smi)
        if m is None:
            continue
        d = compute_desc(m)
        try:
            m3d, e = optimize_3d(m, n_confs=optimize_confs)
            molblock = Chem.MolToMolBlock(m3d)
        except Exception:
            molblock = ""
            e = None

        cards.append(
            {
                "name": name,
                "smiles": c_smi,
                "descriptors": d,
                "molblock_3d": molblock,
                "energy_kcal": e,
            }
        )
    return cards


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Chemformer Retrosynthesis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "chemformer_ckpt_exists": os.path.exists(CKPT_PATH),
        "vocab_exists": os.path.exists(VOCAB_PATH),
        "dataset_exists": os.path.exists(DATASET_CSV),
    }


@app.post("/preview")
def preview(req: PreviewRequest) -> Dict[str, object]:
    try:
        mol = canon_mol(req.smiles.strip())
        c_smi = canon_smiles(mol)
        estimator = get_estimator()
        est = estimator.estimate(mol)

        mol3d, e = optimize_3d(mol, n_confs=20)
        return {
            "input_smiles": req.smiles,
            "canonical_smiles": c_smi,
            "descriptors": est["descriptors"],
            "success_estimate": {
                "probability": est["success_probability"],
                "risk_level": est["risk_level"],
                "message": est["message"],
                "reasons": est["reasons"],
            },
            "preview_2d_svg": make_2d_svg(mol, legend="Input molecule"),
            "preview_3d_molblock": Chem.MolToMolBlock(mol3d),
            "preview_3d_energy_kcal": e,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrosynthesis")
def retrosynthesis(req: RetrosynthesisRequest) -> Dict[str, object]:
    try:
        mol = canon_mol(req.smiles.strip())
        c_smi = canon_smiles(mol)
        estimator = get_estimator()
        est = estimator.estimate(mol)

        rows = run_retrosynthesis(c_smi, n_beams=req.n_beams)
        cards = build_molecule_cards(c_smi, rows, optimize_confs=req.optimize_confs)

        valid_count = sum(1 for r in rows if r["valid_smiles"])
        return {
            "input_smiles": req.smiles,
            "canonical_smiles": c_smi,
            "success_estimate": {
                "probability": est["success_probability"],
                "risk_level": est["risk_level"],
                "message": est["message"],
                "reasons": est["reasons"],
            },
            "retrosynthesis": {
                "n_beams": req.n_beams,
                "valid_beams": valid_count,
                "rows": rows,
            },
            "molecule_cards": cards,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrosynthesis failed: {e}")


@app.get("/example")
def example() -> Dict[str, object]:
    # metoclopramide example
    return {
        "smiles": "CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl",
        "hint": "POST this to /preview or /retrosynthesis",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

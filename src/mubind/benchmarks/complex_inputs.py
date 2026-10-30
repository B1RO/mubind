"""Prepare binder-first, target-second PDB folders for redesign evaluation."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

from .common import DATA, read, require, save, sha256


def default_complexes():
    return DATA / 'ppiref/complexes'


def inspect_complex(path):
    from Bio.PDB import is_aa
    from Bio.SeqUtils import seq1
    from ..common.sequences import normalize_sequence
    from ..inference.helpers.targets import first_model

    model = first_model(path)
    chains = [chain for chain in model if any(is_aa(residue) for residue in chain)]
    require(len(chains) == 2, f'{path.name} must contain exactly two protein chains, binder first and target second')
    sequences = []
    for index, chain in enumerate(chains):
        residues = [residue for residue in chain if residue.id[0] == ' ' or is_aa(residue)]
        require(
            all(residue.id[0] == ' ' and is_aa(residue, standard=True) for residue in residues),
            f'{path.name} chain {chain.id} contains non-standard amino acids; provide canonical protein residues',
        )
        if index == 0:
            require(
                all(all(atom in residue for atom in ('N', 'CA', 'C')) for residue in residues),
                f'{path.name} binder must have N, CA and C coordinates for every residue',
            )
        sequences.append(normalize_sequence(''.join(seq1(residue.resname) for residue in residues)))
    require(1 <= len(sequences[0]) <= 70, f'{path.name} first chain is the binder and must contain 1–70 residues')
    return {
        'ppi_id': path.stem,
        'filename': path.name,
        'binder_chain': chains[0].id,
        'target_chain': chains[1].id,
        'binder_observed_sequence': sequences[0],
        'target_sequence': sequences[1],
    }


def inspect_complexes(directory, mode):
    require(directory.is_dir(), f'Complexes folder not found: {directory}')
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.pdb'), key=lambda p: p.name)
    require(paths, f'No PDB complexes in {directory}')
    manifest_path = directory / 'manifest.json'
    manifest = read(manifest_path) if manifest_path.is_file() else {}
    references = manifest.get('cases', {}) if manifest.get('dataset') == 'ppiref45' else {}
    cases = []
    for path in paths:
        require(path.stem not in {'.', '..'}, f'PDB filename needs a case name: {path.name}')
        case = inspect_complex(path)
        if manifest.get('dataset') == 'ppiref45':
            reference = references.get(path.stem)
            require(reference is not None and sha256(path) == reference['sha256'],
                    f'PPIRef input differs from its recorded structure: {path.name}')
            require(case['binder_observed_sequence'] == reference['binder_sequence']
                    and case['target_sequence'] == reference['target_observed_sequence'],
                    f'PPIRef sequences differ from the original inputs: {path.name}')
            # Original folding used the complete target construct, including
            # residues absent from the experimental coordinates.
            case['target_observed_sequence'] = case['target_sequence']
            case['target_sequence'] = reference['target_sequence']
        if mode == 'inpainting':
            length = len(case['binder_observed_sequence'])
            require(math.comb(length, (length + 1) // 2) >= 20,
                    f'{path.name} binder is too short for 20 distinct half-sequence masks; at least 6 residues are required')
        cases.append({**case, 'source_sha256': sha256(path)})
    require(len({case['ppi_id'] for case in cases}) == len(cases), 'PDB filenames must have distinct stems')
    return cases


def prepare_complexes(directory, out, inspected, foldseek, on_status):
    """Snapshot inputs and encode each isolated binder for this run."""
    from ..common.properties import sequence_properties
    from ..dataset.embed import encode_3di
    from ..evaluation.foldseek import read_foldseek_records
    from ..inference.helpers.targets import write_chain

    saved = out / 'cases.json'
    originals, binders = out / 'complexes', out / 'binders'
    originals.mkdir(parents=True, exist_ok=True)
    binders.mkdir(exist_ok=True)
    for case in inspected:
        snapshot = originals / case['filename']
        shutil.copyfile(directory / case['filename'], snapshot)
        require(sha256(snapshot) == case['source_sha256'], f'Input changed during preparation: {case["filename"]}')
        write_chain(snapshot, binders / (case['ppi_id'] + '.pdb'), case['binder_chain'])
    on_status(f'Encoding binder structures for {len(inspected)} complexes with Foldseek…')
    database = out / 'foldseek'
    with (out / 'foldseek.log').open('w') as log:
        subprocess.run([str(foldseek), 'createdb', str(binders), str(database),
                        '--chain-name-mode', '1', '--threads', '4', '-v', '1'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    encoded = read_foldseek_records(database)
    require(len(encoded) == len(inspected), 'Foldseek did not encode every input binder')
    cases = []
    for case in inspected:
        ident, chain = case['ppi_id'], case['binder_chain']
        keys = {ident + '_' + chain, ident + '.pdb_' + chain}
        matches = [value for key, value in encoded.items() if key in keys]
        require(len(matches) == 1, f'Foldseek binder record missing or ambiguous for {case["filename"]}')
        sequence, three = matches[0]
        require(sequence == case['binder_observed_sequence'] and len(three) == len(sequence),
                f'Foldseek sequence or 3Di length mismatch for {case["filename"]}')
        encode_3di(three, pad_to=70)
        reference = binders / (ident + '.pdb')
        cases.append({
            **case,
            'input_3di': three,
            'requested_properties': sequence_properties(sequence, property_definition='omegamp_modlamp_v1'),
            'reference_binder_pdb': str(reference.resolve()),
            'reference_binder_chain': chain,
            'reference_binder_sha256': sha256(reference),
        })
    save(saved, {'inputs': inspected, 'cases': cases})
    return cases

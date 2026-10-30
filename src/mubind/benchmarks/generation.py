"""Construct benchmark tasks for the current inference API."""

from __future__ import annotations

from .common import PROPERTIES, prostt5_targets, require

def base_tasks(count, torch):
    return {
        "seed_x": torch.zeros(count, 8, 70),
        "seed_3di": torch.zeros(count, 70, dtype=torch.long),
        "edit_mask": torch.ones(count, 70, dtype=torch.bool),
        "c": torch.zeros(count, 3),
        "condition_mask": torch.zeros(count, 3),
    }


def build_jobs(mode, torch, normalized=True, *, complexes=None, property_requests=None, seed=None):
    """Return exact batch boundaries, seeds, task tensors and record metadata."""
    from ..dataset.embed import encode_3di

    if complexes is not None:
        return build_complex_jobs(mode, complexes, torch, normalized)
    jobs = []
    if mode == "unconditional":
        batches = []
        for start in range(0, 1024, 16):
            tasks = base_tasks(16, torch)
            tasks["threedi_conditioning_hidden_mask"] = torch.zeros(
                16, 70, dtype=torch.bool
            )
            tasks["validation_case_id"] = [
                f"unconditional_{i:04d}" for i in range(start, start + 16)
            ]
            batches.append(tasks)
        jobs.append(("unconditional", 20261006, 1.0, batches, None))
    elif mode == "properties":
        require(property_requests is not None and len(property_requests) == 4000,
                "Property control needs 4,000 sampled training requests")
        require(seed is not None, "Property control needs a recorded sampling seed")
        records = property_requests
        batches = []
        for start in range(0, 4000, 200):
            rows = records[start : start + 200]
            tasks = base_tasks(200, torch)
            tasks["c"] = torch.tensor(
                [[r[f"requested_{p}"] for p in PROPERTIES] for r in rows]
            )
            tasks["condition_mask"] = torch.tensor(
                [[float(c) for c in r["condition_mask"]] for r in rows]
            )
            tasks["validation_case_id"] = [r["case_id"] for r in rows]
            tasks["property_conditioning"] = [r["property_conditioning"] for r in rows]
            tasks["target_length"] = tasks["c"][:, 0].long()
            tasks["dataset_index"] = torch.tensor([r["dataset_index"] for r in rows])
            batches.append(tasks)
        jobs.append(("properties", seed, 1.0, batches, None))
    elif mode == "structure":
        targets = prostt5_targets()
        for replicate in range(1, 11):
            for arm, scale in (("baseline", 1.0), ("guidance5", 5.0)):
                tasks = base_tasks(92, torch)
                tasks["seed_3di"] = torch.stack(
                    [encode_3di(r["target_3di"], pad_to=70) for r in targets]
                )
                tasks["validation_case_id"] = [
                    f"{r['target_id']}_rep_{replicate:02d}" for r in targets
                ]
                tasks["threedi_target_id"] = [r["target_id"] for r in targets]
                tasks["candidate_replicate"] = torch.full(
                    (92,), replicate, dtype=torch.long
                )
                tasks["threedi_conditioning"] = ["full_3di"] * 92
                tasks["binder_aa"] = [r["native_sequence"] for r in targets]
                tasks["target_length"] = torch.tensor([r["length"] for r in targets])
                jobs.append(
                    (
                        f"{arm}_{replicate:02d}",
                        20261008 + replicate,
                        scale,
                        [tasks],
                        None,
                    )
                )
    else:
        require(mode not in {"full-design", "inpainting"},
                "Redesign and inpainting require prepared complex inputs")
        raise ValueError(f"Unknown evaluation mode: {mode}")
    return jobs


def complex_masks(cases, torch):
    """Keep the paper's half-sequence span sampler and 20 distinct masks per case."""
    from ..training.tasks.config import TaskConfig
    from ..training.tasks.spans import sample_edit_spans_for_budget
    from .common import require

    config = TaskConfig(span_count_geometric_probability=0.5, dirichlet_concentration=0.5)
    masks = []
    for index, case in enumerate(cases):
        sequence = case['binder_observed_sequence']
        length, budget = len(sequence), (len(sequence) + 1) // 2
        seen = set()
        for number in range(1, 21):
            seed = 202610071000 + index * 1000 + number
            rng = torch.Generator(device='cpu').manual_seed(seed)
            for attempt in range(1, 10001):
                spans = sample_edit_spans_for_budget(length, budget, config, device=torch.device('cpu'), generator=rng)
                edit = torch.zeros(length, dtype=torch.bool)
                for start, end in spans:
                    edit[start:end] = True
                signature = ''.join('1' if value else '0' for value in edit.tolist())
                if signature not in seen:
                    break
            else:
                raise ValueError(f'Could not draw 20 distinct masks for {case["ppi_id"]}')
            require(int(edit.sum()) == budget, 'Inpainting mask budget mismatch')
            seen.add(signature)
            masks.append({
                'ppi_id': case['ppi_id'], 'candidate_id': f'candidate_{number:02d}',
                'mask_seed': seed, 'edit_mask': signature,
                'mask_draws_including_duplicate_rejections': attempt,
                'spans_0based_halfopen': spans,
                'edited_residue_count': budget, 'retained_residue_count': length - budget,
                'partial_sequence': ''.join('?' if edit[i] else aa for i, aa in enumerate(sequence)),
            })
    return masks


def build_complex_jobs(mode, cases, torch, normalized):
    from ..common.properties import condition_tensor
    from ..dataset.embed import encode_3di, embed_diffusion_sequence
    from .checkpoint_generation import inpainting_canvas
    from .common import require

    require(mode in {'full-design', 'inpainting'} and cases, 'Complex inputs require full-design or inpainting')
    lookup = {case['ppi_id']: case for case in cases}
    if mode == 'full-design':
        specs = [{'ppi_id': case['ppi_id'], 'candidate_id': f'candidate_{number:02d}'}
                 for case in cases for number in range(1, 51)]
    else:
        specs = complex_masks(cases, torch)
    jobs = []
    for batch_index, start in enumerate(range(0, len(specs), 100)):
        metadata = specs[start:start + 100]
        tasks = base_tasks(len(metadata), torch)
        for index, spec in enumerate(metadata):
            case = lookup[spec['ppi_id']]
            sequence, three = case['binder_observed_sequence'], case['input_3di']
            tasks['seed_3di'][index] = encode_3di(three, pad_to=70)
            tasks['c'][index] = condition_tensor(sequence, property_definition='omegamp_modlamp_v1')
            if mode == 'inpainting':
                source = embed_diffusion_sequence(sequence, three, pad_to=70, normalized=normalized, joint_3di=True).T.contiguous()
                canvas = inpainting_canvas(source, tasks['seed_3di'][index], seed_length=len(sequence),
                                           internal_edit_mask=torch.tensor([bit == '1' for bit in spec['edit_mask']]))
                for key in ('seed_x', 'seed_3di', 'edit_mask'):
                    tasks[key][index] = canvas[key]
        tasks['condition_mask'].fill_(1)
        seed = (20261011 if mode == 'full-design' else 20261031) + batch_index
        jobs.append((f'batch_{batch_index + 1:04d}', seed, 5.0, [tasks], metadata))
    return jobs

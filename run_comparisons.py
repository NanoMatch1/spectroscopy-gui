#!/usr/bin/env python3
"""Load, validate, and export THz TDS analysis comparison data."""

import logging
import os
import sys

import numpy as np

from export import export_dataset, verify_roundtrip
from loaders import load_all, sanity_check

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JSON = os.path.join(BASE_DIR, 'sample_details.json')
EXPORT_DIR = os.path.join(BASE_DIR, 'exports')


def _section(title: str):
    logger.info('')
    logger.info('=' * 60)
    logger.info(title)
    logger.info('=' * 60)


def main() -> int:
    gui_mode = '--gui' in sys.argv

    # ── Load ─────────────────────────────────────────────────────────────
    _section('Loading datasets')
    datasets = load_all(BASE_DIR, SAMPLE_JSON)

    # ── Summary ──────────────────────────────────────────────────────────
    _section('Dataset summary')
    for ds in datasets:
        logger.info(f'\n  {ds.name}:')
        logger.info(f'    Sample: thickness={ds.sample_info.thickness_m * 1e6:.1f} μm, '
                     f'resistivity={ds.sample_info.resistivity_ohm_m} Ω·m')
        logger.info(f'    Raw reference: {len(ds.raw_reference.time_ps)} pts  '
                     f'[{ds.raw_reference.time_ps[0]:.2f} – {ds.raw_reference.time_ps[-1]:.2f}] ps')
        logger.info(f'    Raw sample:    {len(ds.raw_sample.time_ps)} pts  '
                     f'[{ds.raw_sample.time_ps[0]:.2f} – {ds.raw_sample.time_ps[-1]:.2f}] ps')

        if ds.time_domain:
            td = ds.time_domain
            logger.info(f'    Time-domain:   ref={len(td.ref_time_ps)} pts, '
                         f'sample={len(td.sample_time_ps)} pts')
            if td.raw_ref_time_ps is not None:
                logger.info(f'      (raw ref={len(td.raw_ref_time_ps)} pts, '
                             f'raw sample={len(td.raw_sample_time_ps)} pts)')
        else:
            logger.info('    Time-domain:   None')

        if ds.fft:
            fft = ds.fft
            logger.info(f'    FFT:           {len(fft.ref_frequency_thz)} pts, '
                         f'freq [{fft.ref_frequency_thz[0]:.4f} – '
                         f'{fft.ref_frequency_thz[-1]:.4f}] THz')
        else:
            logger.info('    FFT:           None')

        if ds.optical_constants:
            oc = ds.optical_constants
            valid_n = oc.n[np.isfinite(oc.n)]
            logger.info(f'    Optical const: {len(oc.frequency_thz)} pts, '
                         f'n [{np.nanmin(valid_n):.3f} – {np.nanmax(valid_n):.3f}]')
            if oc.eps_infty is not None:
                logger.info(f'      eps_infty = {oc.eps_infty}')
        else:
            logger.info('    Optical const: None')

    # ── Sanity checks ────────────────────────────────────────────────────
    _section('Sanity checks')
    all_warnings: list[str] = []
    for ds in datasets:
        warnings = sanity_check(ds)
        all_warnings.extend(warnings)
        if warnings:
            for w in warnings:
                logger.warning(f'  ⚠ {w}')
        else:
            logger.info(f'  [{ds.name}] All checks passed')

    # ── Export to CSV ────────────────────────────────────────────────────
    _section('Exporting to CSV')
    for ds in datasets:
        person_dir = os.path.join(EXPORT_DIR, ds.name)
        export_dataset(ds, person_dir)

    # ── Roundtrip verification ───────────────────────────────────────────
    _section('Roundtrip verification')
    all_issues: list[str] = []
    for ds in datasets:
        person_dir = os.path.join(EXPORT_DIR, ds.name)
        issues = verify_roundtrip(ds, person_dir)
        all_issues.extend(issues)

    if all_issues:
        logger.error('')
        logger.error('ROUNDTRIP ISSUES:')
        for issue in all_issues:
            logger.error(f'  ✗ {issue}')
        return 1

    logger.info('  All roundtrip checks passed')

    # ── Final summary ────────────────────────────────────────────────────
    _section('Done')
    if all_warnings:
        logger.info(f'Completed with {len(all_warnings)} warning(s)')
    else:
        logger.info('Completed successfully — no warnings')

    # ── Launch GUI if requested ──────────────────────────────────────────
    if gui_mode:
        from gui import launch
        launch(datasets)

    return 0


if __name__ == '__main__':
    sys.exit(main())

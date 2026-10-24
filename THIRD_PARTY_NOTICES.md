# Third-party components and access

## Original code and independent figure rights

The existing repository's MIT LICENSE and copyright notice are retained
unchanged. For the new release, that software license covers original Python
source under `scripts/` and `tests/` only. It does not relicense third-party
material, datasets, pretrained weights, manuscript text, or figures. This scope
statement does not revoke license grants for previously published material.

The three files in `assets/` are architecture illustrations and an aggregate
results plot. They contain no patient images. They are excluded from the MIT
software license, and no separate figure reuse license is granted here.
Figure copyright and any publication rights remain with the relevant rights
holders. Obtain their permission for reuse where required.

## MRI-CORE

Official code: <https://github.com/mazurowski-lab/mri_foundation>.
Paper: <https://arxiv.org/abs/2506.12186>.

The official README describes an Apache 2.0 release, but no LICENSE, NOTICE,
or COPYING file was present in the remote tree inspected at commit
`0f4f452614013873f8bb58a1fd613dbe36b58aba`, or in the local research dependency.
Some source headers refer to an absent root license. This repository therefore
does not redistribute the MRI-CORE/SAM source or any pretrained weights, and
does not assert that a blanket Apache license covers them. Obtain the dependency
directly from its maintainers and verify applicable terms before use.

MRI-CORE includes code derived from Meta's Segment Anything Model. A license
for CS-MRI-CORE's own scripts does not relicense that dependency or its weights.
Official SAM source: <https://github.com/facebookresearch/segment-anything>.

## BraTS-MEN-RT

Obtain data independently through <https://doi.org/10.7303/syn59059779> and
comply with its current access conditions. The data descriptor states a
CC-BY-NC license for public images, labels, and clinical metadata:
<https://doi.org/10.1038/s41597-026-06649-x>.
No medical images, masks, clinical metadata, case lists, or cache files are
distributed here. Aggregate published results contain no patient-level rows.

## Python dependencies

Install dependencies from their own distributions. Their licenses remain
with the respective upstream projects. No binary dependency is vendored here.

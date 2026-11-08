# PAD evaluation protocol (starter)

## Dataset protocol
- Use bona fide and presentation attack samples with documented consent and provenance.
- Split by subject and capture session, not by random frames, to avoid leakage.
- Keep a locked test set untouched during training and threshold selection.
- Cover print, display-photo, display-video/replay and other in-scope attack species.
- Vary camera models, distance, illumination, compression and capture conditions.
- Evaluate demographic groups responsibly and report limitations.

## Metrics
- APCER for each attack species.
- BPCER for bona fide samples.
- ACER as a summary only, alongside APCER and BPCER.
- Failure-to-acquire, retry rate and end-to-end latency.
- Confidence intervals and per-device / per-condition breakdowns.

## Reporting
Never publish accuracy from a tiny demo sample as evidence of deployment readiness. Record dataset licence, protocol, model hash, preprocessing, threshold, hardware and software versions.

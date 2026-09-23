146 targets audited under one audit-code signature; 189,710 raw models, 150,938 eligible (79.6%), 38,772 rejected.

The eligible pool clears the launcher minimum of 150,000 by 938 models (0.63%) and falls 29,062 short of the conservative builder default of 180,000. This cohort exists only because the threshold was lowered deliberately.

125 targets contribute models, losing 4.7% of their models at the median; 21 targets are fully rejected.

96.8% of rejections are a missing interface_node_degree dataset and 3.2% a missing node_features group; 1 model(s) fail a value check. The limit is incomplete feature propagation in the source graphs, not defective models.

All 150,938 eligible models belong to the complex.* family; the 25,208 synthetic random_*/sampled_* models in 19 targets are rejected outright, and those targets contribute nothing.

Both frozen-split membership gates pass: no eligible target outside the split, no split target without eligible models. Per-split eligible counts are {'train': 109147, 'val': 11325, 'test': 30466}.

ESM2 sidecars cover the audited cohort exactly: 150,938 expected, 150,938 accepted, 0 alignment failures, every embedding 1280 wide at layer 33.

All 150,938 model datasets were verified as hard links to 125 shared layouts, reducing 261 GiB of per-model copies to 0.22 GiB.

This is a coverage and provenance analysis. It reports no training, validation or test metric, and supports no claim about whether ESM2 features or either pooling variant improve DockQ prediction.

The obvious repair is targeted, not global: regenerating the missing derived features for the 21 fully rejected targets and the partially affected files would restore up to 38,772 models and lift the cohort clear of the 180,000 default. Until then, treat the thin margin above 150,000 as a standing fragility of this experiment.

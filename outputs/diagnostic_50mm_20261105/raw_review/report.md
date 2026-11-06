# 50 mm raw-cell review

- deck: `F:\tnt\tnt_files\dp0\SYS\MECH\input.k`
- d3plot: `F:\tnt\tnt_files\dp0\SYS\MECH`
- states: 0, 19, 20

## Material-column mapping (verified on the selected pure-air cells)

| deck part | name | ALE material | alpha column | mass column |
|---:|---|---:|---:|---:|
| 1 | air | 1 | 1 | 9 |
| 5 | air1 | 5 | 5 | 13 |
| 6 | air11 | 6 | 6 | 14 |
| 2,3,4,7 | TNT parts | 2,3,4,7 | 2,3,4,7 | 10,11,12,15 |

The nonzero air mass column follows the cell's own part/ALE index; alpha_product+alpha_air=1 for every selected pure-air cell.

## State-0 pure-air cells (direction-cone samples, not one ray)

| cell id | part / ALE | r (mm) | alpha 1/5/6 | m1 raw | m5 raw | m6 raw | sum raw | V (mm3) | rho_mix | m/(aV) | shell avg |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1096885 | 6 / 6 | 56.737 | 0/0/1 | 0 | 0 | 9.995e-12 | 9.995e-12 | 8.1781 | 1.225000 | 1.222130 | 1.220514 |
| 6241 | 1 / 1 | 75.956 | 1/0/0 | 1.502e-11 | 0 | 0 | 1.502e-11 | 12.3261 | 1.225000 | 1.218553 | 1.219879 |
| 101235 | 1 / 1 | 99.997 | 1/0/0 | 4.964e-11 | 0 | 0 | 4.964e-11 | 40.7639 | 1.225000 | 1.217789 | 1.217073 |
| 630755 | 5 / 5 | 125.409 | 0/1/0 | 0 | 1.074e-10 | 0 | 1.074e-10 | 88.1922 | 1.225000 | 1.217823 | 1.216007 |
| 591137 | 5 / 5 | 150.025 | 0/1/0 | 0 | 1.201e-10 | 0 | 1.201e-10 | 98.5778 | 1.225000 | 1.218262 | 1.216353 |
| 1128379 | 6 / 6 | 199.958 | 0/0/1 | 0 | 0 | 3.053e-10 | 3.053e-10 | 251.2088 | 1.225000 | 1.215469 | 1.219331 |
| 610798 | 5 / 5 | 300.013 | 0/1/0 | 0 | 6.813e-10 | 0 | 6.813e-10 | 557.9395 | 1.225000 | 1.221078 | 1.222460 |
| 1109135 | 6 / 6 | 499.972 | 0/0/1 | 0 | 0 | 1.955e-09 | 1.955e-09 | 1598.0152 | 1.225000 | 1.223536 | 1.224091 |
| 1127759 | 6 / 6 | 999.982 | 0/0/1 | 0 | 0 | 7.556e-09 | 7.556e-09 | 6169.6721 | 1.225000 | 1.224623 | 1.224773 |
| 1133216 | 6 / 6 | 1900.019 | 0/0/1 | 0 | 0 | 3.350e-08 | 3.350e-08 | 27349.7633 | 1.225000 | 1.224913 | 1.224937 |

Sampling note: each bin selects the cell with maximum dot product to the diagonal direction inside a cone; this is **not** a continuous sequence of cells crossed by one ray.  Radial ranges from different angular cells can overlap; they are not, by themselves, proof of same-ray cell thickness or air-layer unresolvability.  The raw values are a cross-check against the shell profile, not a ray-chain test.

The pure-air cells have `alpha_air=1.000`, only their own air material mass column nonzero, and `alpha_1+alpha_5+alpha_6=1`.  The mass/density ratio `m/(alpha V)` is **already below the constant mixture density** at the raw-cell level, before shell averaging; the shell-average column inherits the same radial trend.  The 6-tet and trilinear cell volumes agree to 0.06%; the volume calculation is not the source.

Across all 1002408 pure-air cells, the global mass/(rho V) ratio is 0.99998564, while the per-cell ratio ranges 0.985645..1.008199.  This is a local mass/density redistribution in the d3plot history, not a unit or column bug. The underlying mechanism (initialization, remap, or output convention) remains undetermined; it is not attributed here to ALE remap.

## State 19/20 raw cells (direction-cone samples, not one ray)

| state | cell id | part | bin (mm) | cell range (mm) | alpha | p (MPa) | u (m/s) | rho_mix | rho_air |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 19 | 531837 | 5 | 0.075-0.080 | 66.4-85.6 | 0.6119 | 74.9945 | 2717.4 | 180.2013 | 0.5026 |
| 19 | 1071441 | 6 | 0.085-0.090 | 75.9-104.2 | 0.2325 | 19.9996 | 2722.1 | 62.0728 | 1.5109 |
| 19 | 1147483 | 6 | 0.095-0.100 | 81.8-118.2 | 0.0656 | 5.6490 | 2660.9 | 17.9807 | 1.6538 |
| 19 | 531787 | 5 | 0.105-0.110 | 85.3-126.3 | 0.0300 | 1.9388 | 2822.9 | 6.5213 | 0.5049 |
| 19 | 683560 | 5 | 0.110-0.115 | 97.1-133.4 | 0.0000 | 0.1401 | 1484.7 | 1.5205 | 1.5155 |
| 19 | 643954 | 5 | 0.120-0.125 | 103.6-146.7 | 0.0000 | 0.1213 | 1394.9 | 1.3831 | 1.3785 |
| 19 | 591146 | 5 | 0.135-0.140 | 113.4-167.0 | 0.0000 | 0.1076 | 1198.3 | 1.2756 | 1.2708 |
| 19 | 531737 | 5 | 0.155-0.160 | 125.0-192.5 | 0.0000 | 0.0840 | 1441.5 | 0.9722 | 0.9684 |
| 19 | 657107 | 5 | 0.160-0.165 | 138.6-191.6 | 0.0000 | 0.1016 | 62.1 | 1.2275 | 1.2177 |
| 20 | 531837 | 5 | 0.075-0.080 | 66.4-85.6 | 0.7910 | 73.9559 | 2654.8 | 211.9513 | 0.4953 |
| 20 | 1071441 | 6 | 0.085-0.090 | 75.9-104.2 | 0.3217 | 23.7835 | 2691.9 | 78.9208 | 1.4241 |
| 20 | 1147483 | 6 | 0.095-0.100 | 81.8-118.2 | 0.1306 | 9.4583 | 2728.9 | 31.9976 | 1.5749 |
| 20 | 531787 | 5 | 0.105-0.110 | 85.3-126.3 | 0.0473 | 3.3166 | 2835.1 | 10.6671 | 0.4678 |
| 20 | 683560 | 5 | 0.110-0.115 | 97.1-133.4 | 0.0000 | 0.1508 | 1516.5 | 1.5998 | 1.5939 |
| 20 | 643954 | 5 | 0.120-0.125 | 103.6-146.7 | 0.0000 | 0.1300 | 1438.5 | 1.4524 | 1.4472 |
| 20 | 591146 | 5 | 0.135-0.140 | 113.4-167.0 | 0.0000 | 0.1139 | 1336.0 | 1.3270 | 1.3223 |
| 20 | 531737 | 5 | 0.155-0.160 | 125.0-192.5 | 0.0000 | 0.0979 | 1523.0 | 1.0084 | 1.0048 |
| 20 | 657107 | 5 | 0.160-0.165 | 138.6-191.6 | 0.0000 | 0.1020 | 89.9 | 1.2305 | 1.2208 |

## Actual training path

`build_air_observations` now uses the independent `training_location_valid` mask, then each field's own validity.  Before the split it used `material==1` plus field validity only; at state 22 that admitted one protection-band shell (r=126.40 mm, field-valid=True, location=False) if an R_b had existed.
After the split, `training_location_valid & material==1` is 0 in every selected real frame, so the one protection-band point is excluded before the geometric range is even applied.
Before/after layered counts on the same real frames:

- before split (diagnostic `material==1 & field_valid_all`): state 22 has 1 point at r=126.40 mm; it is in the protection band and would enter an observation if an R_b existed;
- after split (`material==1 & training_location_valid`): 0 points in every frame; the protection point is removed before field validity;
- legacy joint `valid & material==1`: also 0 in every frame;
- geometric range: 0 because no strict R_b; final loss counts 0/0/0.

- material1 & training_location_valid counts: [0]
- material1 & training_location & field-valid counts: [(0, 0, 0)]
- actual `AirObservationStore.size` = 0
- final field-valid counts: rho=0, u=0, p=0
- geometry block, not a material/field count, is the final training block.

## t=0 sensitivity false positive

- verified R_b under 0.005 kg/m3 tolerance = 0.21441834196957715 m
- density-only 'disturbed' cells = 816
- u=0 max |u| = 0.0 m/s
- p range = 101320.00731403015 .. 101320.00848389555 Pa

The 0.005 kg/m3 tolerance is 0.408% of 1.225, not 0.5%.

Minimal calibration direction (not applied in this round): define the ambient density baseline from the t=0/pre-shock radial profile (or use a relative baseline), and require at least one dynamic deviation (pressure or velocity) before calling a cell disturbed.  A t=0 static regression should then yield no verified R_b.  Tolerance alone is not a valid calibration.

## Diagnostic dynamic-evidence check

This table uses the 0.005 kg/m3 diagnostic tolerance only to locate a candidate.  A candidate is not called dynamically supported unless at least two inner cells have pressure or velocity deviation.  This is not an accepted boundary model and no cache is built from it.

| state | t (us) | env run | candidate (mm) | dynamic cells | dynamic evidence | air layer separated |
|---:|---:|---|---:|---:|---|---|
| 0 | 0.000 | True | 214.42 | 0 | False | False |
| 12 | 11.992 | True | 513.26 | 968 | True | False |
| 14 | 13.992 | True | 515.79 | 936 | True | False |
| 16 | 15.969 | True | 518.33 | 827 | True | False |
| 17 | 16.973 | True | 518.33 | 632 | True | False |
| 18 | 17.964 | True | 519.61 | 636 | True | False |
| 19 | 18.986 | True | 520.88 | 639 | True | False |
| 20 | 19.987 | True | 520.88 | 643 | True | False |
| 21 | 20.964 | True | 520.88 | 646 | True | False |
| 22 | 21.968 | True | 522.16 | 650 | True | False |
| 24 | 23.994 | True | 523.44 | 586 | True | False |
| 26 | 25.992 | True | 524.72 | 507 | True | False |
| 28 | 27.963 | True | 524.72 | 319 | True | False |
| 30 | 29.956 | True | 524.72 | 321 | True | False |
| 35 | 34.993 | True | 561.49 | 352 | True | False |
| 40 | 39.947 | True | 564.19 | 269 | True | False |

A true `environment run` identification is not the same as a resolved air layer.  At t=0 the environment run is found but dynamic evidence is False: 214.42 mm is **not** a verified explosion outer edge.  The `air layer separated` column remains False when contact/shock/air-layer evidence is insufficient.

## Shock context

state 20 is doubtful and has no reliable range; its finite 88.96 mm and h/cell=0.068 are **not** confirmation of an air layer.

state 19: {"time_s": 1.898620575957466e-05, "shock_status": "confirmed", "shock_has_supervision": true, "shock_radius_m": 0.10924000000000002, "contact_low_m": 0.06988000000000001, "contact_up_m": 0.11008000000000001, "contact_band_width_m": 0.0402, "contact_cell_thickness_m": 0.02156663038716058, "shock_cell_thickness_m": 0.03289667997800679, "contact_interval_half_width_m": 0.0201, "extractor_contact_position_uncertainty_m": 0.020040000000000002, "contact_mixing_cell_count": 334, "shock_h_over_cell": 0.7529449942350969, "shock_reliable_inner_m": 0.10300000000000001, "shock_reliable_outer_m": 0.10924000000000002}

state 20: {"time_s": 1.9987059204140678e-05, "shock_status": "doubtful", "shock_has_supervision": false, "shock_radius_m": 0.08896000000000001, "contact_low_m": 0.07096000000000002, "contact_up_m": 0.11284000000000002, "contact_band_width_m": 0.04188, "contact_cell_thickness_m": 0.023372981418612486, "shock_cell_thickness_m": 0.02443118310570059, "contact_interval_half_width_m": 0.02094, "extractor_contact_position_uncertainty_m": 0.020880000000000003, "contact_mixing_cell_count": 348, "shock_h_over_cell": 0.06788198234240923, "shock_reliable_inner_m": null, "shock_reliable_outer_m": null}

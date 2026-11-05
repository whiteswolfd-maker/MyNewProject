# 50 mm real d3plot contact_air diagnostic

- Deck: `F:\tnt\tnt_files\dp0\SYS\MECH\input.k`
- d3plot: `F:\tnt\tnt_files\dp0\SYS\MECH` (779 states available)
- Extracted profile: `F:\2-codex-coupled-handoff-pinn\2-codex-coupled-handoff-pinn\2-codex-coupled-handoff-pinn1\outputs\diagnostic_50mm_20261105\profile` (16 selected states)
- Real frames; no synthetic substitution; no time shift.

## Theory and clock

| quantity | value |
|---|---:|
| theoretical_t_sep_s | 1.85313329e-05 |
| theoretical_t0_s | 1.95313329e-05 |
| epsilon_s | 1e-06 |
| t_cj_s | 6.52867542e-06 |
| surface_arrival_ref_s | 7.21500722e-06 |
| theory_R_sep_m | 0.106506254 |
| theory_Rc0_m | 0.113467047 |
| theory_Rs0_m | 0.114859206 |
| contact_up_crosses_R_sep_s | 1.81849461e-05 |
| contact_up_crosses_Rc0_s | 2.0219209e-05 |
| contact_up_crosses_Rs0_s | 2.07346238e-05 |
| reconstruction_crosses_R_sep_s | 2.7056922e-05 |
| reconstruction_crosses_Rc0_s | 2.97007461e-05 |
| reconstruction_crosses_Rs0_s | 3.02387739e-05 |
| contact_low_at_t0_minus_Rc0_m | -0.0429810617 |
| contact_up_at_t0_minus_Rc0_m | -0.00180670302 |
| contact_up_at_t0_minus_Rs0_m | -0.00319886172 |
| reconstruction_at_t0_minus_Rc0_m | -0.0274513284 |
| reconstruction_at_t0_minus_Rs0_m | -0.0288434871 |

Frame before t0: state 19 at 18.986206 us; frame after t0: state 20 at 19.987059 us.

## Pressure comparison

| quantity | value |
|---|---:|
| p_c_t0_pa | 55397732.71324027 |
| p_a_sep_pa | 71225390.21469861 |
| p_air_ic_config_pa | 101320.0 |
| p_air_ic_data_outer_median_pa | 101320.00765363831 |
| p_c_t0_minus_p_a_sep_pa | -15827657.50145834 |
| p_c_t0_vs_p_a_sep_relative | 0.2222193160858531 |
| p_c_t0_minus_air_ic_config_pa | 55296412.71324027 |
| p_c_t0_vs_air_ic_config_relative | 545.760093893015 |
| p_c_t0_over_air_ic_config | 546.760093893015 |
| p_c_t0_minus_air_ic_data_pa | 55296412.705586635 |
| p_c_t0_vs_air_ic_data_relative | 545.7600525911625 |
| forced_equal | False |

## Per-frame table (strict current configuration)

| # | state | t (us) | contact low/up (mm) | R_s (mm) | R_s - R_cup (mm) | R_b cand/verified | R_b failure | sep | native source/value (mm) | sup rho/u/p | field-valid rho/u/p | air withdrawn | first valid air (mm) |
|---|---:|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 0 | 0 | 0.000 | 49.84/50.08 | — | — | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | air_native_resolution_fallback_m/6.418 | 0/0/0 | 2084/2084/2084 | 2084 | 50.08 |
| 1 | 12 | 11.992 | 49.96/76.00 | 72.64 | -3.36 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/15.260 | 0/0/0 | 1823/1823/1823 | 1898 | 81.40 |
| 2 | 14 | 13.992 | 60.16/83.20 | 126.52 | +43.32 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/43.553 | 0/0/0 | 1769/1769/1769 | 1841 | 87.88 |
| 3 | 16 | 15.969 | 62.80/93.64 | 133.24 | +39.60 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/46.255 | 0/0/0 | 1647/1647/1647 | 1772 | 102.52 |
| 4 | 17 | 16.973 | 64.36/99.76 | 113.92 | +14.16 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/35.815 | 0/0/0 | 1448/1448/1448 | 1732 | 126.40 |
| 5 | 18 | 17.964 | 68.56/105.52 | 154.74 | +49.22 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/49.624 | 0/0/0 | 1448/1448/1448 | 1695 | 126.40 |
| 6 | 19 | 18.986 | 69.88/110.08 | 109.24 | -0.84 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/32.897 | 0/0/0 | 1448/1448/1448 | 1657 | 126.40 |
| 7 | 20 | 19.987 | 70.96/112.84 | 88.96 | -23.88 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/24.431 | 0/0/0 | 1448/1448/1448 | 1620 | 126.40 |
| 8 | 21 | 20.964 | 71.92/115.48 | 118.24 | +2.76 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/38.509 | 0/0/0 | 1448/1448/1448 | 1593 | 126.40 |
| 9 | 22 | 21.968 | 73.00/118.36 | 126.52 | +8.16 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/43.553 | 0/0/0 | 1448/1448/1448 | 1570 | 126.40 |
| 10 | 24 | 23.994 | 75.40/125.20 | — | — | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | air_native_resolution_fallback_m/37.985 | 0/0/0 | 1376/1376/1376 | 1523 | 135.04 |
| 11 | 26 | 25.992 | 77.80/133.48 | 148.00 | +14.52 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/49.872 | 0/0/0 | 1292/1292/1292 | 1475 | 145.12 |
| 12 | 28 | 27.963 | 80.92/144.88 | 158.65 | +13.77 | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | shock_cell_thickness_m/49.479 | 0/0/0 | 1099/1099/1099 | 1412 | 193.11 |
| 13 | 30 | 29.956 | 88.12/153.00 | — | — | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | air_native_resolution_fallback_m/50.066 | 0/0/0 | 1099/1099/1099 | 1312 | 193.11 |
| 14 | 35 | 34.993 | 95.32/169.46 | — | — | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | air_native_resolution_fallback_m/49.487 | 0/0/0 | 1099/1099/1099 | 1202 | 193.11 |
| 15 | 40 | 39.947 | 101.80/181.01 | — | — | —/False | no finite ambient run under configured density/velocity tolerances | unresolvable | air_native_resolution_fallback_m/49.192 | 0/0/0 | 986/986/986 | 1160 | 257.73 |

## Ambient verification

- t=0 ambient field-valid shells: 2084
- radial range: 0.05008000000000001 .. 1.99858008 m
- rho: 1.2157976451885464 .. 1.224943036545846 kg/m3 (relative span 0.00746563)
- p: 101320.00731403015 .. 101320.00848389555 Pa
- max |u|: 0.0 m/s
- strict per-frame ambient counts: p/rho/u/all = 2084/0/2084/0 at t=0

## Conclusion

The blocking item is **not a d3plot read failure**: all 16 selected real states read, and the air fields are finite where the extractor marks them valid. The first-order blockers are data-side: (a) the contact mixing band is about 40 mm wide and its upper edge can equal or exceed the detected shock (t0: 111.66 mm vs 109.24 mm), so there is no resolved shocked-air layer; (b) the first pure-air shells outside the contact are contaminated/invalid (first valid air 126.4 mm), and no selected frame has a training-valid air point; (c) the strict ambient tolerances are incompatible with the measured radial air-density profile (1.216-1.225 kg/m3), so no ambient run or R_b is verified. The old shock supervision leaves only 4/16 frames 'confirmed', and the frame immediately after t0 has all air labels withdrawn. The theory clock is not proven incompatible: R_cup reaches R_sep at 18.18 us vs t_sep=18.53 us, and reaches Rc0/Rs0 at 20.22/20.73 us vs t0=19.53 us, but this agreement is with the broad upper bracket edge, not with the reconstruction centre. No version-4 cache was written.

A diagnostic tolerance relaxation is included in `diagnostic.json`; it is not an accepted configuration and no cache was built from it.

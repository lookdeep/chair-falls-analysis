# Gemini Model Characterization Study

## Study Design
- One-off standalone characterization study using Gemini outputs and the v2 consensus GT.
- Primary head-to-head view uses the 3-way successful overlap across all three Gemini models.
- Full-set per-model coverage is reported separately to show availability and failure differences.
- Visible reviewed clips are the headline denominator; offscreen, no-signal, and bad/no-video remain diagnostic strata.

## Inputs
- `gemini-2.5-flash`: `summary_20260312T182543Z.csv` and `results_20260312T182543Z.jsonl` (70 successful rows, 11 errors)
- `gemini-3.1-pro-preview`: `summary_20260313T002026Z.csv` and `results_20260313T002026Z.jsonl` (72 successful rows, 9 errors)
- `gemini-2.5-pro`: `summary_20260313T083405Z.csv` and `results_20260313T083405Z.jsonl` (67 successful rows, 14 errors)

## Primary 3-Way Overlap Ranking
| model                  | visible_fall_sensitivity | visible_nonfall_specificity | location_accuracy_detected_visible_falls | last_furniture_accuracy_detected_visible_falls | fall_tag_exact_match_rate_detected_visible_falls | fall_tag_jaccard_mean_detected_visible_falls | fall_time_mae_seconds_detected_visible_falls |
| ---------------------- | ------------------------ | --------------------------- | ---------------------------------------- | ---------------------------------------------- | ------------------------------------------------ | -------------------------------------------- | -------------------------------------------- |
| gemini-2.5-flash       | 0.842                    | 0.75                        | 0.531                                    | 0.917                                          | 0.031                                            | 0.12                                         | 1199.387                                     |
| gemini-2.5-pro         | 0.684                    | 1                           | 0.654                                    | 0.9                                            | 0.038                                            | 0.256                                        | 520.615                                      |
| gemini-3.1-pro-preview | 0.447                    | 1                           | 0.588                                    | 0.857                                          | 0.118                                            | 0.218                                        | 105.529                                      |

## Coverage
| model                  | successful_summary_rows | error_rows | three_way_overlap_event_keys |
| ---------------------- | ----------------------- | ---------- | ---------------------------- |
| gemini-2.5-flash       | 70                      | 11         | 59                           |
| gemini-3.1-pro-preview | 72                      | 9          | 59                           |
| gemini-2.5-pro         | 67                      | 14         | 59                           |

## Three-Way Overlap Detection Counts
| model                  | tp | fn | fp | tn | sensitivity | specificity |
| ---------------------- | -- | -- | -- | -- | ----------- | ----------- |
| gemini-2.5-flash       | 32 | 6  | 1  | 3  | 0.842       | 0.75        |
| gemini-2.5-pro         | 26 | 12 | 0  | 4  | 0.684       | 1           |
| gemini-3.1-pro-preview | 17 | 21 | 0  | 4  | 0.447       | 1           |

## Leading Location Confusions
| model            | predicted_prefall_location | gt_prefall_location | count |
| ---------------- | -------------------------- | ------------------- | ----- |
| gemini-2.5-flash | bed                        | bed                 | 8     |
| gemini-2.5-flash | bed                        | room                | 5     |
| gemini-2.5-flash | chair                      | chair               | 5     |
| gemini-2.5-flash | room                       | bed                 | 4     |
| gemini-2.5-flash | room                       | room                | 4     |
| gemini-2.5-flash | chair                      | room                | 3     |
| gemini-2.5-flash | missing                    | bed                 | 2     |
| gemini-2.5-flash | room                       | chair               | 1     |
| gemini-2.5-pro   | bed                        | bed                 | 9     |
| gemini-2.5-pro   | room                       | room                | 5     |
| gemini-2.5-pro   | bed                        | room                | 4     |
| gemini-2.5-pro   | room                       | bed                 | 4     |

## Leading False-Negative GT Tags
| model            | gt_tag        | count |
| ---------------- | ------------- | ----- |
| gemini-2.5-flash | slip          | 3     |
| gemini-2.5-flash | support_bed   | 3     |
| gemini-2.5-flash | back          | 2     |
| gemini-2.5-flash | slow          | 2     |
| gemini-2.5-flash | collapse      | 1     |
| gemini-2.5-flash | support_chair | 1     |
| gemini-2.5-flash | support_door  | 1     |
| gemini-2.5-flash | tumble        | 1     |
| gemini-2.5-pro   | slip          | 6     |
| gemini-2.5-pro   | back          | 4     |
| gemini-2.5-pro   | slow          | 4     |
| gemini-2.5-pro   | roll          | 2     |

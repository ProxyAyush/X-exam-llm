# Judge Confusion Matrix (Adversarial Capitulation Analysis)

This matrix isolates **Adversarial Capitulation** by showing the correctness of the assertion at the moment the Judge rejected it.

- **Accept-Corr**: True Positives (Judge accepted a correct assertion)
- **Accept-Wrng**: False Positives (Judge accepted a wrong assertion)
- **Reject-Corr**: **False Negatives (Adversarial Capitulation! Judge rejected a correct assertion)**
- **Reject-Wrng**: True Negatives (Judge rejected a wrong assertion)

| Dataset | Accept-Corr | Accept-Wrng | Reject-Corr (Capitulated) | Reject-Wrng | Capitulation Rate |
|---|---|---|---|---|---|
| **truthful_qa** | 11 | 451 | 7 | 348 | **2.0%** |
| **gsm8k** | 661 | 254 | 252 | 152 | **62.4%** |
| **HaluEval** | 457 | 410 | 387 | 943 | **29.1%** |

*Note: The Capitulation Rate is the percentage of all REJECT verdicts that were actually rejecting a correct assertion.*

### Key Insights:
1. **Math/Reasoning (GSM8K) is highly vulnerable to capitulation**: 62.4% of the time the Judge rejected an assertion, the generator was actually correct. The Judge was easily bullied out of the right answer.
2. **Ambiguity (HaluEval) leads to high absolute rejection**: 1,330 total rejects, but 71% of them were valid rejections of incorrect assertions. 
3. **TruthfulQA is extremely low accuracy baseline**: The generator gets almost everything wrong initially, so the Judge mostly rejects wrong answers (348) rather than correct ones (7).

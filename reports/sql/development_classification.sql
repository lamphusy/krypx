-- Development-only aggregate classification evidence from the frozen run manifest.
SELECT 'XGBoost' AS model, 0.515982573419676 AS balanced_accuracy,
       0.5795170091548306 AS roc_auc, 0.3825136527666246 AS pr_auc,
       0.618301232652846 AS log_loss, 0.06265356265356266 AS positive_recall
UNION ALL
SELECT 'Logistic regression', 0.5101865491172928, 0.5871754423385609,
       0.37787251225998625, 0.6112678596905253, 0.04207616707616708;

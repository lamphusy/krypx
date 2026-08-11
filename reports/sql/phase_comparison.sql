-- Reviewed XGBoost development OOF and one-time final holdout comparison.
-- Development source: artifacts/runs/20260811T101311613823Z_btc_usdt_1h_8f2800e/development_strategy_metrics.json
-- Holdout source: artifacts/evaluations/20260811T101311613823Z_btc_usdt_1h_8f2800e/evaluation_manifest.json
SELECT 'Development OOF' AS period, -0.13425219783160158 AS total_return,
       -0.5412917288544219 AS sharpe_ratio, -0.2093674013789274 AS maximum_drawdown,
       0.05641856475745735 AS market_exposure, 148 AS trades,
       0.8604914981534763 AS profit_factor
UNION ALL
SELECT 'Final holdout', -0.042313656484754714, -0.3958849744422429,
       -0.17409918412573955, 0.028941355674028942, 38,
       0.8379939716510009;

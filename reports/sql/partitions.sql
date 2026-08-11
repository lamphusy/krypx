-- Frozen chronological split geometry from the Phase 1 evaluation manifest.
SELECT 'Development' AS partition, '2023-08-13T19:00:00Z' AS start,
       '2026-01-04T06:00:00Z' AS end, 20988 AS rows
UNION ALL
SELECT 'Boundary purge', '2026-01-04T07:00:00Z', '2026-01-04T11:00:00Z', 5
UNION ALL
SELECT 'Final holdout', '2026-01-04T12:00:00Z', '2026-08-11T04:00:00Z', 5249;

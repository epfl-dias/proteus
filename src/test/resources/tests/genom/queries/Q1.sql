SELECT count(*)
FROM genom_broad as L, genom_broad2 as R
WHERE L.chrom = R.chrom2
  AND L.`start` < R.stop2
  AND R.`start2` < L.stop
  AND L.filename = 'ENCFF271CVJ.bed' ;
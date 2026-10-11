# Journal : estimation de la position (SLAM 2D) pour Scanner

Objectif : retirer la position exacte du simulateur et la remplacer par une position **estimée**, sans aucun apprentissage, puis mesurer ce que ça change.
Règle : aucun chiffre n'est écrit ici sans le fichier de résultats et le script qui l'ont produit ; sinon « non vérifié ».

## Vocabulaire

- **Position exacte (oracle)** : la position vraie donnée par le simulateur. Un vrai drone ne l'a pas.
- **Odométrie / dead-reckoning** : on additionne, pas après pas, la vitesse et la rotation mesurées. Les petites erreurs s'accumulent.
- **Dérive** : l'écart entre la position estimée et la vraie, qui grandit avec le temps et la distance.
- **Scan-matching** (étape 2) : on recale la position en superposant le scan LiDAR du moment sur la carte déjà construite.
- **Fermeture de boucle** (étape 3) : on reconnaît qu'on revient dans un endroit déjà vu, et on corrige tout le trajet pour que la carte se referme.
- **Graphe de poses** (étape 3) : les positions successives sont des nœuds, les mesures entre elles sont des liens ; on cherche les positions qui respectent le mieux tous les liens.

## Montage (étape 1)

- `pose_source="oracle"` (défaut) : comportement historique, inchangé.
- `pose_source="odom"` : la position et l'orientation utilisées pour **construire la carte de planification** et **planifier** viennent de l'odométrie bruitée.
  Le PID de vol continue d'utiliser l'état vrai (comme une centrale inertielle).
- Deux cartes tournent en parallèle : `grid` (placée avec la vraie position, sert uniquement à mesurer : couverture sur la vraie surface, succès, carte de référence)
  et `est_grid` (placée avec la position estimée, la seule que voient l'expert, le BFS et les frontières).
- Bruit (`src/pose_estimation.py`, valeurs choisies à la main, **non calibrées sur un capteur réel**) :

| niveau | erreur d'échelle (σ) | biais de vitesse (σ, m/s) | bruit de vitesse (m/s) | biais gyro (σ, °/s) | bruit gyro (°/s) |
|---|---|---|---|---|---|
| low | 0,5 % | 0,01 | 0,02 | 0,05 | 0,3 |
| medium | 1,5 % | 0,03 | 0,05 | 0,2 | 0,6 |
| high | 3 % | 0,06 | 0,10 | 0,5 | 1,0 |

  L'erreur d'échelle et les biais sont tirés une fois par vol (constants), le bruit est tiré à chaque pas.
- Protocole de mesure : `src/pose_eval.py`, expert scripté seul (BFS, marge 0,45 m, réflexe 0,4), niveaux 1 à 4, graines 9000-9019 (20 maisons par niveau),
  arrêt à 90 % de couverture vraie, 5 000 pas maximum, cap fixe (sans lacet qui suit).
- Référence de carte : les mêmes scans placés avec la position exacte (`grid`). F1 sur les cellules occupées, tolérance d'une cellule.

## GATE 1 : dérive seule, sans correction

Mesure : `python src/pose_eval.py --levels 1 2 3 4 --noise none low medium high --seeds 9000 9020`, résultats bruts dans `docs/results/gate1_dead_reckoning.csv`
(320 vols : 20 maisons par niveau et par bruit, mêmes maisons et mêmes graines partout). « none » = position exacte, mesurée avec le même banc.

| niveau | bruit | succès | collisions | timeouts | couverture vraie | RMSE pos. (m) | max (m) | RMSE lacet (°) | dérive finale (% distance) | F1 carte | ms/pas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exacte | 100 % | 0 % | 0 % | 90,3 % | - | - | - | - | - | 3,4 |
| 1 | low | 100 % | 0 % | 0 % | 90,4 % | 0,05 | 0,08 | 0,2 | 0,7 | 1,00 | 5,6 |
| 1 | medium | 100 % | 0 % | 0 % | 90,3 % | 0,13 | 0,22 | 0,6 | 1,9 | 0,93 | 5,6 |
| 1 | high | 95 % | 5 % | 0 % | 90,3 % | 0,24 | 0,42 | 1,6 | 3,6 | 0,81 | 5,8 |
| 2 | exacte | 100 % | 0 % | 0 % | 90,2 % | - | - | - | - | - | 4,3 |
| 2 | low | 100 % | 0 % | 0 % | 90,1 % | 0,13 | 0,22 | 0,3 | 0,9 | 0,95 | 5,8 |
| 2 | medium | 85 % | 0 % | 15 % | 85,7 % | 0,68 | 1,17 | 2,4 | 10,6 | 0,71 | 5,6 |
| 2 | high | 50 % | 0 % | 50 % | 83,2 % | 2,95 | 5,06 | 11,6 | 25,9 | 0,54 | 6,1 |
| 3 | exacte | 95 % | 0 % | 5 % | 87,0 % | - | - | - | - | - | 5,3 |
| 3 | low | 100 % | 0 % | 0 % | 90,2 % | 0,16 | 0,29 | 0,5 | 0,8 | 0,93 | 8,5 |
| 3 | medium | 70 % | 0 % | 30 % | 86,9 % | 1,22 | 2,13 | 4,3 | 8,3 | 0,64 | 9,5 |
| 3 | high | 55 % | 5 % | 40 % | 79,0 % | 2,87 | 4,93 | 13,9 | 27,4 | 0,50 | 8,6 |
| 4 | exacte | 85 % | 5 % | 10 % | 87,9 % | - | - | - | - | - | 6,0 |
| 4 | low | 95 % | 0 % | 5 % | 89,8 % | 0,28 | 0,49 | 0,8 | 1,2 | 0,85 | 8,1 |
| 4 | medium | 60 % | 5 % | 35 % | 84,0 % | 1,32 | 2,27 | 5,6 | 6,5 | 0,56 | 9,9 |
| 4 | high | 45 % | 5 % | 50 % | 72,2 % | 2,93 | 5,15 | 14,4 | 22,3 | 0,47 | 7,6 |

Lecture (20 maisons par case : environ ±10 points, donc seuls les grands écarts comptent) :
- La position exacte retrouve les chiffres déjà publiés dans le README (100 / 100 / 95 / 85 %) : le banc est cohérent.
- Bruit faible : pas de différence démontrable avec la position exacte (les 100 % et 95 % de « low » contre 95 % et 85 % de « exacte » aux niveaux 3 et 4 sont dans le bruit d'échantillonnage).
- Bruit moyen : la perte devient nette dès le niveau 2 (85 / 70 / 60 % contre 100 / 95 / 85 %), surtout sous forme de timeouts. Bruit fort : 50 / 55 / 45 %.
- L'erreur de position et la qualité de carte se dégradent avec la durée du vol : le F1 de la carte tombe de 0,93-1,00 (niveau 1, low-medium) à 0,47-0,64 (niveaux 3-4, medium-high).
- Le calcul de la carte estimée coûte environ 2 à 4 ms de plus par pas (temps mesuré avec 8 vols en parallèle, valeur indicative).
- Non vérifié : si la dérive CAUSE l'échec ou si les vols qui errent finissent simplement par dériver davantage. Les vols en timeout durent 5 000 pas contre environ 1 000
  pour les réussites (erreur finale moyenne 7,7 m contre 0,9 m pour medium et high, niveaux 2 à 4), donc les deux effets sont mélangés ; il faudrait comparer l'erreur à un même instant du vol.

Conclusion de la GATE 1 : la dérive dégrade clairement l'exploration dès que le bruit est moyen ; le SLAM a donc quelque chose à corriger. Les valeurs de bruit sont arbitraires (non calibrées).

## GATE 2 : scan-matching (corrélation de scan sur la carte déjà construite)

Montage : `src/pose_estimation.py` (`ScanMatcher`, `ScanMatchedPose`). À chaque scan, après la prédiction par odométrie, on essaie des centaines de petites corrections (x, y, cap) autour de la position prédite
(grossier ±20 cm / ±4° par pas de 5 cm / 1°, puis fin ±5 cm / ±1° par pas de 1,25 cm / 0,25°). Score d'une correction : somme, sur les points du scan, d'un champ de vraisemblance (1 sur un mur de la carte,
décroissant jusqu'à 0 sur 4 cases, interpolation bilinéaire), moins une préférence pour les petites corrections. Pas de correction si le gain est trop faible ou s'il y a moins de 25 points.
Première version (carte binaire, bande « mur » de 45 cm) : presque aucune correction possible, car toutes les corrections de moins de 15 cm donnaient le même score ; remplacée par le champ de vraisemblance lisse.

Mesure : `python src/pose_eval.py --noise low medium high --pose-source slam`, résultats bruts `docs/results/gate2_scan_matching.csv` (240 vols, mêmes maisons et graines que la GATE 1).
Succès (20 maisons par case, ±10 points) ; entre parenthèses : erreur de position RMSE en m / erreur de cap RMSE en degrés / F1 de la carte.

| niveau | exacte | odométrie seule, bruit fort | scan-matching, faible | scan-matching, moyen | scan-matching, fort |
|---|---|---|---|---|---|
| 1 | 100 % | 95 % (0,24 / 1,6 / 0,81) | 100 % (0,07 / 0,2 / 1,00) | 100 % (0,08 / 0,3 / 1,00) | 100 % (0,08 / 0,4 / 0,99) |
| 2 | 100 % | 50 % (2,95 / 11,6 / 0,54) | 100 % (0,08 / 0,2 / 1,00) | 100 % (0,11 / 0,3 / 0,97) | 100 % (0,12 / 0,5 / 0,96) |
| 3 | 95 % | 55 % (2,87 / 13,9 / 0,50) | 100 % (0,11 / 0,2 / 0,99) | 95 % (0,15 / 0,4 / 0,96) | 90 % (0,18 / 1,2 / 0,91) |
| 4 | 85 % | 45 % (2,93 / 14,4 / 0,47) | 85 % (0,10 / 0,2 / 0,99) | 95 % (0,16 / 0,6 / 0,93) | 80 % (0,29 / 2,2 / 0,86) |

- Le scan-matching ramène le succès au niveau de la position exacte (aux ±10 points près) et l'erreur de position à 0,07-0,29 m, quel que soit le niveau de bruit.
- Les échecs restants (timeouts de 5 à 20 % aux niveaux 3-4) sont du même ordre que ceux de la position exacte (5 % et 10 %) : non attribuables à la dérive.
- Seuls 3 % des scans reçoivent une correction (les autres sont déjà cohérents avec la carte) ; l'appariement coûte en moyenne 5,4 ms par scan (8 vols en parallèle, Python non optimisé).
- Réserves : LiDAR simulé SANS bruit de mesure (plus facile que le réel) ; position de départ connue ; bruits d'odométrie choisis à la main.

## GATE 3 : fermeture de boucle, graphe de poses, reconstruction de la carte

Montage : `src/slam_backend.py` (`PoseGraphSlam`).
- Keyframe (position + scan brut) tous les 0,5 m ou 10° ; lien « odométrie » entre keyframes consécutifs (mesure relative issue du scan-matching).
- Candidats de boucle : anciens keyframes (au moins 20 keyframes plus tôt) à moins de 3 m ; vérification : le scan courant est apparié à une petite carte construite avec les keyframes j-2 à j+2 (fenêtre ±60 cm, ±10°) ;
  accepté si au moins 65 % des points tombent contre un mur de cette carte et si la correction reste sous 0,8 m ; repos de 8 keyframes entre deux boucles.
- Graphe de poses optimisé par Gauss-Newton (numpy, 8 itérations, premier nœud fixé) ; une boucle encore incohérente après optimisation (plus de 0,5 m ou 8°) est retirée.
- Reconstruction : si un keyframe bouge de plus de 2 cm, la carte estimée est vidée et refaite avec les scans des keyframes à leur position corrigée.
- Vraie position utilisée UNIQUEMENT pour les statistiques (fausses boucles, erreur avant / après).

Mesure : `docs/results/gate3_loop_closure.csv` (bruit faible, moyen, fort ; 240 vols) et `docs/results/extreme_*.csv` (bruit extrême : échelle 6 %, biais de vitesse 0,15 m/s, bruit de vitesse 0,2 m/s, biais gyro 1,5 °/s, bruit gyro 2 °/s ; niveaux 2 à 4, 60 vols par méthode).
Script : `docs/results/run_gate3.sh`.

Bruit faible / moyen / fort, succès (RMSE m / cap ° / F1), 20 maisons par case :

| niveau | scan-matching seul, fort | + fermeture de boucle, faible | + boucle, moyen | + boucle, fort |
|---|---|---|---|---|
| 1 | 100 % (0,08 / 0,4 / 0,99) | 100 % (0,07 / 0,2 / 1,00) | 100 % (0,08 / 0,3 / 0,99) | 100 % (0,08 / 0,4 / 0,99) |
| 2 | 100 % (0,12 / 0,5 / 0,96) | 100 % (0,08 / 0,2 / 1,00) | 100 % (0,11 / 0,3 / 0,97) | 100 % (0,11 / 0,4 / 0,96) |
| 3 | 90 % (0,18 / 1,2 / 0,91) | 100 % (0,10 / 0,2 / 0,99) | 100 % (0,13 / 0,4 / 0,96) | 85 % (0,14 / 0,7 / 0,94) |
| 4 | 80 % (0,29 / 2,2 / 0,86) | 95 % (0,10 / 0,2 / 0,99) | 90 % (0,15 / 0,6 / 0,94) | 90 % (0,25 / 1,4 / 0,89) |

Bruit extrême (niveaux 2, 3, 4) :

| méthode | succès | RMSE position (m) | RMSE cap (°) | F1 carte |
|---|---|---|---|---|
| odométrie seule | 35 / 20 / 15 % | 7,3 / 8,0 / 7,6 | 41 / 50 / 55 | 0,43 / 0,36 / 0,37 |
| scan-matching | 100 / 80 / 80 % | 0,13 / 0,48 / 0,57 | 1,1 / 5,3 / 6,3 | 0,94 / 0,86 / 0,80 |
| scan-matching + fermeture de boucle | 100 / 90 / 80 % | 0,13 / 0,20 / 0,42 | 1,0 / 1,3 / 3,0 | 0,94 / 0,92 / 0,82 |

Boucles :
- Bruit faible à fort : 499 boucles acceptées sur 240 vols (98 vols avec au moins une boucle), 0 fausse (contrôle contre la vraie position : écart de plus de 0,5 m ou 10°), 0 rejetée par la vérification.
  Erreur de position aux fermetures : 0,134 m avant, 0,128 m après (le scan-matching laisse peu à corriger).
- Bruit extrême : 189 boucles sur 60 vols (37 avec boucle), 1 fausse, 0 rejetée ; erreur aux fermetures 0,253 m avant, 0,219 m après.
- La fermeture de boucle n'apporte presque rien quand le scan-matching suffit (bruit faible à fort) ; elle réduit nettement la dérive résiduelle au bruit extrême (RMSE 0,48 -> 0,20 m au niveau 3).
  Avec 20 maisons par case, les différences de succès entre « avec » et « sans » boucle (90 % contre 80 %) sont dans le bruit d'échantillonnage ; les différences d'erreur de position sont plus nettes.
Coût : appariement 6,3 ms par scan en moyenne ; back-end 7,6 ms par keyframe (31 ms au bruit extrême, reconstructions de carte comprises). Temps de calcul total d'un pas simulé : 9 à 11 ms avec scan-matching,
11 à 22 ms avec boucle (contre 3 à 6 ms pour la simulation seule), mesuré avec 8 vols en parallèle sur un PC de bureau, Python non optimisé : sous le budget de 20 ms d'un pas à 50 Hz dans ces mesures ; non vérifié sur une machine embarquée.

Démonstrations : `python src/make_demo.py --compare --odom-noise high --level 3 --seeds 9002 --stride 10 --gif` (voir la section « Comparaison sur le même vol » ci-dessous). Les premiers gifs à bruit extrême (niveau 3, seed 9008) ont été retirés : trop chargés, et l'odométrie seule et le SLAM n'y avaient ni la même trajectoire ni la même mesure, donc pas comparables à l'oeil.

Non vérifié / limites : LiDAR sans bruit ; position de départ connue ; bruits choisis à la main (non calibrés) ; 2D à altitude fixe ; 20 maisons par case (±10 points) ; la détection de frontières utilise encore le masque réel du bâtiment (effet négligeable mesuré plus tôt) ;
une seule fausse boucle observée (1 sur 688) : trop peu pour estimer un taux ; maisons aux pièces très semblables non testées spécifiquement.

## Travaux précédents : RL pur, imitation (résumé des mesures)

Le journal détaillé de cette période (versions v1 à v2.x, hypothèses, essais abandonnés) est dans `experiments.md` à la racine, hors dépôt. Ici : les tableaux qui servent au README.
Évaluations sur graines fixes 9000+, 30 épisodes sauf indication ; source : `src/eval_logs/journal.csv` et le journal détaillé.

### Versions clés (génération 1 : le réseau commande les couples moteurs ; bâtiments en chaîne de 2 à 4 pièces)

| version | changement | succès | collision | retourné | couverture moyenne |
|---|---|---|---|---|---|
| v1 | baseline | 0 % | 100 % | 0 % | ~35-40 %* |
| v2 | rapports de couple corrigés, entrées d'orientation | 0 % | 90 % | 10 % | 58,5 %* |
| v7 | pénalité de vitesse angulaire renforcée (0,08) | 3 % | 93 % | 3 % | 65,9 % |
| v11 | amortissement d'articulation 0,05 + pénalité de vitesse angulaire 0,02 | 43 % | 47 % | 10 % | 81,4 % |
| v12 | v11 sur 1 M de pas | 47 % | 50 % | 3 % | 82,9 % |
| v16 | plafond dur de vitesse horizontale (6 m/s) | 60 % | 37 % | 3 % | 80,7 % |
| v18 | pénalité de collision x10 | 67 % | 20 % | 13 % | 81,9 % |

\* avant la correction d'un artefact de mesure de la couverture : non comparable aux lignes suivantes.

### Génération 2 (consigne de vitesse + cascade PID, chaînes de 3 pièces, 350k pas)

| essai | seed | succès | couverture moyenne |
|---|---|---|---|
| `ctrl6` (première cascade réglée) | 42 | 83 % | 86 % |
| `ctrl6` (identique) | 43 | 7 % | 70 % |
| `plant` (amortissement 0, gains réglés) | 42 | 97 % | 89 % |
| `plant` (identique) | 43 | 87 % | 87 % |
| `caps1` (+ CAPS) | 42 et 43 | 100 % et 100 % | 90,4 % |

Cause racine du long blocage de pilotage : un amortissement d'articulation hérité (0,05) combiné à un gain de vitesse angulaire trop faible ; temps de montée de la boucle d'attitude de 414 ms à 76 ms après réglage.
CAPS : commandes 4 fois plus lisses (95e centile du changement de commande 1,2 m/s contre 4,8 m/s) ; sur bâtiments non vus de 2 à 4 pièces : 97 % / 97 % contre 87 % / 77 % sans CAPS.
`caps1`, entraîné sur chaînes, essayé sans réentraînement sur les maisons : 17 / 0 / 3 / 0 % aux niveaux 1 à 4.

### RL pur sur les maisons de niveau 1 (seed 42, 350k pas sauf indication, 30 maisons)

| essai | succès | collision | timeout | couverture |
|---|---|---|---|---|
| référence : carte CNN + compas de frontière à vol d'oiseau | 27 % | 10 % | 63 % | 77,2 % |
| sans compas (carte seule) | 23 % | 30 % | 47 % | 77,1 % |
| compas BFS (distance de chemin, marge 0,30 m) | 27 % | 47 % | 27 % | 77,0 % |
| BFS + récompense de progression (potentiel x 50) | 17 % | 10 % | 73 % | 73,9 % |
| pénalité de stagnation (après 100 pas sans case nouvelle, -0,1 par pas, arrêt à 300) | 13 % | 17 % | 70 % | 73,6 % |
| carte à cases de 30 cm, 500k pas (référence à 500k : 23 / 27 / 50 %, 78,3 %) | 27 % | 37 % | 37 % | 72,5 % |
| gamma 0,999, lambda 0,98 (7 environnements au lieu de 8, non comparable) | 10 % | 27 % | 63 % | 67,0 % |

Un seul seed et 30 maisons : environ ±8 points, donc ces taux de succès sont indiscernables. Mesures de diagnostic : la droite vers la frontière la plus proche coupe un mur dans 80-99 % des pas ;
les drones en timeout sont presque immobiles (0,3 à 0,5 m sur les 300 derniers pas, 95 % de pas sans case nouvelle) ; avec du bruit sur l'action à l'évaluation les timeouts passent de 45 % à 10 % et les collisions de 20 % à 50 %.

### Expert scripté et imitation (maisons, graines 9000+)

| niveau | expert (20 maisons) | clone par imitation (30 maisons) | collisions du clone | timeouts du clone |
|---|---|---|---|---|
| 1 | 100 % | 100 % | 0 % | 0 % |
| 2 | 100 % | 93 % | 0 % | 7 % |
| 3 | 95 % | 60 % | 13 % | 27 % |
| 4 | 85 % | 53 % | 23 % | 23 % |

Expert : suit la direction du premier pas du chemin BFS, ralentit près des murs, s'écarte des murs à moins de 0,8 m ; il décide avec les mêmes entrées que le réseau. Le clone (80 157 échantillons, 210 maisons) copie ce script et reçoit sa direction BFS en entrée :
ce n'est pas un résultat de RL de bout en bout. Sans le plan réel du bâtiment dans la détection de frontières, l'expert fait 95 % (niveau 3) et 90 % (niveau 4) : effet négligeable.


## Comparaison sur le même vol (vidéos `demo/demo_compare_*`)

Problème des premières démos : maison, bruit et trajectoire différaient entre « odométrie seule » et « SLAM » (le drone suit sa propre carte, donc deux estimateurs donnent deux trajets). Correction : un SEUL vol, piloté par le SLAM
(scan-matching + fermeture de boucle), et, en parallèle, une pose et une carte « odométrie seule » calculées sur exactement les mêmes mesures bruitées et les mêmes scans (`shadow_odom=True` dans `ExplorerEnv`, option `--compare` de `make_demo.py`).
Le « fantôme » ne sert à aucune décision. Les vols SLAM sont identiques à la mesure de la GATE 3 (12 vols vérifiés sur 12).

Vols testés (bruit fort, lacet réaliste, exploration poursuivie jusqu'à plus de frontière ; ce n'est donc pas le même protocole que les tableaux des GATES, qui s'arrêtent à 90 %) :

| niveau | seed | couverture vraie | erreur finale SLAM (max) | erreur finale odométrie seule, même vol (max) |
|---|---|---|---|---|
| 3 | 9000 | 98,1 % | 0,32 m (0,45) | 1,66 m (1,78) |
| 3 | 9001 | 87,1 % | 0,07 m (0,18) | 1,25 m (1,25) |
| 3 | **9002** | 97,6 % | 0,18 m (0,26) | 2,81 m (3,65) |
| 3 | 9003 | 91,3 % | 0,15 m (0,38) | 0,97 m (1,48) |
| 3 | 9004 | 98,4 % | 0,11 m (0,27) | 1,95 m (2,75) |
| 3 | 9005 | 98,4 % | 0,15 m (0,18) | 0,39 m (1,06) |
| 3 | 9006 | 28,7 % | 0,08 m (0,17) | 1,18 m (1,18) |
| 3 | 9007 | 98,2 % | 0,22 m (0,22) | 0,81 m (0,81) |
| 4 | 9000 | 100,0 % | 0,00 m (0,11) | 0,62 m (1,56) |
| 4 | 9001 | 99,0 % | 2,03 m (2,23) | 3,60 m (3,82) |
| 4 | 9002 | 99,2 % | 0,26 m (0,45) | 1,19 m (3,42) |
| 4 | 9003 | 59,7 % | 0,18 m (0,23) | 2,82 m (2,83) |
| 4 | **9004** | 95,7 % | 0,29 m (0,33) | 2,61 m (2,65) |
| 4 | 9005 | 99,2 % | 0,12 m (0,27) | 0,70 m (1,09) |
| 4 | 9006 | 100,0 % | 0,23 m (0,45) | 2,80 m (5,69) |
| 4 | 9007 | 99,4 % | 0,09 m (0,13) | 0,56 m (0,72) |

- Sur ces 16 vols, l'erreur finale du SLAM est inférieure à celle de l'odométrie seule dans les 16 cas ; elle est de 0,2 m ou moins dans 10 cas sur 16.
- Une exception notable : niveau 4, seed 9001, le SLAM dérive à son tour (2,03 m) : cas non expliqué.
- Les seeds 9006 (niveau 3, couverture 28,7 %) et 9003 (niveau 4, 59,7 %) sont des vols où l'exploration s'est arrêtée très tôt (critère de fin : plus aucune frontière valide) : problème de planification, pas de localisation ; non étudié.
- Les deux vidéos retenues (en gras) ont été choisies pour le contraste ; l'erreur RMSE médiane du SLAM au bruit fort est de 0,12 m (mesure des GATES 2 et 3).

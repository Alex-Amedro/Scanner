# Journal d'expérimentation — Drone explorateur
 
*Complète `eval_logs/journal.csv` (les chiffres bruts) avec le contexte
narratif : pourquoi chaque changement, ce qu'on en attendait, ce qu'on en a
tiré. Une entrée par version, ajoutée au fur et à mesure.*
 
---
 
## v1 — baseline cassée
 
**Changements** : première version entraînée, 500k steps, sur l'architecture
issue de nos discussions (CNN sur crop local + MLP sur frontières/relief/
cinématique, PPO).
 
**Résultat** : 100% de morts par collision, couverture ~35-40%, drone
quasiment impilotable.
 
**Diagnostic** : les `gear` roll/pitch/yaw du XML avaient été recopiés bruts
(2.0/2.0/1.0) au lieu des valeurs calibrées par `test_gear.py` sur le projet
précédent (0.5/0.5/0.25, validées à 97-100% de succès contre 20-33% avec les
valeurs brutes). Erreur de copier-coller, pas un problème d'entraînement.
 
## v2 — gear corrigé + conscience de l'orientation
 
**Changements** (groupés — avant qu'on applique la règle d'un changement à
la fois) :
- `gear_roll_pitch=0.5`, `gear_yaw=0.25` (valeurs calibrées)
- Ajout de roll, pitch, uprightness à l'observation (`kinematics` 7→10
  valeurs) — avant, le réseau ne percevait pas son inclinaison actuelle
- Mort immédiate si uprightness < 0,3 (~70° d'inclinaison), avant seule la
  collision terminait l'épisode
**Résultat** (30 épisodes, seeds 9000-9029) : 90% mort par collision, 10% par
retournement, 0% victoire, couverture moyenne 58,5% (±7,9), trajectoire
moyenne 15,4 m, survivant repéré dans 33% des épisodes. Nette progression
par rapport à v1 (épisodes 3 à 4 fois plus longs, couverture presque doublée).
 
**⚠️ Note pour les comparaisons futures** : ces chiffres de couverture ont été
mesurés *avant* la découverte du bug de mesure ci-dessous — pas directement
comparables aux évaluations faites avec le code corrigé sur ce point
précis. Les taux mort/retourné/survivant, eux, restent valides.
 
## Bug trouvé entre v2 et la suite : couverture plafonnée par un artefact de mesure
 
La grille est un rectangle dimensionné sur la pièce la plus large générée ;
les pièces plus étroites laissent de l'espace mort (jamais atteignable par
le drone, jamais atteint par le LiDAR) qui comptait quand même dans le
dénominateur de la couverture — plafonnant la couverture max possible bien
avant 100% selon la variabilité des tailles de pièces.
 
**Corrigé** : `coverage_ratio()` calculée uniquement sur les cellules
réellement navigables (masque calculé depuis les rectangles des pièces et
couloirs à la génération). Vérifié sur un cas de test : une exploration
parfaite simulée donnait 66% avant le fix, 100% après.
 
## v2.1 — bonus d'alignement cap/déplacement (EN COURS)
 
**Hypothèse testée** : le comportement de "toupie" observé en visuel vient
du fait qu'un LiDAR (capteur de proximité) découple naturellement le cap de
la direction de déplacement — contrairement à la navigation par caméra, où
les deux sont normalement liés. Documenté dans un papier 2025 (*Quadrotor
navigation considering attitude*, ScienceDirect), qui introduit un "tangent
path reward" pour cette raison précise.
 
**Changement (un seul, règle appliquée à partir de maintenant)** :
`coeff_align=0.2` — bonus = cos(angle entre le cap et la direction de
vitesse horizontale), actif seulement si la vitesse horizontale dépasse
0,3 m/s (en dessous, la direction de vitesse est trop bruitée pour être
significative).
 
**Point de départ** : les poids de v2 (`--from-version v2`), pas un
entraînement depuis zéro — pour isoler l'effet du changement sur une base
déjà fonctionnelle plutôt que réapprendre le vol de base en même temps.
 
**Résultat brut** (30 épisodes, seeds 9000-9029) : 90% victoire, 3% mort,
7% retourné, couverture moyenne 88,3%, 92 steps en moyenne jusqu'à la
victoire. Saut spectaculaire par rapport à v2 (0% victoire).
 
**⚠️ Problème méthodologique découvert après coup** : cette comparaison
v2 vs v2.1 a DEUX facteurs confondus, pas un seul comme prévu :
1. v2.1 a reçu 500k steps *en plus* de ceux déjà faits sur v2 — soit
   1 million de steps cumulés contre 500k pour v2. Un budget d'entraînement
   double peut à lui seul expliquer une grosse partie du progrès, sans
   rapport avec `coeff_align`.
2. Le bug de mesure de couverture a été corrigé entre les deux — la barre
   des 90% était quasiment inatteignable avant le fix (à cause de l'espace
   mort compté contre le drone), et devient un objectif raisonnable après.
Observation qui corrobore ce doute : en `--visual`, le comportement de
"toupie" est toujours visible malgré le score de victoire — si l'alignement
avait vraiment réglé la rotation, on ne devrait plus la voir du tout.
Verdict sur l'hypothèse "tangent path reward" : **en suspens**, pas validé
par ce résultat. Erreur de protocole : `--from-version` n'aurait pas dû être
utilisé ici sans un groupe de contrôle à budget d'entraînement égal.
 
**Contrôle envisagé puis abandonné** : `v2.2` (= v2 continué sans
`coeff_align`, pour comparer à budget égal). Abandonné au profit de mieux :
puisque v2 a été entraîné *from scratch*, la comparaison propre à une seule
variable, c'est un `v3` *from scratch* avec `coeff_align` actif dès le
départ, sur le même budget (500k) — pas besoin de relancer un contrôle, v2
sert déjà de référence "sans alignement, from scratch, 500k". Ça évite en
plus le risque qu'une politique déjà convergée (v2) mette plus de temps à
désapprendre une habitude de spin qu'à ne jamais la prendre.
 
## v3 — alignement dès le départ, from scratch (À LANCER)
 
**Objectif** : comparaison propre à une seule variable contre v2 — même
point de départ (poids aléatoires), même budget (500k steps), seule
différence : `coeff_align=0.2` actif depuis le tout premier step au lieu
d'ajouté après coup sur une politique déjà convergée.
 
**Commande** : `python train.py --name explorer --coeff-align 0.2
--total-timesteps 500000` (sans `--from-version`, sans `--resume`).
 
**Résultat brut** (30 épisodes, seeds 9000-9029) : 70% victoire, 27% mort,
3% retourné, couverture moyenne 87,1%. Net progrès vs v2 (0% victoire), à
budget d'entraînement égal cette fois — donc pas expliqué par un simple
effet de budget comme pour v2.1.
 
**⚠️ Mais un problème bien plus grave découvert via `--debug-angles`** : la
vitesse angulaire (`v_ang`) grimpe de façon quasiment continue tout au long
des épisodes, jusqu'à des valeurs absurdes (195+ en fin d'épisode, contre
~1-10 en début). Le drone tourne de plus en plus vite sans jamais se
stabiliser. Diagnostic : le corps du drone n'a **aucun amortissement**
défini sur son articulation libre — rien ne s'oppose à l'accumulation de
vitesse angulaire si la politique applique un couple de lacet même
faiblement biaisé dans un sens. Aggravant : `up_z` (qui déclenche la mort
par retournement) est **invariant au lacet pur** — un drone qui spin sur
lui-même sans jamais basculer roll/pitch ne déclenche jamais la sécurité.
Confirmé en isolant un couple de lacet constant sur un joint sans
amortissement : croissance non bornée de la vitesse angulaire.
 
**Conséquence méthodologique** : ce résultat ne permet PAS de valider
`coeff_align` — le spin incontrôlé était déjà présent avant même que
l'alignement entre en jeu, donc le progrès observé (0%→70% victoire)
pourrait venir d'ailleurs (le drone couvre peut-être plus de terrain juste
en tournant vite sur lui-même, sans rapport avec un meilleur pilotage).
L'hypothèse "tangent path reward" reste **non tranchée**.
 
## v4 — amortissement de l'articulation libre (À LANCER)
 
**Changement (un seul)** : `damping="0.05"` ajouté au joint libre du corps
du drone (physiquement réaliste — un vrai drone est freiné par l'air ; le
nôtre ne l'était pas du tout). `coeff_align` remis à 0 pour cette run, pour
ne pas mélanger deux changements non encore isolés proprement.
 
**Vérifié avant entraînement** : sous couple de lacet maximal soutenu, la
vitesse angulaire plafonne maintenant à ~5 (au lieu de croître sans limite).
 
**Commande** : `python train.py --name explorer --total-timesteps 500000`
(from scratch — la physique elle-même a changé, pas juste le reward, donc
repartir d'une politique habituée à l'ancien comportement n'a pas de sens,
même logique que pour le fix du gear en v2).
 
**Résultat brut** (30 épisodes, seeds 9000-9029) : 67% victoire, 20% mort,
13% retourné, couverture moyenne 87,3%. Comparable à v3 en surface.
 
**⚠️ Vérifié via `--debug-angles` : le spin n'est PAS réglé.** `v_ang`
continue de grimper tout au long des épisodes (jusqu'à 135 en fin
d'épisode, contre ~195 sur v3 — une baisse réelle mais très insuffisante).
La validation isolée pré-entraînement (couple constant sur un seul axe →
plafond à 5) ne reflétait pas ce qui se passe réellement : la politique
entraînée bouge sur les 4 canaux en même temps, en changeant de sens sans
arrêt — un scénario que ce test isolé ne couvrait pas. L'amortissement seul
ne suffit pas.
 
**Décision** : retirer l'amortissement (pas le garder en plus d'un nouveau
changement — éviter d'empiler). Retour à la physique de v2/v3 (aucun
amortissement) pour le prochain test.
 
## v5 — pénalité de vitesse angulaire (`coeff_spin`) (À LANCER)
 
**Changement (un seul)** : `coeff_spin=0.02` — mécanisme déjà présent dans
le code depuis un moment (jamais activé jusqu'ici) : pénalité = coefficient
× norme de la vitesse angulaire, à chaque step. Amortissement retiré pour
ce test (pas cumulé avec v4). Contrairement à `coeff_agressivite` sur le
projet précédent (qui punissait l'amplitude de commande et avait causé une
régression), celui-ci punit directement l'état de vitesse angulaire — plus
proche de la cause du problème que de son symptôme de surface.
 
**Ordre de grandeur choisi** : à `v_ang≈5-10` (début d'épisode, normal),
coût ≈0,1-0,2 par step (quasi négligeable) ; à `v_ang≈100+` (l'emballement
qu'on veut tuer), coût ≥2 par step (nettement supérieur aux récompenses
observées jusqu'ici).
 
**Commande** : `python train.py --name explorer --coeff-spin 0.02
--total-timesteps 500000` (from scratch — pénalité de reward nouvelle,
même logique que pour les changements précédents).
 
**Résultat** : ✅ **RÉUSSI.** `v_ang` reste borné entre 1 et 11 sur tout
l'épisode (contre un emballement jusqu'à 195 sans la pénalité) — plus aucune
accélération de rotation sans limite. Résultat sur 3 épisodes debug : 100%
victoire, 0% mort, 0% retourné, couverture 90,4%. À confirmer sur un plus
grand échantillon (30-50 épisodes), mais le problème de fond (spin
incontrôlé) semble réglé. Modèle retenu comme nouvelle base de travail.
 
## Fix : désaxage des couloirs (contre le "couloir sniper")
 
**Changement (un seul)** : chaque couloir a maintenant sa propre position en
x (`corridor_x_offsets`), tirée aléatoirement mais contrainte pour rester
dans les deux pièces qu'il relie (avec marge). Avant, tous les couloirs et
toutes les portes étaient centrés en x=0 par construction — un rayon LiDAR
bien aligné pouvait voir à travers plusieurs portes d'affilée jusqu'au fond
du bâtiment. Vérifié visuellement avec `debug_grid.py` : le couloir fait
maintenant un coude à chaque porte, la ligne de vue rectiligne est cassée.
 
**À tester** : entraînement frais avec ce changement, sur la base de
`coeff_spin=0.02` (la config qui marche) :
`python train.py --name explorer --coeff-spin 0.02 --total-timesteps 500000`
→ créera `v6`.
 
## v5.1 / v5.2 / v6 — résultats qualitatifs (chiffres détaillés pas encore recueillis)
 
**v5.1** : `--from-version v5`, génération de bâtiment désaxée, 300k steps
supplémentaires. Observation (visuelle, pas encore quantifiée) : garde les
habitudes "fonce tout droit" de v5, mal adaptées aux nouveaux couloirs
coudés.
 
**v5.2** : `--from-version v5.1`, seuil `up_z_min` abaissé (plus tolérant
à l'inclinaison avant de déclencher la mort par retournement). Observation :
mouvements erratiques/étranges.
 
**v6** : PAS une continuation — from scratch, avec la génération de
bâtiment désaxée (même changement que v5.1, mais poids aléatoires plutôt
que repartir de v5). Commande utilisée d'après le paste utilisateur :
`python train.py --name explorer --coeff-spin 0.02 --total-timesteps 500000`
(donc en principe `coeff_spin=0.02` actif). Observation : **le spin
incontrôlé est revenu**, alors que v5 (mêmes réglages de reward, ancienne
génération de bâtiment) n'avait pas ce problème.
 
**⚠️ Point à vérifier en priorité dans la prochaine conversation** :
confirmer les réglages RÉELLEMENT utilisés pour v6 en lisant directement
`models/explorer/v6/metadata.json` (champ `env_kwargs`) — plutôt que de
se fier à la commande retapée de mémoire. Si `coeff_spin` y est bien à
0.02, alors le retour du spin s'explique par autre chose que le reward
(peut-être : les nouveaux couloirs coudés demandent des virages plus
brusques, rendant `coeff_spin=0.02` insuffisant sur cette géométrie plus
dure ; ou variance d'entraînement normale sur un seul run from-scratch).
 
## Résolution du point ouvert sur v6 (metadata confirmé + éval quantitative)
 
**Vérification demandée dans la passation** : `models/explorer/v6/metadata.json` confirme
`coeff_spin=0.02`, run complet à 500k steps — pas une erreur de commande.
 
**Résultat** (30 épisodes, seeds 9000-9029) : 27% victoire, 73% mort, 0% retourné, couverture
moyenne 72,9% (±17,2). Net recul vs v5 (96% victoire, 89,3% couverture) — mais la comparaison
n'est plus à variable unique, la génération de bâtiment a aussi changé entre les deux.
 
**`--debug-angles`** : `coeff_spin` fait bien quelque chose (plafond `v_ang` ~40-65, pas 195+
comme v3) mais très en dessous du "1 à 11" obtenu sur v5 avec le même coefficient — donc soit le
réglage est sous-dimensionné pour cette géométrie plus dure (couloirs désaxés), soit c'est de la
variance normale sur un seul run from-scratch (cf. découverte sur le seed plus bas, qui invalide
un peu cette distinction). Le yaw ne fait pas des allers-retours (pas un vrai "zigzag" de manœuvre)
mais grimpe de façon quasiment continue dans un seul sens sur tout l'épisode — les wraps ±180°
dans les chiffres bruts donnaient une fausse impression de va-et-vient. Bascules roll/pitch
extrêmes déjà présentes en fin d'épisode (roll jusqu'à -58°, `up_z` jusqu'à 0,47) — pattern qui va
s'avérer important pour la suite (v8).
 
**Piste VecNormalize testée et écartée** : std du retour normalisé plus bas sur v6 (263) que sur
v5 (583) — si ça jouait un rôle, ça amplifierait `coeff_spin` sur v6, pas l'atténuerait.
N'explique donc pas le recul observé.
 
## v7 — `coeff_spin` renforcé (0.08) — ÉCHEC
 
**Changement (un seul)** : `coeff_spin` 0.02→0.08, `coeff_align=0.0`, from scratch, même
génération désaxée que v6.
 
**⚠️ Budget d'entraînement coupé** : arrêté à 350k/500k steps (courbe tensorboard jugée sans
espoir, même trajectoire que les échecs précédents) — comparaison à v6 biaisée par un facteur
budget en moins, même défaut méthodologique que v2.1.
 
**Résultat** (30 épisodes) : 3% victoire, 93% mort, 3% retourné, couverture moyenne 65,9%. Net
recul vs v6 (27%/72,9%).
 
**`--debug-angles`** : `v_ang` mieux borné (~30 contre ~65 sur v6) mais victoire et couverture
s'effondrent quand même — punir plus fort la vitesse angulaire *en bloc* dégrade le contrôle plus
qu'il ne règle le problème. Ne distingue pas "spin inutile" de "couple nécessaire pour manœuvrer/
éviter un mur".
 
**Décision** : abandon de la piste "augmenter `coeff_spin`". Retour au réglage v6 (0.02) comme
base, pour tester `coeff_align` à la place (idée en réserve depuis la v2.1, jamais tranchée).
 
## v8 — `coeff_align=0.2` (tangent path reward), enfin testé proprement contre v6 — SUCCÈS PARTIEL
 
**Changement (un seul)** : `coeff_align` 0.0→0.2, `coeff_spin=0.02` (réglage v6), from scratch,
500k steps confirmés (`metadata.json` vérifié) — élimine le facteur budget qui avait biaisé v2.1
et v7.
 
**Résultat** (30 épisodes) : 37% victoire, 53% mort, 10% retourné, couverture moyenne 78,6%.
Premier changement testé depuis le fix du spin (v5) qui va dans le bon sens plutôt que le mauvais.
 
**⚠️ Nouveau mode d'échec : `retourné` 0%→10%.** `--debug-angles` montre des bascules roll
extrêmes en cours de vol (pas juste en fin d'épisode) — jusqu'à `roll=-68°`, `up_z=0,37`, à deux
doigts du seuil de mort (0,3).
 
**Précision importante après relecture de la trace v6** : ce même pattern de bascule extrême
(`roll` jusqu'à -58°, `up_z` 0,47) existe déjà sur v6, en fin d'épisode seulement. `coeff_align`
n'invente donc pas ce comportement, il l'amplifie — plus fréquent, plus profond, au point de
franchir le seuil dans 10% des cas. Explication physique probable : la poussée étant verticale
dans le repère du corps (`gear="0 0 20 0 0 0"`), rediriger vite la vitesse passe par du roll/pitch
— et `coeff_align` récompense justement une réorientation rapide, donc la politique a intérêt à
banker plus fort/plus souvent.
 
**Hypothèse "perte de contrôle quand la frontière est loin" (proposée par l'utilisateur) réfutée
sur cette évidence précise** : ajout de `nearest_frontier_dist` (distance normalisée à la
frontière valide la plus proche) à l'observation de debug. Sur les 3 épisodes testés, reste bas
(0,05-0,33) tout du long, jamais de cas de frontière lointaine observé pour même tester
l'hypothèse — et les pires bascules arrivent avec `frnt_d` bas (frontière proche), pas haut.
 
## Check gratuit : `up_z_min` desserré sur v8 sans réentraîner — pas le bon levier
 
**Changement d'évaluation seul, aucun réentraînement** : ajout d'un override `--up-z-min` à
`evaluate.py` (l'espace d'observation ne change pas, `VecNormalize` reste compatible). Testé sur
le modèle v8 existant à 0,15 au lieu de 0,3.
 
**Résultat** : quasi identique (37% victoire inchangé, couverture 78,6%→79,3% dans le bruit, mort
53%→57%, retourné 10%→7%). Un seul épisode bascule d'issue (retourné→mort, en tapant un mur 21
steps plus tard au lieu de se retourner tout de suite).
 
**Conclusion : `up_z_min` n'est pas le bon levier.** Les bascules extrêmes ne sont pas des morts
"au bord" qui seraient évitées avec plus de marge — c'est le symptôme d'un vol déjà compromis
(route vers une collision), pas la cause. Piste abandonnée, retirée de la liste des tests à faire
— même verdict que pour l'idée d'agrandir les espaces (cf. check couloir/pièce plus bas).
 
## v9 — `coeff_align` réduit (0.1) — comportement NON-MONOTONE, remet en cause la méthode
 
**Changement (un seul)** : `coeff_align` 0.2→0.1 (reste identique à v8), from scratch, 500k steps
confirmés.
 
**Résultat** (30 épisodes) : 13% victoire, 67% mort, 20% retourné, couverture moyenne 73,8% — pire
que v8 (37%/78,6%) ET pire que v6 sans `coeff_align` du tout (27%/72,9%).
 
**⚠️ Point critique.** `v_ang` plafonne à ~65 sur v9 (comme v6, sans `coeff_align`), pas comme v8
(~20-23). Sur les trois réglages testés (`coeff_align` 0.0 / 0.1 / 0.2), la réponse n'est pas
graduelle : 0.1 se comporte comme 0.0 (mauvais), 0.2 comme l'exception qui marche bien. Un vrai
effet de coefficient donnerait une réponse monotone, pas ce zigzag.
 
**Cause identifiée** : `train.py` ne fixait aucun seed pour l'initialisation du modèle ni la
stochastique PPO — seuls les seeds des 8 environnements d'entraînement étaient fixes (0 à
n_envs-1, indépendants de tout hyperparamètre). Chaque lancement de `train.py`, même à
hyperparamètres identiques, part donc d'un tirage différent. **v6, v8 et v9 pourraient chacun être
un tirage chanceux ou malchanceux, indépendamment de `coeff_align` — remet en cause la fiabilité
de toutes les comparaisons de versions faites sur un seul run jusqu'ici.**
 
**Fix appliqué** : `--seed` ajouté à `train.py` (init modèle + stochastique PPO), consigné dans
`metadata.json` aux côtés de `env_kwargs`. `evaluate.py` et `explorer_env.py` patchés en
cohérence (voir section outillage plus bas).
 
## Check gratuit : localisation des collisions (couloir vs pièce) sur v8 — réfute "espace trop restreint"
 
**Changement d'évaluation seul** : ajout de `_in_corridor()` à `evaluate.py`, classe chaque
collision comme "couloir" ou "pièce" selon la position du drone à l'impact
(`compute_navigable_rects`).
 
**Résultat sur v8** (30 épisodes, 16 morts par collision) : 88% des collisions en pièce (14/16),
12% en couloir (2/16). Cohérent avec le gabarit de collision réel du drone (~0,2×0,3m — seuls le
corps et le petit bloc avant ont `contype`/`conaffinity` actifs, bras et hélices sont purement
cosmétiques) largement plus petit que les pièces (4-7m) et les couloirs (1,6m de large).
 
**Conclusion : réfute l'hypothèse "espace trop restreint".** Pas la peine d'agrandir bâtiments ou
couloirs — les collisions viennent très probablement des bascules de pilotage documentées sur v8,
pas d'un manque de place. Piste abandonnée, même verdict que `up_z_min`.
 
## v10 — réplication exacte de v8 avec seed fixé (À LANCER)
 
**Objectif** : vérifier si le résultat de v8 (37% victoire, `coeff_align=0.2`) est robuste, ou
juste un tirage d'initialisation chanceux — vu la découverte ci-dessus sur v9.
 
**Commande prévue** : `python train.py --name explorer --coeff-spin 0.02 --coeff-align 0.2 --seed
42 --total-timesteps 500000` (from scratch, hyperparamètres identiques à v8 au bit près, seule
différence : seed fixé pour la reproductibilité — à noter dans le journal comme "réplication de
v8", pas un nouveau réglage, une fois le résultat obtenu).
 
**⚠️ Point de vigilance machine** : v1 à v9 ont tous tourné sur le même PC portable (neuf, RTX
5070 / Intel Core Ultra 7 255HX). Un lancement sur le PC fixe (RTX 2060 / 12 cœurs, différent)
avait été envisagé le temps que le patch `--seed` soit prêt — abandonné pour ne pas mélanger deux
variables (seed + machine) dans la même comparaison. v10 doit tourner sur le même portable que
v1-v9 pour rester une comparaison propre.
 
**Piste GPU en cours d'investigation, ne bloque pas v10** : la RTX 5070 (Blackwell, `sm_120`)
nécessite un build PyTorch compilé CUDA 12.8+ pour être reconnue — sinon fallback silencieux vers
le CPU (confirmé par recherche : problème connu et documenté pour cette génération de carte, pas
un souci de bascule Intel/NVIDIA comme pour un jeu). À vérifier via `torch.cuda.is_available()`
avant de conclure quoi que ce soit sur GPU vs CPU pour ce projet — vu la taille minuscule du
réseau (petit CNN+MLP), le vrai goulot reste probablement la physique MuJoCo (CPU-bound, un thread
par env), donc le gain potentiel du GPU est incertain. Pas prioritaire : v10 peut tourner sur CPU
sans attendre que ce soit résolu.
 
## v11 — `damping=0.05` + `coeff_spin=0.02` (idée en réserve depuis v4), `coeff_align` retiré — MEILLEUR RÉSULTAT DEPUIS LE DÉSAXAGE DES COULOIRS (⚠️ checkpoint original détruit par un bug `--resume`, à reproduire — voir incident plus bas)
 
**Changement** : retour à `coeff_align=0.0` (3 runs consécutifs — v8/v9/v10 — sans effet fiable),
`joint_damping=0.05` ajouté au joint libre (idée testée seule en v4, *avant* l'existence de
`coeff_spin`, jugée insuffisante à l'époque — jamais testée en même temps que `coeff_spin`). Pas
un test à variable strictement unique par rapport à v10 (deux choses changent : align retiré,
damping ajouté), assumé sciemment après trois échecs consécutifs de réglage fin. From scratch,
`coeff_spin=0.02` (réglage v6), `seed=42` (même seed que v10, pour une comparaison au moins
partiellement contrôlée), 500k steps confirmés.
 
**Nécessite un ajout de code** : `joint_damping` n'était pas exposé — ajouté à `ExplorerEnv`
(`explorer_env.py`, appliqué au `<joint type="free" damping="...">` du corps du drone) et à
`train.py` (`--damping`).
 
**Résultat** (30 épisodes) : 43% victoire, 47% mort, 10% retourné, couverture moyenne 81,4%
(±13,1). Meilleur score obtenu depuis le désaxage des couloirs (v6) — dépasse même le score
chanceux de v8 (37%). Comparaison la plus parlante : à seed strictement identique (42), v10 (align,
pas de damping) donnait 13% ; v11 (damping, pas d'align) donne 43%. Collisions toujours
massivement en pièce (13/14, 93%) — cohérent avec v8/v10, confirme une nouvelle fois que ce n'est
pas un problème d'espace.
 
**`--debug-angles` confirme la vraie amélioration structurelle** : `v_ang` reste dans la
fourchette 1-11 sur la quasi-totalité des deux épisodes testés — la plage saine de v5, plus du
tout les plafonds à 60-70 de v6/v9/v10. Le spin incontrôlé semble réglé par la contrainte physique
plutôt que par une incitation de reward.
 
**Observation qualitative (visuel + trace)** : les morts restantes ont une signature différente de
l'ancien spin. Sur l'épisode 0 (mort au step 78), `v_ang` reste bas et stable (1-8) jusqu'au bout,
puis bondit d'un coup à 30,98 *au step même de la collision* — pas une dérive progressive qui
couvait, un pic isolé au moment de l'impact (cohérent avec un à-coup physique de contact, pas une
perte de contrôle continue). Lecture de l'utilisateur en observant le rendu `--visual` : les morts
restantes ressemblent à des "morts d'inertie" — le drone identifie la bonne direction mais garde
trop d'élan de la manœuvre précédente pour freiner/tourner à temps ; observé aussi des
rétablissements réussis après une bascule complète (accélération vers l'arrière puis
restabilisation), donc pas un problème de "compétence de freinage" absente, plutôt un défaut de
précision de timing — catégorie d'échec différente et plus ordinaire que le spin structurel
chassé depuis v6.
 
**Courbe tensorboard** : montée aussi rapide que v5 en début d'entraînement (cohérent avec le
damping qui élimine un piège d'instabilité que la politique devait auparavant apprendre à éviter,
contrairement aux versions précédentes qui perdaient du temps dedans), mais se stabilise vers un
plateau proche des valeurs de v8. Hypothèse à tester : la difficulté résiduelle (précision de
freinage/virage sur la géométrie désaxée) demande plus que 500k steps pour se polir, contrairement
au spin qui était un mécanisme cassé plutôt qu'un manque de pratique.
 
**Bug corrigé au passage** : `model_manager.list_versions()`/`latest_version()` triaient les
dossiers alphabétiquement (`sorted(os.listdir(...))`) — cassait dès `v10`/`v11` (`"v9" > "v11"` en
tri texte). `--resume` aurait repris silencieusement `v9` au lieu de `v11` sans erreur visible.
Corrigé avec une clé de tri numérique (`_version_sort_key`, parse `vN`/`vN.M`).
 
## ⚠️ INCIDENT — `v11` (checkpoint 500k, le bon) détruit par un bug de `--resume`
 
**Ce qui a été tenté** : `python train.py --name explorer --resume --total-timesteps 500000`, pour
tester si `v11` (43% victoire à 500k) continuait de progresser avec plus de budget — sans retaper
`--coeff-spin 0.02 --damping 0.05 --seed 42`.
 
**Bug réel dans `train.py`** (présent depuis la version d'origine du fichier, jamais audité avant
de recommander `--resume`) : `env_kwargs` était construit à partir des flags CLI *avant même* de
regarder si `--resume` était utilisé. Tout flag non retapé retombe silencieusement sur son défaut
argparse (`coeff_spin=0.0`, `joint_damping=0.0`, `seed=None`). Le modèle rechargé (`PPO.load`)
était le bon, mais **les 500k steps suivants ont tourné dans un environnement sans amortissement
ni pénalité de spin** — pas juste un problème d'affichage de métadonnées, une vraie divergence de
physique et de reward en cours d'entraînement.
 
**Résultat** : `metadata.json` de `v11` après coup confirme `coeff_spin: 0.0, coeff_align: 0.0,
joint_damping: 0.0, seed: null` — plus aucun rapport avec les réglages d'origine. Éval sur le
`v11` à 1M steps : 40% victoire, 37% mort, **23% retourné** (contre 10% à 500k), couverture 74,2%
— dégradation nette du taux de retournement, cohérente avec la perte de l'amortissement pendant la
deuxième moitié de l'entraînement. `--debug-angles` confirme : bascules extrêmes de nouveau
présentes (`up_z` jusqu'à 0,35-0,40, `v_ang` jusqu'à 31 en fin d'épisode).
 
**Le checkpoint `v11` à 500k (le bon, 43% victoire) est très probablement perdu** —
`RotatingCheckpointCallback` écrase en continu, la sauvegarde finale du run cassé a aussi écrasé
`models/explorer/v11/`. Pas de backup connu au moment de l'incident.
 
**Fix appliqué à `train.py`** : pour `--resume` spécifiquement, `env_kwargs` est maintenant
*toujours* relu depuis le `metadata.json` de la version reprise, jamais reconstruit depuis les
args CLI (qui sont désormais explicitement ignorés et un message le confirme au lancement).
Erreur explicite si une version n'a pas d'`env_kwargs` en metadata plutôt qu'un silence dangereux.
`--from-version` et l'entraînement from-scratch affichent maintenant aussi `env_kwargs` avant de
lancer, pour repérer ce genre de dérive avant de perdre du temps de calcul dessus.
 
**Décision** : `v11` (1M steps) est invalide, à ignorer/écraser. Prochaine étape : réentraîner
`v11` from scratch directement à 1M steps (pas de `--resume`), avec les réglages d'origine :
 
```
python train.py --name explorer --coeff-spin 0.02 --damping 0.05 --seed 42 --total-timesteps 1000000
```
 
## v12 — reproduction propre de v11 (recette identique, from scratch, 1M steps) — NOUVEAU MEILLEUR SCORE
 
**Contexte** : suite à l'incident `--resume` (v11 corrompu, voir plus haut), réentraînement direct
from scratch avec la recette d'origine de v11 (`coeff_spin=0.02`, `damping=0.05`, `seed=42`) mais
`--total-timesteps 1000000` d'un coup, sans passer par `--resume`. `v11` existant déjà sur le
disque (dans son état corrompu), le nouveau run a pris le numéro suivant : **`v12`, pas une
nouvelle mise à jour de `v11`**. À retenir pour la suite : `v11` sur le disque = le run corrompu
(1M steps, `coeff_spin`/`damping` retombés à 0 en cours de route, à ignorer) ; `v12` = la
reproduction propre.
 
**Résultat** (30 épisodes) : 47% victoire, 50% mort, 3% retourné, couverture moyenne 82,9%
(±10,5). Légèrement meilleur que le `v11` original à 500k (43% victoire, 10% retourné, couverture
81,4%) — amélioration réelle mais modeste avec le double de budget, pas un plateau strict ni un
saut spectaculaire. Collisions toujours à 100% en pièce (15/15 puis 2/2 sur des sous-échantillons)
— confirme une nouvelle fois que ce n'est pas un problème d'espace.
 
**Observation qualitative (utilisateur, visuel)** : rendu "plutôt bon et naturel" pour un usage
portfolio. Reste un défaut mineur observé : marge de virage/anticipation un peu juste par moments
("sur l'angle il est un peu trop petit") — des morts qui ne viennent pas de mouvements aberrants,
juste d'un ajustement trop tardif ou trop franc par rapport à l'élan déjà engagé. Cohérent avec la
lecture "mort d'inertie" déjà notée sur v11.
 
**Trace `--debug-angles`** : confirme le pattern déjà observé sur v11 — `v_ang` reste modéré
(2-9) sur la quasi-totalité de chaque épisode, puis bondit d'un coup *au step exact de la mort*
(21,4 / 55,3 / 10,3 sur les 3 épisodes détaillés) plutôt que de dériver progressivement. Cohérent
avec un à-coup de contact physique plutôt qu'une perte de contrôle qui couvait.
 
**⚠️ Diagnostic prévu mais pas encore obtenu** : les colonnes `speed`/`minLid` (ajoutées à
`evaluate.py`/`explorer_env.py` pour tester quantitativement l'hypothèse "distance de freinage mal
anticipée") n'apparaissent pas dans cette trace — l'éval a tourné avec des fichiers antérieurs à ce
patch. À refaire avec les fichiers à jour si on veut confirmer l'hypothèse avant d'agir dessus.
 
## Nouvelle piste : `coeff_proximity` — pénalité de proximité (PAS ENCORE LANCÉE)
 
**Diagnostic préalable (sur `v12`, via `speed`/`minLid` ajoutés à `--debug-angles`)** :
confirmation quantitative que les morts restantes ne sont pas un problème de vitesse — la vitesse
continue de monter jusqu'à l'impact dans les trajectoires de mort (`speed` 12→13+ pendant que
`minLid` s'effondre 2,2→0,2). Mais un épisode victorieux montre le même pattern de vitesse
croissante en s'approchant d'un obstacle (`minLid` 0,6) — la différence entre survie et mort n'est
donc pas la vitesse, c'est si le virage d'évitement engagé réussit ou non à temps. Une pénalité
"vitesse × proximité" pénaliserait donc autant les manœuvres réussies que les ratées — mauvais
levier, abandonné avant même d'être testé.
 
**Piste retenue à la place** : pénalité de proximité pure (basée sur `min_lidar_dist` seul), pour
donner un gradient plus dense et précoce vers "s'écarter" que la seule pénalité terminale de
collision (-50).
 
**⚠️ Premier réglage proposé (seuil à 1.0) invalidé avant l'entraînement** — remarque de
l'utilisateur, vérifiée empiriquement plutôt que suppposée : `corridor_width=1.6` →
`gap_half=0.8`, donc un couloir vit naturellement autour de 0,8 de distance au mur le plus proche.
Vérification sur `v12` (10 épisodes, nouvelle colonne `loc` en `--debug-angles`, split couloir/
pièce) : **minLid en couloir : médiane 0,55, min 0,26 — minLid en pièce : médiane 1,82, min
0,02.** Un seuil à 1.0 aurait pénalisé la moitié du temps passé en couloir, sans rapport avec le
vrai problème (0% des collisions y arrivent). Descendre le seuil pour éviter ça (~0,25, sous le
minimum couloir observé) ne marche pas non plus : trop proche des valeurs où le danger devient
réel en pièce, le signal ne se déclencherait qu'1-2 steps avant l'impact — pas assez tôt pour être
utile. Les deux distributions (couloir vs pièce dangereuse) se chevauchent trop pour qu'un seuil
unique règle les deux problèmes à la fois.
 
**Fix retenu** : restreindre la pénalité aux pièces (`ExplorerEnv._in_corridor()`, même logique que
le helper d'`evaluate.py`, rectangles de couloir mis en cache une fois par épisode) plutôt que de
chercher un seuil universel. Permet de remonter le seuil à 0,6 (généreux vis-à-vis de la médiane
en pièce à 1,82) sans faux positifs en couloir.
 
**Commande prévue** :
```
python train.py --name explorer --coeff-spin 0.02 --damping 0.05 --coeff-proximity 0.3 --proximity-threshold 0.6 --seed 42 --total-timesteps 1000000
```
→ `v13`, from scratch, même seed que v11/v12, seule nouvelle variable vs `v12` :
`coeff_proximity`.
 
## v13 — `coeff_proximity=0.3` (seuil calibré empiriquement) — ÉCHEC, RÉGRESSION
 
**Résultat** (30 épisodes, seed=42, 1M steps confirmés) : 30% victoire, 50% mort, **20%
retourné** (contre 3% sur v12), couverture 74,4% — net recul sur toute la ligne par rapport à v12
(47% / 3% / 82,9%). Confirmé par l'utilisateur sur tensorboard : courbe moins bonne que v12.
 
**Nouveau symptôme inquiétant** : plusieurs épisodes `retourné` en seulement 10-11 steps — un
retournement quasi immédiat, jamais observé auparavant (les retournements précédents prenaient des
dizaines de steps à se développer).
 
**Diagnostic** : même mécanisme que `coeff_align` en v8 — un reward qui pousse vers "s'écarter
d'un obstacle" incite la politique à emprunter le levier le plus rapide pour ça (un bank agressif
en roll/pitch), pas une trajectoire propre, ce qui rapproche `up_z` du seuil de retournement plus
souvent. Le seuil calibré empiriquement (0,6, restreint aux pièces) a évité le faux problème
identifié à l'avance (faux positifs en couloir), mais n'a pas empêché ce nouveau vrai problème.
 
**Décision** : abandon de `coeff_proximity` — retour à `coeff_proximity=0.0`.
 
## Ce que le modèle observe RÉELLEMENT — écart identifié entre diagnostic et observation
 
Question de l'utilisateur : "je veux savoir exactement ce que le modèle voit". Vérification du
code (`observation_space` + `_get_obs()`) : `local_crop` (grille d'occupation locale, **alignée
sur les axes du monde, pas sur le cap du drone** — TODO jamais fait, noté dans
`occupancy_grid.py`), `coverage`, `frontier_vector`, `relief_rays` (relief vertical, pas obstacles
horizontaux), `kinematics` (vitesse linéaire/angulaire brutes, roll/pitch/up_z — **pas de yaw
absolu**).
 
**Point central** : `min_lidar_dist`/`speed_horiz`, utilisés depuis plusieurs versions pour le
diagnostic (`--debug-angles`) et dans le calcul de reward (`coeff_proximity`), **n'ont jamais fait
partie de l'observation donnée à la politique**. Le réseau doit déduire toute proximité d'obstacle
uniquement via le CNN sur `local_crop` (downsampling agressif, fenêtre 9,6m à résolution
0,15m/cellule) — sans jamais avoir un chiffre direct de distance, et sans connaître son cap absolu
pour interpréter ce crop de façon cohérente d'une orientation à l'autre.
 
## v14 — `proximity_rays` ajouté à l'OBSERVATION (pas au reward) — RÉSULTAT NON CONCLUANT
 
**Changement** : nouvelle clé d'observation `proximity_rays` (16 bins, binning égocentrique de
`self._last_lidar_dists`). `coeff_proximity` remis à 0.0 — reward identique à v12, seule
nouveauté : l'observation. Code touché : `explorer_env.py`, `feature_extractor.py` (`scalar_keys`),
`train.py` (`NORM_OBS_KEYS`). Incompatible avec les modèles précédents (nouvel espace
d'observation).
 
**⚠️ N'a tourné que 500k steps, pas les 1M demandés** (commande coupée/mal relancée). Comparaison
à faire contre `v11` (même recette, même budget, sans `proximity_rays`), pas contre `v12` (1M).
 
**Résultat** (30 épisodes, seed=42, 500k) : 33% victoire, 53% mort, 13% retourné, couverture
78,2% — contre v11 (43%/47%/10%/81,4%) à budget identique. **Légèrement moins bon, pas mieux.**
Donner l'info brute au réseau sans incitation à l'exploiter pour ralentir n'a pas suffi —
conclusion reprise et confirmée par v15/v16 ci-dessous. `proximity_rays` reste dans l'observation
(pas retiré, coûte rien avec `coeff_proximity=0`) mais n'explique pas les gains suivants.
 
## v15 — `coeff_speed=0.03` (pénalité de vitesse générale, pas gatée par la proximité) — RÉSULTAT MIXTE
 
**Changement** : pénalité `coeff_speed * vitesse_horiz` à chaque step, générale (pas restreinte
aux moments proches d'un obstacle, contrairement à `coeff_proximity` en v13 — pour éviter le même
effet de bord). From scratch, base v12 (`coeff_spin=0.02`, `damping=0.05`), seed=42, **1M steps
confirmés** cette fois.
 
**Résultat** (30 épisodes) : 50% victoire, 27% mort (contre 50% sur v12 — net progrès), **23%
retourné (contre 3% sur v12 — net recul)**. Échec net-neutre en taux d'échec total, mais la
*nature* de l'échec a complètement changé.
 
**Diagnostic (confirmé par `speed`/`minLid` dans `--debug-angles`)** : la pénalité n'empêche pas
la vitesse de monter — un épisode `retourné` montre `speed` grimper de 0 à 14,87 en continu
jusqu'au flip, malgré un coût de ≈0,42/step à cette vitesse (plus cher que la plupart des rewards
positifs). **La politique "achète" le droit d'aller vite plutôt que de ralentir.** Et aller vite
rend les manœuvres de virage bien plus violentes (plus de moment à rediriger), d'où l'explosion de
`retourné`. Confirme le diagnostic de l'utilisateur : "il fonce dans le mur même quand il veut
tourner".
 
**Décision** : un coût négociable ne suffit pas — passer à une contrainte physique dure,
non-négociable (même levier que `damping` pour le spin, qui avait déjà mieux marché qu'une
incitation de reward).
 
## v16 — `max_speed=6.0` (plafond DUR de vitesse, pas une incitation) — MEILLEUR RÉSULTAT DU PROJET
 
**Changement** : plafond physique de vitesse horizontale, appliqué directement sur `qvel` à
**chaque sous-step** MuJoCo (pas une fois par step, pour qu'aucun pic transitoire n'y échappe).
`coeff_speed` repassé à 0 (remplacé, pas cumulé). Base v12 (`coeff_spin=0.02`, `damping=0.05`),
seed=42. Seuil choisi : 6 m/s, nettement sous la zone où les échecs se concentraient (10-19 m/s).
 
**⚠️ N'a tourné que 500k steps, pas 1M** (même souci que v14) — comparaison valide contre `v11`
(500k, même recette sans `max_speed`), pas directement contre `v12`/`v15` (1M).
 
**Résultat** (30 épisodes) : **60% victoire, 37% mort, 3% retourné**, couverture 80,7% — contre
v11 à budget identique (43%/47%/10%/81,4%). Net progrès sur toute la ligne, et net mieux que v12
(1M steps !) alors que v16 n'a tourné que la moitié du budget. Le nombre de steps moyen jusqu'à
victoire monte (101 contre 64-65 avant) — attendu et accepté : `max_speed` ralentit délibérément
le drone, la vitesse n'est plus un objectif pour l'instant (choix explicite de l'utilisateur :
détection complète > rapidité).
 
**`--debug-angles` confirme que le plafond tient vraiment** : `speed` monte jusqu'à exactement
6,00 puis reste pinné dessus sur de longues séquences (jamais dépassé) — contrairement à v15 où
le coût de reward était contourné. Confirme qu'une contrainte physique dure est plus fiable qu'une
incitation, une deuxième fois après `damping`/`coeff_spin`.
 
**Observation qualitative (utilisateur)** : comportement visuellement "intelligent" pour la
première fois depuis le désaxage des couloirs — exploration qui a l'air délibérée, pas juste
rapide. Sous-échantillon debug (3 épisodes) : 67% victoire, 0% retourné.
 
**À refaire à 1M steps complets** pour confirmer que le gain tient à budget plein et permettre une
comparaison propre à `v15`/`v12` — pas encore fait.

## v17 — plafond de vitesse dynamique (`proximity_slowdown_dist`) — ÉCHEC

**Changement (un seul)** : plafond de vitesse non plus fixe (`max_speed=6.0` partout) mais réduit
progressivement quand un obstacle approche dans un cône aligné sur la direction de déplacement
réelle (pas le cap), jamais à 0 (`speed_cap_floor=1.5`) — toujours une contrainte physique dure,
pas une incitation de reward. Base v16 (`coeff_spin=0.02`, `damping=0.05`, `max_speed=6.0`),
`proximity_slowdown_dist=2.5`, `slowdown_min_dist=0.3`, seed=42, 1M steps demandés.

**⚠️ Run coupé à 750k steps** (crash ou interruption, cause non identifiée) — multiple de
`checkpoint_freq`, signe d'une sauvegarde périodique plutôt que finale.

**Bug découvert et corrigé** : `RotatingCheckpointCallback._on_step()` (`model_manager.py`)
n'a jamais passé `extra_meta` à `save_checkpoint()`, contrairement à la sauvegarde finale de
`train.py` — donc `env_kwargs` disparaît du `metadata.json` à chaque fois qu'un run s'arrête sur
une sauvegarde périodique plutôt que la sauvegarde finale. Conséquence : `evaluate.py` retombe sur
tous les défauts (dont `max_speed=0.0`, aucun plafond) → première éval de v17 invalide (`speed`
observé jusqu'à 9+ dans `--debug-angles`, alors que `max_speed=6.0` aurait dû l'interdire). Fix :
`_on_step()` passe maintenant `dict(self.extra_meta, base_timesteps=self._base_timesteps)` à
chaque sauvegarde périodique, pas seulement à la finale.

**Résultat, après correction manuelle du `metadata.json` et ré-évaluation** (30 épisodes, seed=42,
750k steps) : **60% victoire, 37% mort, 3% retourné, couverture 83,4%** — identique à v16 (500k
steps) au pourcentage près. Seule différence : steps moyens jusqu'à victoire **175** (contre **101**
sur v16) — quasiment le double. `--debug-angles` confirme que le mécanisme fonctionne (`speed`
redescend bien vers `speed_cap_floor` avant les morts, sur 15-18 steps d'anticipation) — mais ça ne
change rien au taux de survie, seulement une mort plus lente.

**Diagnostic** : retarder la vitesse avant l'impact n'aide pas si le drone n'a de toute façon jamais
appris à anticiper un virage — il continue sur sa trajectoire, juste plus lentement. Piste abandonnée.

## v18 — `death_penalty=500` (pénalité de collision/retournement x10) — SUCCÈS PARTIEL

**Diagnostic préalable** (script `check_reward_norm.py`, nouveau, lit `vecnormalize.pkl` directement
sans réentraîner) : sur v17, `ret_rms.std ≈ 297,5` — la pénalité de collision brute (-50) normalisée
tombe à **-0,17**, à peine plus qu'un step d'exploration normal (0,034) et très inférieure à ce
qu'un épisode entier rapporte en moyenne. Vérifié directement sur les évals v17 : reward moyen des
épisodes "mort" (~6,1) quasiment égal à celui des épisodes "victoire" (~7,2) — mourir à 72% de
couverture rapporte presque autant que gagner à 90%. Le modèle n'a presque aucune incitation nette
à éviter la mort une fois l'épisode bien avancé.

**Changement (un seul)** : `death_penalty` exposé en paramètre (était `-50.0` en dur dans le code),
monté à **500** (x10). Base identique à v16 (`coeff_spin=0.02`, `damping=0.05`, `max_speed=6.0`,
seed=42), from scratch, 500k steps.

**Résultat** (30 épisodes) : **67% victoire, 20% mort, 13% retourné**, couverture 81,9% — contre v16
(60%/37%/3%/80,7%). Net progrès sur le taux de mort par collision, mais **même effet de bord que
`coeff_align` (v8) et `coeff_proximity` (v13)** : `retourné` grimpe (3%→13%) — le modèle évite plus
fort, au prix de bascules plus extrêmes. Amélioration nette globalement, mais le symptôme de fond
(bascules non maîtrisées, manque d'anticipation) reste entier.

## v19 — architecture hiérarchique : contrôle en vitesse + cascade PID (remplace le contrôle en
couple direct) — RÉSULTAT MITIGÉ, ouvre le chantier v2.0 (ex-v20)

**Motivation** : sur v16/v18, roll/pitch restent tenus à -40/-60° en continu, y compris sur des
épisodes victorieux — aucune incitation dans le reward à revenir à plat une fois stabilisé, et le
seul moyen de se déplacer horizontalement (poussée purement verticale dans le repère du corps) est
d'incliner le drone. Hypothèse : le problème est structurel (l'agent pilote l'inclinaison
directement), pas réglable par du reward shaping seul. Confirmé par deep research externe (Gemini) :
la séparation stratégie RL (consigne de vitesse) / contrôle bas niveau classique (PID en cascade) est
le standard de l'industrie et de la littérature (PX4, Kaufmann et al./Scaramuzza — *Learning
High-Speed Flight in the Wild*), perçue comme plus mature qu'un contrôle bout-en-bout en contexte
portfolio.

**Changement (majeur)** : espace d'action passe de `[poussée, couple_roll, couple_pitch, couple_yaw]`
à `[vx_body_désiré, vy_body_désiré, vz_désiré, yaw_rate_désiré]` (repère du drone, comme
`frontier_vector`/`relief_rays`). Cascade PID interne (`_control_cascade`, appelée à chaque
sous-step physique) : boucle de vitesse (PI) → attitude cible → boucle de taux (PID) → couples
moteur, poussée dérivée de l'erreur de vitesse verticale. Reward et observations inchangés
volontairement pour isoler l'effet du changement d'action seul. Base reward = v18
(`death_penalty=500`, `damping=0.05`, `max_speed=6.0`, `coeff_spin=0.02`), from scratch, 500k steps.

**Résultat** (visuel + 1 épisode `--debug-angles` détaillé — les 30 épisodes complets abandonnés,
bien trop longs à observer/attendre) : **roll/pitch restent contrôlés sur tout l'épisode (jamais
au-delà de ±30°)**, contre les -40/-60° tenus en continu sur v16-v18 — **la cascade règle bien le
symptôme ciblé**. Mais : vitesse jamais montée au-delà de ~3 m/s (plafond `max_speed=6.0` jamais
approché), épisode observé 3-6x plus long (366 steps contre 100-200 avant) pour 55,4% de couverture,
yaw dérive lentement sans but apparent sur toute la durée, comportement perçu visuellement comme
"errant"/indécis plutôt que délibéré. Mort finale par collision : même mécanisme de fond que sur
v16/v18 (`minLid` s'effondre sans freinage anticipé), juste à une vitesse absolue plus faible.

**Deux lacunes trouvées par relecture du code** (aucun bug de signe dans la cascade elle-même,
formules de rotation body/world vérifiées cohérentes) :
- Boucle de vitesse (PI) recalculée à **chaque sous-step physique** (~500Hz) au lieu d'une fois par
  décision RL (~50Hz) comme le prescrit l'architecture cascade standard (confirmé par la recherche
  externe) — surcoût de calcul Python ~10x, explique probablement la lenteur d'évaluation.
- Poussée non compensée par l'inclinaison (perte de composante verticale en virage, `thrust_ctrl`
  jamais divisé par `up_z`).

**Diagnostic retenu** (après désaccord sur "juste plus de steps nécessaires" — tranché par
l'observation utilisateur que la courbe tensorboard plafonne dès ~400k steps habituellement sur ce
projet, pas 500k-1M) : **le reward et les observations ont été conçus pour l'ancien drone (contrôle
en couple), jamais réétudiés pour le nouveau (contrôle en vitesse)**. Incohérences identifiées :
1. `kinematics` observe `vel_lin` en repère **monde**, alors que l'action est maintenant en repère
   du **corps** — incohérent avec `frontier_vector`/`relief_rays`/`proximity_rays` déjà égocentriques,
   oblige le réseau à apprendre implicitement la rotation par yaw.
2. Clamp physique dur sur `qvel` (`max_speed` appliqué à chaque sous-step) devenu redondant avec le
   plafond déjà imposé par l'espace d'action, et en conflit avec le terme intégral du PID qui ne
   "voit" pas ce clamp (vitesse volée sans que le PID le sache).
3. `coeff_spin` conçu contre un spin qui émergeait d'un contrôle direct du couple — la vitesse
   angulaire est maintenant régulée en interne par la boucle de taux + `max_yaw_rate`, probablement
   obsolète.
4. `coeff_proximity` (abandonné en v13 à cause d'un bank agressif) redevient potentiellement viable :
   l'angle est maintenant plafonné par `max_tilt_angle_deg` et découle d'une commande de vitesse, pas
   d'un choix libre de couple — le mécanisme d'échec de v13 est structurellement bloqué.
5. Aucune mémoire de la dernière consigne dans l'observation — problématique avec un contrôle en
   vitesse où deux commandes incohérentes consécutives s'annulent physiquement (contrairement au
   contrôle en couple, où le bruit d'exploration PPO se moyennait sans ce problème).
6. Pas de lissage de l'action — le bruit d'exploration gaussien indépendant de PPO à chaque step
   fait osciller la consigne de vitesse d'un step à l'autre ; hypothèse retenue comme explication
   principale du comportement "errant" observé (le PID passe son temps à annuler le mouvement qu'il
   vient de créer plutôt qu'à avancer).

**Outillage ajouté** : `check_reward_norm.py` (nouveau script, lit `vecnormalize.pkl` et affiche
`ret_rms.std` + impact de la normalisation sur `death_penalty`/`r_exp`/un step typique, sans
réentraîner).

## Génération 2 — nouvelle numérotation : v2.0, v2.1, v2.2…

**Tout ce qui précède (v1 à v19) est désormais la « génération 1 »** (contrôle en couple direct, puis
première version de la cascade PID). La refonte ci-dessous change l'action, les observations, le
reward et le nombre d'environnements parallèles : elle n'est plus comparable en conditions
identiques aux versions précédentes. Elle s'appelle **v2.0** (anciennement « v20 »), les suivantes
seront v2.1, v2.2… Les comparaisons à v12/v16 restent indicatives seulement.

**Changements de protocole** :
- `--n-envs` passe de 8 à **16** (défaut de `train.py`) : rollout de 16 384 steps au lieu de 8 192,
  donc deux fois moins de mises à jour PPO à budget égal (500k ≈ 30 mises à jour). À surveiller ;
  `--n-steps 512` rétablirait la taille de rollout d'avant.
- **Garde-fou RAM** (`ram_guard.py`, `--ram-reserve-gb`, `--max-ram-gb`, `--no-ram-guard`) : `--n-envs`
  est réduit automatiquement si la RAM libre ne suffit pas (la machine de dev a 15,4 Go dont souvent
  moins de 4 Go libres) ; refus propre avec message si même 1 env ne tient pas ; pendant
  l'entraînement, arrêt propre + sauvegarde si la RAM libre passe sous la moitié de la réserve. Ctrl+C
  et erreurs mémoire (MemoryError, OOM CUDA, worker tué) sauvegardent aussi (`--resume` possible,
  `stopped_early` dans `metadata.json`), ferment les workers et libèrent la mémoire ; codes de
  sortie : 0 ok, 2 erreur mémoire/refus, 3 arrêt RAM, 130 Ctrl+C. Vérifié par scénarios simulés
  (0 processus orphelin) et par une courbe `ram/*` dans tensorboard : 2,65 -> 2,71 Go sur 40k steps,
  pas de fuite. Coût mesuré : ~70 Mo par worker ; le gros poste est le processus principal (torch)
  et le rollout buffer (le crop 3x64x64 pèse 48 Ko par step).
- **Recalibrage du garde-fou (après un refus abusif : « 1 env = 4,8 Go » avec 6,2 Go libres)** : la
  1re calibration utilisait la RAM *résidente* des processus, qui compte plusieurs fois les
  bibliothèques partagées entre workers (~1,5x trop haut). Recalé sur la baisse réelle de RAM
  disponible : CPU 8 envs ~3,3 Go, CUDA 10 envs ~6,5 Go (estimations maintenant 3,5 et 6,6 Go).
  `--ram-reserve-gb` passe à 1,0. **`--device auto` choisit maintenant le CPU** : mesuré 814 fps en
  CUDA (10 envs) contre 807 fps en CPU (8 envs), le goulot étant la physique des envs ; CUDA coûte
  ~2,6 Go de RAM en plus. `--device cuda` pour le forcer.
- `train.py` : les emojis/« ≈ » des messages faisaient planter le programme en console Windows
  cp1252 (y compris `--help`) : sortie reconfigurée en `errors="replace"`.
- Bug corrigé : `RotatingCheckpointCallback` comptait `base_timesteps` deux fois après `--resume`.
- Budget d'entraînement : **500k steps**, pas 1M. Les anciennes versions plafonnaient dès ~400k.

## v2.0 (ex-v20) — refonte groupée reward + observations pour le contrôle en vitesse (lissage d'action 0,3 = ÉCHEC ; sans lissage = 60% de victoires, cf. ablations)

**Changements groupés (assumé — pas à variable unique, cohérent avec la décision prise après
l'incident v11 : parfois plusieurs changements liés à une même cause doivent être testés ensemble)** :
- `kinematics` : `vel_lin` tourné en repère du corps avant observation.
- Clamp dur `qvel`/`max_speed` par sous-step **retiré**. À la place, la *consigne* horizontale est
  plafonnée en norme (`|v_cmd| <= max_speed`, pour que la diagonale ne dépasse pas 6 m/s : sans ça
  la boîte d'action `[-1,1]²` autorisait 8,5 m/s). Le PID voit tout, rien n'est volé en douce.
- `coeff_spin` désactivé (0.0, défaut) pour ce test.
- `coeff_proximity` réactivé (0.3, seuil 0.6, restreint aux pièces).
- Nouvelle observation `last_action` (dernière consigne lissée, 4 valeurs dans [-1,1]) — dans
  `observation_space`, `_get_obs()`, et `scalar_keys` de `ExplorerFeaturesExtractor`.
- Lissage de l'action (`action_smoothing_alpha=0.3`, EMA sur l'action brute PPO).
- Cascade scindée en `_outer_velocity_loop` (1x par action RL : consigne vitesse -> roll/pitch
  cibles + poussée de base + intégrateurs, `dt` = 10 sous-steps) et `_inner_attitude_rate_loop`
  (chaque sous-step : attitude -> taux -> couples).
- Poussée compensée par l'inclinaison (`thrust_base / max(up_z, 0.5)`, dans la boucle interne
  donc avec l'`up_z` instantané).

**Ajouts après audit du code (pas dans le plan initial)** :
- **Crop de grille égocentrique** (`ego_crop=True`, `--no-ego-crop` pour l'ancien comportement) :
  le crop était aligné sur le monde alors que l'action, `frontier_vector`, les rayons et maintenant
  la vitesse sont tous en repère du corps, et que le yaw absolu n'est pas observé : le CNN ne
  pouvait pas savoir où était "devant" sur la carte. Même incohérence que le point 1, sur la
  seule observation qui restait en repère monde. Haut de l'image = devant, gauche = gauche du drone.
- **`coeff_potential=30`** (`--coeff-potential`) : mesuré avec une politique heuristique "va vers la
  frontière" sur 6 bâtiments, le terme `phi - last_phi` vaut en moyenne |0,0025| par step contre
  8,1 pour `n_new_cells` (~3000x plus petit) — il ne guidait donc rien (cf. revue externe plus bas).
  À 30 : moyenne |0,076|, pics de -4,6 (la frontière la plus proche disparaît) à +2,4.
- **`RotatingCheckpointCallback` : `extra_meta` toujours pas passé** malgré ce que dit la section
  v17 (le fix n'était pas dans le code) — corrigé pour de bon : un run coupé sur checkpoint
  périodique garde `env_kwargs`, sinon `evaluate.py` retombe sur les défauts, ce qui est
  désormais bien plus grave (`action_smoothing_alpha`, `ego_crop`, `coeff_potential` changeraient).
- **`evaluate.py` : `--max-steps` marquait "victoire" un épisode simplement coupé** par la limite
  (l'env n'avait pas terminé). Corrigé : compté "timeout". Important pour `--max-steps 400`.
- Le contact est testé à chaque sous-step (sortie anticipée), plus seulement après les 10 : un
  rebond sur un mur ne peut plus effacer la collision.
- Nouveaux flags d'ablation, défauts inchangés : `--substeps` (10 = 50Hz), `--gamma` (0.99),
  `--action-smoothing-alpha`, `--no-ego-crop`.

**Vérifications faites** (`src/test_cascade.py`, contacts désactivés pour voler à travers les murs) :
hover 6 s sans dérive ni inclinaison ; avance à 3 m/s atteinte (3,10) avec inclinaison max 21,8° et
altitude constante (compensation d'`up_z` OK) ; freinage jusqu'à l'arrêt ; après un virage de 93°
la vitesse monde suit le cap et l'observation reste (vx>0, vy≈0) en repère corps ; latéral, montée,
lacet atteignent leur consigne ; diagonale pleine plafonnée à ~6 m/s ; crop égocentrique testé à 4
yaws (mur devant -> haut de l'image, mur à gauche -> colonnes de gauche). Mini-entraînement
(16k steps) -> checkpoint -> `evaluate.py` : OK de bout en bout.

**Points d'attention identifiés mais NON modifiés (à tester ensuite, un à la fois)** :
- Temps de réponse de la boucle de vitesse : 0,68 s (63%) avec `kp_vel=1.5`, soit ~34 steps RL. Les
  pièces font 4-7 m et on freine de 6 m/s en ~2,6 m minimum : un drone qui ne ralentit pas
  d'avance finit au mur. Leviers : `--kp-vel 3`, ou `--max-speed 3`.
- `gamma=0.99` à 50Hz = horizon effectif de 2 s (~6 m de vol) ; un trajet pièce->pièce dure plus.
  Leviers : `--gamma 0.995` ou `--substeps 25` (décision à 20Hz, 2,5x moins de steps pour la même
  durée simulée, donc aussi un entraînement plus rapide).
- Pas de pénalité de durée : rien ne distingue "errer" de "explorer" tant qu'aucune cellule n'est
  découverte, hormis le potentiel.

**Commande** :
```
python train.py --name explorer --damping 0.05 --death-penalty 500 --max-speed 6.0 --coeff-proximity 0.3 --proximity-threshold 0.6 --seed 42 --total-timesteps 500000
```
(`--n-envs` vaut 16 par défaut ; pas de `--coeff-spin` : retombe à 0.0 par défaut, voulu — point 3 ci-dessus)

**À comparer après coup** : contre v16 (60%/37%/3%/80,7%, 500k steps, même budget) et v12
(47%/50%/3%/82,9%, 1M steps), tous deux génération 1 (ancien contrôle en couple) — pour trancher si le contrôle en vitesse égale ou dépasse
l'ancien contrôle en couple à budget comparable, et si le comportement "errant" de v19 a disparu
(`--debug-angles` sur quelques épisodes, `--max-steps 400` pour éviter des traces trop longues).
 
### Résultat v2.0 — ÉCHEC (nettement pire que la génération 1)

Entraînement : 501 760 steps, **10 envs** (le garde-fou a réduit de 16 à 10 : 7,4 Go libres), CUDA,
RAM stable à ~7,25 Go sur toute la durée (pas de fuite), fin propre. Évaluation 30 épisodes
(seeds 9000-9029) : **3% victoire, 83% mort, 0% retourné, 13% timeout, couverture 64,0% (± 12,6)**,
trajectoire moyenne 26 m. Contre v16 (60/37/3/81) et v18 (67/20/13/82). **100% des morts en pièce**.

**Ce qui est réglé** : 0% de retournement, roll/pitch bornés à ~±25° (la cascade fait son travail).

**Courbes d'entraînement** : `coverage_mean` plate à 0,6-0,7 du début à la fin (aucune progression),
victoires ~0%, `timeout_rate` jusqu'à 87% vers 215k-300k puis les collisions reviennent (~50%) ;
`std` de la politique encore à 0,61 à la fin (exploration pas retombée), 49 mises à jour PPO au total.

**Mécanisme observé** (`--debug-angles`, seeds 9007 et 9019, morts en 91 et 110 steps, 2-3 m
parcourus) : le drone décrit une dérive lente et périodique (roll et pitch en sinusoïdes déphasées,
période ~60 steps = 1,2 s, soit une spirale en espace de vitesses), avec `alignement` ≈ -0,7 (il se
déplace à reculons / en crabe par rapport à son cap) à 2-2,7 m/s, et `minLid` passe de ~2 m à 0,14 m
en ~40 steps **sans aucun freinage**, alors que la frontière la plus proche est à ~0,2. Récompense ~0
pendant 90 steps puis -4 (mort). La politique ne réagit pas au mur ; le comportement ressemble à un
cycle limite plutôt qu'à de l'errance aléatoire.

**Hypothèses initiales** : cycle limite via `last_action`, budget trop court, horizon `gamma` trop
court, `kp_vel` trop faible. **Aucune n'était la bonne** — voir l'ablation ci-dessous.

### Ablations de v2.0 — le lissage d'action était le coupable

Le temps d'un entraînement complet n'est que de ~10 min (814 fps), donc au lieu de deviner :
5 runs identiques (500k steps, seed 42, 8 envs CPU, mêmes 30 épisodes d'évaluation), UN réglage
changé à chaque fois (scripts et logs supprimés depuis ; les résultats sont dans le tableau) :

| Run | Victoire | Mort | Timeout | Couverture |
|---|---|---|---|---|
| base v2.0 (lissage 0,3, potentiel 30, `last_action`, crop ego) | 7% | 63% | 30% | 65,8% |
| `--coeff-potential 1` | 7% | 83% | 10% | 65,2% |
| `--no-last-action` | 3% | 87% | 10% | 66,1% |
| **`--action-smoothing-alpha 1.0` (sans lissage)** | **60%** | **37%** | **3%** | **80,1%** |
| `--no-ego-crop` | 7% | 83% | 10% | 69,3% |

Toutes les variantes AVEC lissage restent à 3-7% de victoires ; la seule sans lissage retrouve
**60/37/0/3 (0% de retournement), couverture 80%, au niveau de v16** (60/37/3/81) avec le nouveau
contrôle en vitesse. Un seul seed par config, mais 60% contre 3-7% sur quatre configs est bien
au-delà du bruit.

**Pourquoi (hypothèse, non prouvée)** : avec l'EMA, l'action que la politique choisit ne pèse que
30% sur la consigne réellement appliquée, et la boucle de vitesse est déjà un passe-bas (0,68 s) :
le signal action -> récompense est dilué et retardé, et le gradient PPO n'a presque rien à
attribuer. La trace de la politique entraînée (actions ~bruit, moyenne ~0, rotation de lacet
constante) correspond à une politique qui n'a rien appris, pas à un contrôle défaillant. Le
raisonnement d'origine (« le PID annule le mouvement que le bruit PPO vient de créer ») était faux :
le plant lisse déjà tout seul. **Leçon** : un lissage d'action entre la politique et l'env n'est pas
gratuit pour PPO ; ne pas en mettre sans l'avoir testé seul.

**Décision** : `action_smoothing_alpha` passe à **1.0 par défaut** (train.py et env). (Les modèles d'ablation et `explorer/v2` ont été supprimés.) Les autres ajouts de v2.0 (potentiel 30, crop égocentrique, `last_action`,
proximité 0,3, kinematics en repère corps) sont dans ce run à 60% mais **leur utilité propre n'est
pas démontrée** : les ablations ci-dessus étaient confondues avec le lissage. À refaire autour de
`abl_nosmooth` si on veut savoir lesquels aident.

### Batterie 2 — autour du run SANS lissage (350k steps, seed 42, 8 envs CPU, 30 épisodes)

Un réglage change à chaque fois par rapport au témoin (scripts et logs supprimés depuis). Budget raccourci à 350k : les runs précédents plafonnaient déjà.

| Run | Ce qui change | Victoire | Mort | Timeout | Couverture |
|---|---|---|---|---|---|
| **control** | rien (sans lissage, potentiel 30, `last_action`, crop ego, proximité 0,3) | **70%** | 27% | 3% | 82,7% |
| potential1 | `--coeff-potential 1` | 53% | 47% | 0% | 85,0% |
| **nolast** | `--no-last-action` | **3%** | 57% | 40% | 63,9% |
| **noegocrop** | `--no-ego-crop` | **3%** | 63% | 33% | 66,0% |
| **noproximity** | `--coeff-proximity 0` | **77%** | 23% | 0% | 86,3% |
| speed3 | `--max-speed 3` | 3% | 83% | 13% | 57,0% |

Le témoin à 350k (70%) fait déjà mieux que le run sans lissage à 500k (60%) : le budget n'était pas le
problème, et le bruit d'un run à l'autre est de l'ordre de ±10 points (un seed, 30 épisodes).

**Lecture (prudente, UN seed par config)** :
- **Entrées `last_action` et crop égocentrique : semblent indispensables** (70% -> 3% en les
  retirant, avec ~40% de timeouts : la politique tourne en rond au lieu d'explorer). Cohérent avec
  la physique : un crop aligné sur le monde est ininterprétable sans le yaw, et avec un contrôle en
  vitesse la politique a besoin de savoir ce qu'elle poursuit déjà.
- **Pénalité de proximité : aucune preuve qu'elle aide** (77% sans, 70% avec : dans le bruit). Elle a
  déjà fait du mal en v13. Retirée de la commande recommandée.
- **Potentiel 30 vs 1 : pas de différence claire** (70% vs 53%, dans le bruit) ; laissé à 30.
- **`max_speed=3` : très mauvais (3%)**, contre-intuitif (plus lent = plus dangereux ?). Les runs à 3%
  forment peut-être un mode d'échec « la politique ne démarre pas » plutôt qu'un vrai effet ; à
  confirmer avec un 2e seed avant d'en tirer une conclusion sur `max_speed`.
- **Limite importante** : les trois runs à 3% (nolast, noegocrop, speed3) peuvent aussi être de la
  malchance d'init (distribution bimodale ?). Un 2e seed (43) sur control / nolast / noegocrop
  trancherait. Non fait.

**Config recommandée (v2.1)** : défauts actuels (sans lissage) SANS `--coeff-proximity`, 350k steps :
```
python train.py --name explorer --damping 0.05 --death-penalty 500 --max-speed 6.0 --seed 42 --total-timesteps 350000
```
(Le modèle `abl2_noproximity` à 77% a été supprimé avec les autres ablations.)

**Pistes non testées** : bloquer l'altitude (la tâche est plane, le LiDAR aussi, et `vz` ne sert qu'à
mourir au sol/plafond) ; 2e seed ; `--substeps 25`.

### Batterie 3 et verdict : le setup est fragile, on reprend par étapes

Config « recommandée v2.1 » lancée par l'utilisateur (`explorer/v3`, 14 envs, 500k) : **3% victoire, 53% mort,
43% timeout, couverture 65%** — l'échec, avec la même signature que les « runs ratés » des ablations
(timeouts 40-60% à partir de ~200k steps, couverture plate). Bilan sur 7 runs sans lissage : 3 réussissent
(53/70/77%), 4 échouent (~3%). **La conclusion « last_action / crop ego indispensables » de la batterie 2
est donc à jeter** : ces runs à 3% sont probablement de la malchance, la config normale échoue aussi.
Un essai `--time-penalty 0.1` (seed 43) : 0% victoire, 70% mort, 30% timeout -> pas de correction ;
interrompu à la demande (flag `--time-penalty` laissé dans le code, défaut 0).

**Fait clé mesuré** : couverture AU SPAWN (le LiDAR voit toute la 1re pièce + un bout de couloir) =
**95% à 1 pièce, 59% à 2, 40% à 3, 30% à 4**. Les runs à ~64% de couverture n'ont donc quasiment rien
exploré au-delà du départ : **le problème tout entier est de franchir les portes** (couloirs 1,6 m,
désaxés). Tout le reste (récompenses annexes, lissage...) n'est qu'un détail à côté.

## PLAN DE REPRISE — retrouver d'où vient le problème, étape par étape

**Règles** : (1) on part du plus simple possible et on n'ajoute RIEN sans preuve (le lissage « évident »
a tout cassé) ; (2) chaque étape a un critère de passage chiffré, sinon on ne passe pas à la suivante ;
(3) un run isolé ne prouve rien (même config = réussite ou échec) -> on juge sur le signal à 100-150k
steps (`episodes/coverage_mean` doit dépasser nettement la couverture au spawn, `timeout_rate` ~0)
et on ne « garde » un élément qu'avec 2 seeds ; (4) un seul changement par test.

**Étape 0 — Outillage** : `test_env.py --visual` réparé (restes de l'ancien projet) ; un script de sonde
qui joue une politique scriptée pour vérifier que l'env est franchissable ; log du % de temps passé dans
la 1re pièce (le vrai indicateur d'échec).

**Étape 1 — Audit des entrées, une par une** (garder / retirer, avec la raison) :
| Entrée | Question | Avis a priori |
|---|---|---|
| `proximity_rays` (16, ego) | seule info directe « mur à X m dans cette direction » | GARDER |
| `frontier_vector` (5x4) | la frontière « la plus proche » est en ligne droite et peut être DERRIÈRE un mur (vue au pilote scripté : cibles à 3-4 m à travers le mur, porte ailleurs) ; 1-2 frontières suffisent ? distance par chemin (BFS) ? | À TESTER (suspect n°1) |
| `kinematics` (10) | roll/pitch/up_z/altitude : le PID stabilise déjà, le réseau n'en a pas besoin | GARDER vitesse corps + vitesse de lacet, RETIRER le reste |
| `relief_rays` (18) | rayons verticaux, tâche plane | RETIRER |
| `coverage` (1) | scalaire global, ne dit pas où aller | RETIRER |
| `local_crop` (CNN 3x64x64) | mémoire spatiale, mais plus gros et plus risqué ; sans lui le réseau est un petit MLP | RETIRER d'abord, réintroduire à l'étape 5 |
| `last_action` | utilité non démontrée | tester à l'étape 5 |

**Étape 2 — Sorties** : 3 sorties (vx, vy, vitesse de lacet), altitude bloquée : `vz` ne sert qu'à mourir au
sol/plafond (LiDAR à plat). `max_speed` revu à la lumière des pièces de 4-7 m.

**Étape 3 — Récompenses minimales, uniquement celles dont l'utilité est certaine** : (a) cellules
nouvellement découvertes (c'est l'objectif lui-même), (b) pénalité de mort, (c) bonus de victoire.
Rien d'autre : ni potentiel, ni proximité, ni spin, ni alignement, ni vitesse, ni temps.

**Étape 4 — Échelle de difficulté** (le test le plus informatif ; il n'existe pas de palier « 1 pièce », la
couverture y est de 95% au spawn) : (a) 2 pièces exactement, (b) 2 à 3, (c) 2 à 4. Critère : >= 70%
de victoire à 150-200k steps. Un palier qui échoue localise le problème. Demande de petites modifs :
flags `--n-rooms-min/max` (aujourd'hui `(2, 4)` en dur dans `train.py`).

**Étape 5 — Réintroduire UN élément à la fois**, dans cet ordre : `last_action`, crop CNN ego, potentiel de
frontière, puis le reste. Chaque élément : critère de gain mesuré, 2 seeds, sinon retiré pour de bon.

**Étape 6 — Si l'étape 4a échoue encore** (hypothèses ciblées, après vérification) : frontière à travers
les murs ; hyperparamètres PPO (seulement ~30-50 mises à jour PPO dans un run, `n_steps`, `lr`,
`ent_coef`, `gamma`) ; normalisation du reward (clip à 10 : une découverte de 180 cellules et une mort
à -500 sont toutes deux écrasées au même niveau tant que l'écart-type est petit) ; taille du réseau.

### Reprise par étapes — ce qu'on a trouvé (entrées minimales, 3 pièces)

**Mise en place** : entrées minimales (`proximity_rays`, `frontier_vector` x2, `velocity`), altitude bloquée,
3 pièces fixes, récompense = cellules découvertes + mort + victoire. Flags ajoutés : `--obs-keys`,
`--k-frontiers`, `--free-altitude`, `--n-rooms-min/max`, `--no-yaw`, `--coeff-progress`, `--time-penalty`.
`--device auto` = CPU, garde-fou RAM recalibré (cf. plus haut).

**1. Run minimal `min3rooms` (mort -500, 150k steps)** : 0 victoire, couverture ~0,6-0,7 plate,
timeouts 17-43%. Sonde des actions : commandes faibles (|a| moyen 0,2 = ~1 m/s), jamais saturées,
un changement de signe tous les ~33 steps (balancement ~1,3 s), cap qui dérive de -39 à +138 deg.
**2. Pilote scripté** n'utilisant QUE ces entrées (frontière -> direction, ralentit près des murs) :
27-40% de victoires sur 30 bâtiments à 3 pièces -> **l'information minimale suffit** ; c'est
l'apprentissage qui échoue. (Au spawn, la frontière la plus proche n'est derrière un mur que dans 3%
des cas : l'hypothèse « frontière à travers les murs » est écartée.)
**3. Cause trouvée : l'échelle des récompenses.** `ret_rms.std` ~ 145 : après normalisation, la mort
(-500) vaut **-3,4** pour le réseau, une progression d'un step vers une porte +0,004 à +0,03, 8 cellules
+0,05. Un trajet parfait de 100 steps rapporte moins qu'une mort -> rester sur place est optimal. C'est
le « il reste dans la pièce / il oscille en esquivant les murs » depuis le début, et la pénalité de mort
de 500 traînait dans toutes les commandes sans que son poids réel ait été mesuré.

**4. Résultats à 150k steps (30 épisodes, 3 pièces, entrées minimales, sans lacet)** :
| Run | Victoire | Mort | Timeout | Couverture |
|---|---|---|---|---|
| mort -500 + but dense | 7% | 77% | 17% | 68% |
| **mort -50 + but dense (`--coeff-progress 10`)** | **17%** | 83% | 0% | **79%** |
| mort -20 + but dense | 10% | 90% | 0% | 70% |
| mort -50, sans but dense | 0% | 100% | 0% | 71% |
Mort moins pénalisée = plus de timeouts (il avance) ; but dense = c'est lui qui fait progresser (79%
contre 71%, 17% contre 0% de victoire). Morts à 92% en pièce.
**5. Prolongé à 360k steps (150k + 200k via `--resume`), mort -50 + but dense + sans lacet** :
3 pièces : **60% victoire / 40% mort / 0% timeout, couverture 86,3%**, 182 steps pour gagner. Courbe
d'apprentissage RÉGULIÈRE : victoires 0 -> 31% -> 65% et collisions 100% -> 69% -> 35% par tiers
d'entraînement, pas d'effondrement. **Généralisation, évalué sur 2 à 4 pièces (la vraie tâche) :
63% victoire / 37% mort / 0% timeout, couverture 85,2%**, avec seulement 3 entrées (rayons de proximité,
2 frontières, vitesse), AUCUN CNN, 2 sorties (vx, vy), récompense = cellules + progression + mort + victoire.
**Un seul run (seed 42)** : à confirmer avec un 2e seed avant d'en faire une règle.
Modèle conservé : `src/models/goal2_d50/v1` (les autres runs de cette série ont été supprimés).

Commande équivalente pour le refaire d'un coup (défauts : entrées minimales, 3 pièces, altitude bloquée) :
```
python train.py --name explorer --damping 0.05 --death-penalty 50 --coeff-progress 10 --no-yaw --max-speed 6.0 --seed 42 --total-timesteps 350000 --device cpu --n-envs 8
python evaluate.py --name explorer --n-episodes 30 --n-rooms-min 2 --n-rooms-max 4
```

**Correctif `--no-yaw` (signalé à l'oeil par l'utilisateur : « le lacet dérive naturellement vers la droite »)** :
reproduit, cap jusqu'à **-18 deg** (extrêmes -18/+7) pendant des manoeuvres. Cause : la cascade ne régule que
la VITESSE de lacet, aucune boucle sur l'ANGLE. Ajout d'un maintien de cap (P sur l'angle, `yaw_hold_kp=4`) :
extrêmes **-4/+3 deg**. NB : `goal2_d50` a été entraîné AVANT ce correctif (avec la dérive) ; ses résultats
(60%/63%) ne sont donc pas strictement ceux de la dynamique actuelle.
**Mesures de freinage** (contacts désactivés, `kp_vel=1.5`, tilt 35 deg) : arrêt depuis 3 m/s en 1,8 m / 1,1 s,
depuis 6 m/s en 4,0 m / 1,2 s ; `kp_vel=3` : 1,3 m et 3,8 m ; tilt 55 deg + `kp_vel=3` : 2,7 m depuis 6 m/s mais
dépassement de vitesse ~1 m/s. Les morts de `goal2_d50` ont lieu à ~3,7 m/s en moyenne, surtout dans la 2e pièce
(8/12). Levier le plus efficace attendu : `--max-speed` (distance d'arrêt en v au carré). Non testé.
**CNN ajouté seul (`cnn1`, 350k)** : 10% victoire / couverture 69% (3 pièces), 17% / 72% (2-4 pièces), contre
60% / 86% sans CNN ; plafonne à 18% de victoires dès le 3e quart. Un seul run.

**Témoin avec cap corrigé (`ctrl6`, config de `goal2_d50`, `--max-speed 6`, 350k, seed 42)** : **83% victoire / 17% mort
(3 pièces), 80% / 20% (2-4 pièces), couverture 86% / 85%**, ~195 steps pour gagner. Contre 60% / 63% pour `goal2_d50`
(sans maintien de cap) : le défaut de dérive de cap pesait environ 20 points. Courbe régulière : victoires
0 -> 8 -> 43 -> 62% par quart. Un seul run.
**`--max-speed 3` (`speed3`, même config)** : **0% victoire / 100% mort** sur 3 pièces comme sur 2-4, couverture 64%,
collisions à 98% pendant TOUT l'entraînement (aucun apprentissage de l'évitement) alors que `explained_variance`
atteint 0,91. Contre-intuitif (plus lent devrait aider) et déjà vu une fois (3% avec l'ancienne échelle de
récompense). **Cause non identifiée.** Trace : le drone fonce droit vers un mur à ~2,5 m/s, `minLid` passe de 2,4 m
à 0,2 m en ~90 steps sans freiner. Conclusion pratique : ne pas réduire `max_speed` ; `ctrl6` (6 m/s) est la
référence. Les autres leviers de freinage (`--kp-vel 3`, `--max-tilt-angle-deg 45`) restent non testés.

**Lecture** : les deux comptent. À vérifier ensuite : budget plus long, `max_speed`, puis réintroduire
`last_action` / crop / potentiel UN par un avec le critère ci-dessus (étape 5 du plan).

### Réactivité (`agile`) et 2e seed — le point faible : la ROBUSTESSE entre seeds

Config commune : entrées minimales (proximity_rays, frontier_vector x2, velocity), 2 sorties (vx, vy) + maintien de cap, altitude
bloquée, mort -50, `--coeff-progress 10`, 350k steps, 8 envs CPU. Évaluation : 30 épisodes, 3 pièces puis 2-4 pièces.
| Run | seed | 3 pièces | 2-4 pièces |
|---|---|---|---|
| `ctrl6` (6 m/s) | 42 | 83% victoire / 86% couv. | 80% / 85% |
| `ctrl6_s2` (identique) | **43** | **7%** / 70% | **10%** / 68% |
| `agile` (`--kp-vel 3 --max-tilt-angle-deg 45`) | 42 | 77% / 86,5% | 77% / 85,8% |
Mesure de réactivité (+v -> -v) : 2,0 s -> 1,4 s à 6 m/s. Visuellement (utilisateur) : `agile` est « trop abusé », rapide,
peut-être trop, mais pour des pompiers aller vite est voulu ; peut-être plus de temps d'apprentissage nécessaire.
**Fait important : à config identique, le seed 42 réussit (83%) et le seed 43 échoue (7%).** Les deux runs qui réussissent
décollent vers 120-200k steps ; le seed 43 reste à ~0% de victoires jusqu'au bout (collisions 97%) avec la même baisse d'entropie.
Les 80% ne sont donc PAS fiables tels quels : tout chiffre publié devra venir de plusieurs seeds. (`agile` n'a qu'un seed.)

**Vérification de reproductibilité (suite à un doute légitime de l'utilisateur : « tu as changé quelque chose ? »)** :
1. Réglages enregistrés de `ctrl6` et `ctrl6_s2` : identiques, seuls le seed, le nom et l'heure diffèrent ; code inchangé depuis
   23:44 (avant les 3 entraînements), identique au commit `f16c9de`.
2. **Ré-évaluation du fichier `ctrl6` d'origine avec le code du lendemain** : 83% / 86,1% (3 pièces) et 80% / 85,2% (2-4 pièces),
   IDENTIQUE à la décimale -> le comportement du vol/contrôleur/vitesses n'a pas changé.
3. **Entraînement `rep42`** (seed 42, 8 envs, 352 256 steps, aucun arrêt anticipé) : **83% / 86,1%**, identique à `ctrl6` ; la
   courbe d'entraînement se superpose à la virgule près (constat de l'utilisateur). -> **l'entraînement est déterministe à seed égal.**
4. Conclusion : le seed 43 (`ctrl6_s2`, 7%) est un VRAI effet du seed, pas un défaut de code ni de reproduction. Sur 2 seeds valides
   on a 1 réussite et 1 échec ; le taux de réussite réel reste inconnu.
**Incident à ne pas répéter** : un premier « seed 42 refait » était invalide : mes mesures lancées en parallèle ont fait baisser la
RAM, le garde-fou a réduit à 5 envs puis arrêté l'entraînement à 236 750 steps (code de sortie 3). Règle : AUCUNE autre tâche pendant un
entraînement/test de reproductibilité, et vérifier `n_envs` et `steps` dans `metadata.json` avant de lire un résultat.
**Vitesses mesurées** (12 bâtiments, 3 pièces) : vitesse moyenne `goal2_d50` 2,3 m/s, `ctrl6` 3,1, `ctrl6_s2` 2,5, `agile` 3,6 ; part du
temps à plus de 4 m/s : 13% / 34% / 10% / 50%. Réglages `max_speed`, `kp_vel`, inclinaison identiques entre `goal2_d50`, `ctrl6`, `ctrl6_s2`.

### Protocole léger à 2 seeds (42 = connu bon, 43 = connu mauvais) et interface de commande

Constat de l'utilisateur (à l'oeil) : trop lent, trop prudent près des portes, à-coups. Mesuré : victoires en ~101 steps en génération 1
(couples directs) contre ~195 pour `ctrl6`. Recherche web : les politiques à CONSIGNE DE VITESSE donnent un vol « quasi stationnaire »,
alors que poussée + vitesses de rotation (CTBR) ou commandes d'attitude permettent des manoeuvres bien plus agressives (benchmark
arXiv 2202.10796) ; PX4 fait sortir du contrôleur de vitesse une ACCÉLÉRATION, convertie en inclinaison.
Essais (3 pièces, 30 épisodes, 350k steps, 8 envs, mort -50, progression 10, cap maintenu) :
| Run | seed | victoire | couv. | steps pour gagner | vitesse moy. | cmd saturées |
|---|---|---|---|---|---|---|
| `ctrl6` (vitesse, 35 deg) | 42 | 83% | 86% | ~199 | 3,1 m/s | 24% |
| `ctrl6_s2` | 43 | 7% | 70% | - | 2,5 | 14% |
| inclinaison 25 deg | 42 | 70% | 87% | - | 2,2 | 8% |
| inclinaison 25 deg | 43 | 7% | 71% | - | 2,5 | 29% |
| `--max-speed 4.5` | 42 | 17% | 70% | - | 1,8 | 12% |
| **`--action-mode accel`** | 42 | **97%** | **90,1%** | 230 | 2,6 | 13% |
| `--action-mode accel` | 43 | 13% | 77% | 283 | - | - |
Lecture : (1) réduire vitesse ou inclinaison adoucit mais ralentit, et `max_speed` plus bas échoue (3e fois) ; (2) le mode accélération
(`_outer_accel_loop` : a = a_max * commande - c * vitesse, c = a_max/max_speed) donne le meilleur taux jamais vu (97%) sur le seed 42 MAIS
ne règle pas le seed 43 (13%) ni la lenteur (230 steps) : le drone vole calmement à ~2,6 m/s alors que 6 m/s sont permis. (3) le seed 43
échoue avec TOUTES les variantes (7%, 7%, 13%) : son échec est lié à l'initialisation, pas à l'interface de commande.
Mesures du mode accélération (contacts désactivés) : freinage à fond depuis 5,6 m/s en 3,5 m (4,0 m en mode vitesse) ; inversion +v -> -v
non plus rapide (2,4 s contre 2,0 s) à cause du frottement virtuel.
Pistes pour la vitesse (non testées) : `--time-penalty` (0,05 à 0,1 à l'échelle actuelle, mort -50), `--coeff-progress` plus élevé,
frottement virtuel plus faible.

### Récompense minimale + départ à l'extérieur (4 octobre) — décisions de l'utilisateur appliquées

Décisions : suppression de la récompense de progression (et du code A*/BFS, des modes de progression) ; `coeff_potential` = 0 ; CNN +
`last_action` à retester ; lacet plus tard ; **départ à l'extérieur** (porche fermé de 3,5-5 m devant une porte d'entrée dans le mur sud de la
1re pièce ; `start_outside`, défaut de `train.py`, `--start-inside` pour l'ancien) : couverture au spawn 11 % (au lieu de 39 %), pièces et
couloirs identiques à avant, vol scripté extérieur -> intérieur sans collision 30/30. Non-régression après le ménage : `ctrl6` 83 %/86,1 %
et `accel_s42` 97 %/90,1 % inchangés. Les anciens `metadata.json` (avec `coeff_progress`, `progress_mode`) restent chargeables (réglages ignorés).
Résultats (3 pièces, départ extérieur, 350k, seed 42, mode vitesse 35 deg, mort -50, aucun but dense) :
| Run | victoire | mort | couverture (réelle, départ ~11 %) | vitesse moy. |
|---|---|---|---|---|
| `out_min_s42` (27 entrées) | 3 % | 97 % | 62 % | 2,4 m/s |
| `out_full_s42` (+ CNN + `last_action`) | 3 % | 97 % | 58 % | 2,4 m/s |
Sans signal de but, ça n'atteint pas 90 % mais la couverture réelle atteint ~60 % (soit ~50 points gagnés depuis le départ) : la récompense
« cellules + mort » seule apprend à explorer, pas à finir. Le CNN + `last_action` n'améliore rien à ce budget.
**Cycle limite mesuré** (`out_min_s42`, seed 9000, aussi vu à l'oeil par l'utilisateur : « il part à gauche, à droite, à l'infini devant la
3e porte ») : sur 200 steps il parcourt 8,7 m mais ne se déplace que de 0,6 m ; balancement de ±1,5 m en x avec période ~4 s sur la commande
avant/arrière et ~1,3 s sur le côté.
**Physique modifiée SANS réentraîner (même modèle `out_min_s42`)** : référence 3 % de victoires ; `drag=0,3` : 20 % (couv. 71 %) ;
`drag=0,6` : 7 % ; `kp_vel=3` : 3 % ; `ki_vel=0` (sans intégrale) : 17 % (couv. 64 %). Le frottement de l'air (paramètre `drag`, force =
-masse*drag*vitesse, 0 par défaut) et l'intégrale du contrôleur de vitesse (dépassement par accumulation) pèsent donc sur le comportement,
sans que ce soit un correctif (30 épisodes, un seul modèle, politique non entraînée pour ces physiques). À confirmer en RÉENTRAÎNANT.
Comparaison honnête à faire : réintroduire l'ancien contrôle direct (couples) comme `action_mode` dans le MÊME environnement pour trancher
objectivement « le PID est-il en cause ? » (l'utilisateur est convaincu que oui, non démontré).

### DÉCOUVERTE (4 octobre) : le contrôleur interne était mal réglé — l'amortissement `--damping 0.05` de toutes nos commandes

Suite à un conseil externe (régler UNE boucle à la fois, de l'intérieur vers l'extérieur, sans RL, en regardant des COURBES) :
`src/step_response.py` trace les réponses indicielles de la boucle de taux, d'attitude et de vitesse isolées (`diagnostics/step_response.png`).
Avec les réglages de TOUS nos runs (`--damping 0.05`, `kp_rate 0.15`, `kp_att 6`, `kp_vel 1.5`) :
- **boucle de taux : n'atteint jamais la consigne** (plafonne à ~60 % puis rattrape très lentement) ;
- **boucle d'attitude : 414 ms de montée, 1,19 s d'établissement** pour 10 deg (un vrai drone : 100-200 ms) ;
- boucle de vitesse : propre mais lente (~1,5 s pour 6 m/s).
Le contrôleur n'oscille donc pas : il est SLUGGISH et sur-amorti (la « glace » de l'utilisateur). Cause : `joint_damping` (`--damping 0.05`, hérité de
la génération 1 où il servait à calmer la toupie en contrôle de couples) freine TOUTE rotation, du même ordre que le gain de taux (0,15*gear 0,5 =
0,075) : erreur permanente de 40 %. Avec `--damping 0` : taux 86 ms, attitude 252 ms ; avec en plus `--kp-rate 0.4 --kp-att 20 --kp-vel 4` :
**taux 36 ms (0,8 % de dépassement), attitude 76 ms (1,1 %), vitesse 380-720 ms selon la consigne** (limitée par l'inclinaison max : 6 m/s
demande > 0,87 s quoi qu'on fasse). Il n'y avait que des réglages à changer (flags existants), aucun nouveau code.
**Run de confirmation** (même config que `out_min_s42` : entrées minimales, départ extérieur, cellules + mort -50 + victoire, AUCUN signal de but ;
seul le contrôleur change : `--damping 0 --kp-rate 0.4 --kp-att 20 --kp-vel 4`), 350k steps, 3 pièces, 30 épisodes :
| Run | seed | victoire | mort | couverture | épisodes « coincés » | vitesse moy. | steps/victoire |
|---|---|---|---|---|---|---|---|
| ancien réglage (`out_min_s42`) | 42 | 3 % | 97 % | 62 % | - | 2,4 m/s | - |
| **`plant_s42`** | 42 | **97 %** | 3 % | 89 % | 0/30 | **4,2 m/s** | 206 |
| **`plant_s43`** | **43** | **87 %** | 13 % | 87 % | 0/30 | 3,5 m/s | 261 |
Le seed 43, qui échouait avec TOUTES les configurations précédentes (7 %, 7 %, 13 %, 3 %), réussit : la « loterie des seeds » venait aussi du
contrôleur trop lent. Le va-et-vient devant les portes a disparu (0/30), le drone vole nettement plus vite (steps/victoire mesurés en PARTANT
DE L'EXTÉRIEUR, couverture de départ 11 % : non comparables aux anciens 195-230 steps qui partaient de ~39 %). Sans récompense de progression,
sans carte, avec 27 entrées. Valeurs par défaut de `train.py` mises à jour (`kp_rate 0.4`, `kp_att 20`, `kp_vel 4`, `damping 0`) ; les anciens
modèles gardent leurs valeurs (écrites dans leur `metadata.json`). Limite : 2 seeds, 30 épisodes chacun, 3 pièces alignées.

**Généralisation des modèles à contrôleur réglé** (2 à 4 pièces, départ extérieur, bâtiments JAMAIS utilisés pour les choisir : seeds 9100-9129) :
`plant_s42` **87 %** victoire / 13 % mort / couverture 85,8 % / 204 steps ; `plant_s43` **77 %** / 23 % / 83,5 % / 274 steps. Un peu en dessous des
97 % / 87 % sur les 3 pièces des seeds 9000-9029 (la plage 2-4 contient des bâtiments à 4 pièces plus durs, et d'autres bâtiments), mais pas
d'effondrement : pas de sur-apprentissage manifeste du jeu de test.
**Retour visuel de l'utilisateur (4 oct.)** : « vraiment mieux à fond, trop cool enfin » ; le mouvement est « abusé » (trop de force) -> piste :
inclinaison max plus faible (28 deg, 25 deg) avec le contrôleur désormais réglé, à tester sur les seeds 42 et 43.
**Ménage du 4 oct.** : modèles conservés `plant_s42`, `plant_s43` (référence actuelle), `ctrl6` et `accel_s42` (ancrages de NON-RÉGRESSION du code :
leurs résultats 83 %/86,1 % et 97 %/90,1 % doivent rester identiques à la décimale), `explorer/` (v1, v3 de l'utilisateur). Supprimés (résultats
consignés plus haut) : accel_s43, agile, cnn1, ctrl6_s2, goal2_d50, out_full_s42, out_full_s43, out_min_s42, rep42, spd45_s42, spd45_s43,
tilt25_s42, tilt25_s43, leurs courbes tensorboard, et `src/eval_logs/`.

## FEUILLE DE ROUTE (v2) — objectif : une vidéo qui marche, un drone qui visite un bâtiment et produit une carte pour les pompiers

**Objectif de fin** : vidéo d'un drone devant un bâtiment qui le visite seul et construit une carte utilisable ; projet fiable
et présentable (CV). **Principe** : on ne casse jamais ce qui marche : `ctrl6` (83% / 80%) est figé comme modèle de secours,
chaque étape part d'une copie et ne devient la référence que si elle fait au moins aussi bien, sur 2 seeds au minimum.

| # | Étape | Pourquoi | Taille | Critère de passage |
|---|---|---|---|---|
| 0 | Verrouiller la base : `ctrl6` en 2e seed, commit | un seul run ne prouve rien ici | S (en fond) | >= 70% sur les deux seeds |
| 1 | Réactivité : `--kp-vel 3` + `--max-tilt-angle-deg 45` | passer de +6 à -6 m/s : 2,0 s -> 1,4 s (mesuré) | S | >= `ctrl6`, mouvement plus vif à l'oeil |
| 2 | Lacet : décision (cf. plus bas) | ne sert à rien pour un LiDAR 360 deg ; sert pour un capteur directionnel et pour la vidéo | S/M | pas pire que le cap fixe |
| 3 | Cibles de réussite : 95% puis 98% de couverture | on est à ~85% de moyenne pour 90% demandé ; plafond à mesurer proprement | S | taux de victoire maintenu |
| 4 | Cartes plus dures : générateur v2 | couloir central avec salles à gauche ET à droite, plusieurs sorties, impasses, boucles, salles plus grandes/longues | L | paliers : T/H simple -> grandes salles -> boucles |
| 5 | Mémoire : réintroduire CNN (petite carte locale), `coverage`, `last_action`, un par un | avec des branches il faut se souvenir des salles faites ; le CNN n'a pas marché (10-17%) sur la carte actuelle, à retester sur les cartes à branches | M | un gain mesuré, sinon retiré |
| 6 | Carte de précision (pompiers) | couche SÉPARÉE du RL : carte haute résolution construite pendant le vol (mêmes tirs, plus denses), métrique de qualité vs plan réel, export d'un plan propre | M | précision/rappel des murs ; image nette |
| 7 | Vidéo + README/CV | rendu 3e personne + carte qui se construit à côté, tableau de résultats multi-seeds, limites assumées | M | vidéo reproductible par une commande |
| ! | Vidéo de SECOURS dès l'étape 0/1 | avoir quelque chose qui marche tout de suite avec `ctrl6` sur les cartes actuelles | S | un mp4 propre |

**À propos du lacet (décision à prendre)** : pour la navigation seule avec un LiDAR à 360 deg il est inutile (le cap fixe a donné
80%). Il redevient utile (a) avec un capteur DIRECTIONNEL (le LiDAR « de précision », une caméra) et (b) pour la VIDÉO : un drone qui
vole en crabe est étrange, face à sa direction c'est naturel. Proposition : pas une sortie du réseau (il reste à 2 sorties) ;
le contrôleur bas niveau fait tourner le cap vers la direction de déplacement, lissé. À tester contre le cap fixe avec la même méthode.
Idée de LiDAR à plus longue portée : sans intérêt sur des pièces de 4-7 m (portée déjà 15 m), utile seulement avec de grands halls.

**Risques assumés** : (1) variance d'un run à l'autre -> toute affirmation dans le README sera sur >= 2 seeds ; (2) les cartes à
branches changent la nature du problème (mémoire), donc un modèle entraîné sur chaînes de pièces ne se transférera pas tel quel ;
(3) `max_speed=3` échoue de façon inexpliquée -> on garde 6 m/s.

## Revue de code externe (deux documents, model-level + code review) — pistes non testées, pour mémoire
 
Deux revues obtenues via un autre outil (pas testées empiriquement, à prioriser plus tard) :
 
**Reward** : le terme de potentiel `phi - last_phi` (delta typique 0,01-0,05) est noyé dans
`n_new_cells` (typique 10-80) — ratio ~200-1000×, donc le potentiel ne guide quasiment jamais la
navigation vers une frontière lointaine quand il n'y a plus de cellules proches à découvrir. Fix
suggéré : `n_new_cells + 30.0 * (phi - last_phi)`. Idem pour `r_exp=100` (bonus victoire) :
négligeable après actualisation PPO (`gamma=0,99`, jamais changé) sur un épisode de plusieurs
centaines de steps — quasi aucune incitation réelle à *finir* l'exploration plutôt que continuer à
scanner indéfiniment.
 
**Observation** : `kinematics` contient `vel_lin`/`vel_ang` en repère **monde**, pas égocentrique
— contrairement à `frontier_vector`/`relief_rays`/`proximity_rays` qui le sont déjà. Le réseau
doit apprendre la rotation entre ces deux repères. Fix simple : projeter `vel_lin` par rotation
`-yaw` avant de l'ajouter à l'observation. Padding des frontières manquantes trompeur
(`distance_norm=1.0` ressemble à une vraie frontière lointaine ; `0.0` serait plus neutre).
 
**Perf (pas des bugs de justesse, mais ralentit l'entraînement)** : Bresenham et flood-fill en
pur Python (`occupancy_grid.py`), raycasting séquentiel (198 appels `mj_ray`/step) — tous
vectorisables (`skimage.draw.line`, `scipy.ndimage.label`, batch raycasting MuJoCo).
 
**Fiabilité** : `evaluate.py` affiche le reward *normalisé/clippé* par VecNormalize dans
`total_reward`, pas le reward brut — comparer ce chiffre entre versions aux stats de
normalisation différentes ne veut rien dire (mais on ne s'en est jamais servi pour comparer les
versions, seulement victoire/couverture/collision rates — donc impact réel limité). `_in_corridor`
recalculé inutilement à chaque step dans `evaluate.py` (gaspillage, pas un bug). `info_gain`
calculé au centroïde du cluster de frontière plutôt que sur le cluster entier — pour une
frontière en L/U le centroïde peut tomber sur un mur. `RotatingCheckpointCallback` écrase toujours
le dernier checkpoint — en cas de régression tardive (catastrophic forgetting), pas de retour en
arrière possible (déjà vécu indirectement avec l'incident `--resume` sur v11).
 
**Confirmé bien fait par la revue** : la protection `env_kwargs` sur `--resume` (apprise à la dure
avec l'incident v11), les seeds fixes en évaluation, la séparation layout/XML/grille/env, le
masque de zones atteignables, `EpisodeMetricsCallback`, la pénalité de proximité restreinte aux
pièces.
 
## Outillage ajouté à `evaluate.py` / `train.py` / `explorer_env.py` / `model_manager.py` (pour mémoire)
 
- `explorer_env.py` : `nearest_frontier_dist` ajouté à l'`info` dict de `step()` (distance
  normalisée à la frontière valide la plus proche) — n'affecte pas l'espace d'observation,
  compatible avec tous les modèles déjà entraînés (v1 à v9).
- `evaluate.py` : colonne `frnt_d` dans `--debug-angles` ; option `--up-z-min` pour surcharger le
  seuil de retournement à l'évaluation SANS réentraîner ; détection et affichage "Collisions en
  couloir / en pièce" dans le résumé ; colonnes `speed`/`minLid`/`loc` en `--debug-angles` +
  résumé agrégé "minLid en couloir/pièce" (médiane, min).
- `train.py` : option `--seed` (init modèle + stochastique PPO), consignée dans `metadata.json`
  aux côtés de `env_kwargs` ; option `--damping` (amortissement du joint libre du drone) ;
  `--resume` corrigé pour relire `env_kwargs` depuis `metadata.json` au lieu des args CLI (cf.
  incident v11 ci-dessus) ; `env_kwargs` affiché au lancement dans les trois cas (resume,
  from-version, from scratch) ; options `--coeff-proximity`/`--proximity-threshold` (pénalité de
  proximité restreinte aux pièces, cf. section dédiée) ; `--coeff-speed` (pénalité de vitesse,
  abandonnée en v15) ; `--max-speed` (plafond dur de vitesse, v16, le levier qui a marché).
- `explorer_env.py` : `speed_horiz`/`min_lidar_dist` ajoutés à l'`info` dict ; `joint_damping`
  paramétrable (joint libre du drone) ; `coeff_proximity`/`proximity_threshold` + `_in_corridor()`
  (restreint la pénalité de proximité aux pièces) ; `proximity_rays` ajouté à l'observation
  (résumé LiDAR égocentrique, cf. section v14) ; `coeff_speed` (abandonné) ; `max_speed` (plafond
  physique dur sur `qvel`, appliqué à chaque sous-step, cf. section v16).
- `feature_extractor.py` : `proximity_rays` ajouté à `scalar_keys` (sinon présent dans
  l'observation mais jamais branché au réseau).
- `model_manager.py` : tri des versions corrigé (numérique au lieu d'alphabétique — cassait dès
  `v10`/`v11`, cf. section v11).
## Idées en réserve (une à la fois si le spin persiste)
 
- Version affinée de `coeff_spin` : pénaliser seulement la vitesse angulaire
  **soutenue** (moyenne glissante), pas instantanée — tolérer les rotations
  brèves nécessaires, punir seulement la rotation prolongée. À essayer si
  la version simple ne suffit pas ou casse la maniabilité normale.
- ~~Remettre l'amortissement en plus de `coeff_spin`~~ → testé en v11, succès net (43% victoire,
  meilleur résultat depuis le désaxage des couloirs, `v_ang` revenu dans la fourchette saine de
  v5). Cf. section v11.
- `coeff_align` (tangent path reward) — testé (v8 à 0.2, v9 à 0.1). Résultat positif sur v8
  (37% victoire, mais nouveau mode d'échec "retourné" à 10%, cf. section v8), pas confirmé tant
  que v10 (réplication avec seed fixé) n'a pas tranché si c'est le coefficient ou le tirage
  d'initialisation qui explique l'écart avec v6/v9 (comportement non-monotone entre 0.0/0.1/0.2,
  cf. section v9).
- Version affinée de `coeff_align` ou pénalité dédiée sur l'amplitude de roll/pitch, si v10
  confirme que 0.2 est un vrai effet et pas juste un tirage chanceux : viser le même bénéfice de
  redirection de vitesse sans pousser vers les bascules extrêmes qui causent les `retourné`.
- Mode `hover` (curriculum : apprendre la stabilité pure avant
  l'exploration complète). Mis en pause.
- Accéléromètre linéaire — pas jugé prioritaire, l'info la plus pertinente
  (vitesse angulaire + orientation) est déjà présente.
## Diagnostic (pas un bug) : couverture "trop facile" via ligne de vue en couloir droit
 
Outil créé : `debug_grid.py` — carte 2D de la grille + rayons LiDAR de la
dernière frame superposés + fenêtre du CNN encadrée. Vérifié visuellement
sur la seed 9002 : les cellules marquées libres suivent exactement l'éventail
des rayons tirés, pas un remplissage par zone — pas de triche dans le calcul
de couverture lui-même.
 
**Cause réelle identifiée** : tous les couloirs et toutes les portes sont
toujours centrés en x=0 par construction du générateur — donc chaque
bâtiment généré a ses portes parfaitement alignées sur un même axe. Un
rayon tiré droit dans cet axe voit à travers toutes les portes d'affilée,
jusqu'au fond de la dernière pièce, sans que le drone s'en approche. Ce
n'est pas un bug de raycasting, c'est un artefact du générateur de
bâtiments (portes jamais désaxées).
 
**Piste de fix identifiée (pas encore implémentée)** : désaxer légèrement
la position des portes/couloirs dans `building_generator.py` au lieu de les
centrer systématiquement, pour casser cette ligne de vue parfaitement
rectiligne. Complémentaire (pas contradictoire) avec l'idée d'agrandir les
bâtiments et de monter la barre de victoire, qui atténue l'effet sans
traiter la cause. À faire après `v4`, un seul changement à la fois.
 
## Règle de méthode (depuis la v2.1)
 
Un seul changement testé à la fois entre deux évaluations comparatives —
pour pouvoir attribuer un résultat à une cause précise.
 
## Addendum méthode (depuis la v9) : le seed d'entraînement n'était pas fixé
 
Jusqu'à v9, `train.py` ne fixait aucun seed pour l'initialisation du modèle ni la stochastique
PPO — seuls les seeds des environnements d'entraînement (0 à n_envs-1) étaient fixes. Deux runs
aux mêmes hyperparamètres pouvaient donc diverger uniquement à cause du tirage d'initialisation,
ce qui a probablement causé une partie du zigzag observé entre v6/v8/v9 (comportement non
monotone de `coeff_align`). Corrigé via `--seed`, consigné dans `metadata.json`.
 
Un seed fixé règle la reproductibilité sur une même machine, mais ne suffit pas à lui seul à
garantir une comparaison propre entre deux machines différentes (non-déterminisme cuDNN côté
GPU) : privilégier la même machine pour comparer deux versions tant que possible, et le noter
explicitement dans le journal quand ce n'est pas le cas.

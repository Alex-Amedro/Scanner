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
couple direct) — RÉSULTAT MITIGÉ, ouvre le chantier v20

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

## v20 — refonte groupée reward + observations pour le contrôle en vitesse (À LANCER)

**Changements groupés (assumé — pas à variable unique, cohérent avec la décision prise après
l'incident v11 : parfois plusieurs changements liés à une même cause doivent être testés ensemble)** :
- `kinematics` : `vel_lin` tourné en repère du corps avant observation.
- Clamp dur `qvel`/`max_speed` par sous-step **retiré** (redondant avec l'espace d'action).
- `coeff_spin` désactivé (0.0, défaut) pour ce test.
- `coeff_proximity` réactivé (0.3, seuil 0.6, restreint aux pièces).
- Nouvelle observation `last_action` (dernière consigne de vitesse lissée, 4 valeurs) — ajoutée à
  `observation_space`, `_get_obs()`, et `scalar_keys` dans `ExplorerFeaturesExtractor`.
- Lissage de l'action (`action_smoothing_alpha=0.3`, moyenne mobile exponentielle sur l'action brute
  PPO avant conversion en consigne de vitesse).
- Cascade PID scindée en `_outer_velocity_loop` (une fois par action RL, ~50Hz) et
  `_inner_attitude_rate_loop` (chaque sous-step, ~500Hz) — conforme à l'architecture standard,
  réduit aussi le coût de calcul.
- Poussée compensée par l'inclinaison (`thrust_ctrl` divisé par `up_z`, clampé à 0,5 min).

**Commande** :
```
python train.py --name explorer --damping 0.05 --death-penalty 500 --max-speed 6.0 --coeff-proximity 0.3 --proximity-threshold 0.6 --seed 42 --total-timesteps 1000000
```
(pas de `--coeff-spin` : retombe à 0.0 par défaut, voulu — point 3 ci-dessus)

**À comparer après coup** : contre v12 (47%/50%/3%/82,9%, 1M steps, ancien contrôle en couple) et
v16 (60%/37%/3%/80,7%, 500k steps) — pour trancher si le contrôle en vitesse égale ou dépasse
l'ancien contrôle en couple à budget comparable, et si le comportement "errant" de v19 a disparu
(`--debug-angles` sur quelques épisodes, `--max-steps 400` pour éviter des traces trop longues).
 
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

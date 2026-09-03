# Comparaison humaine CPT / assistant 17M

Les essais libres ont révélé un effondrement vers quelques réponses supervisées.
Avant tout nouvel entraînement, les questions sont figées et rejouées sur le
checkpoint préentraîné et le candidat assistant stage 3.

Le rapport mesure la diversité des sorties, la fréquence de la réponse dominante,
les réponses stéréotypées, la terminaison EOS et les correspondances exactes avec
les corrections humaines. La correspondance exacte reste un diagnostic : la
justesse finale nécessite une lecture humaine.

Les questions de ce benchmark ne doivent plus être utilisées pour entraîner le
17M. Elles deviennent un contrôle fermé de non-régression. Aucun split test
historique n'est ouvert par cette comparaison.

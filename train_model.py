import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

df =  pd.read_csv('../data/dataset_assurance_ML.csv')
if df.empty:
    print ("Date set vide ")
else:
    print("Donnée bien chargées")

df.shape
df.head()


TARGET = 'Résiliation'
print(df[TARGET].value_counts())

print ('en ourcentage') 
pourcentages = df[TARGET].value_counts(normalize=True) * 100
pourcentages.round(2)


print(pd.crosstab(df['Statut Contrat'], df[TARGET]))

# je remarque avec la variable `status contrat`  nous avons un `100%` en actif et `100%` en resilié 
# saut le dernier cas suspendu qui pose un proble 
# 
# `Nom` nous ne pouvons pas utilser  cette varible car déjà `etiquété` 


num_cols = [
    'Âge', 'Salaire Annuel (€)', 
    'Prime Annuelle (€)', 
    'Ancienneté (mois)',
    'Coeff. Bonus-Malus', 
    'Nb Sinistres (3 ans)',
    'Montant Sinistres (€)', 
    'Score Risque (0-100)'
    ]
cat_cols = ['Type Contrat', 
            'Catégorie Prof.', 
            'Usage Véhicule', 
            'Dernier Sinistre'
            ]

X = df[num_cols + cat_cols]
y = df[TARGET]
print(X.shape, y.shape)

# Etape 5:
# 
# En Machine Learning, calculer la corrélation d'une variable numérique avec la variable cible (Résiliation ou TARGET) permet de mesurer l'intensité du lien linéaire entre cette variable et le fait de résilier.
#     - \Une corrélation proche de $+1$ : Quand la variable augmente, le risque de résiliation augmente fortement.
#     - \Une corrélation proche de $-1$ : Quand la variable augmente, le risque de résiliation diminue fortement.
#     - \Une corrélation proche de $0$ : La variable n'a pas de lien linéaire évident avec la résiliation.


print(X[num_cols].corrwith(y).round(3).sort_values(ascending=False))
print("#############################""")
print(df.groupby('Dernier Sinistre')[TARGET].mean().round(2).sort_values())

# les trois variables sont: `'Résiliation', 'Statut Contrat', 'ID_Client'`


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42, stratify= y)
print(X_train.shape, X_test.shape)
print(y_train.mean().round(2), y_test.mean().round(2))


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
preprocessor = ColumnTransformer([
('num', StandardScaler(), num_cols),
('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
candidats = {
'Régression Logistique': LogisticRegression(
max_iter=1000, class_weight='balanced', random_state=42),
'Random Forest': RandomForestClassifier(
n_estimators=300, max_depth=4, min_samples_leaf=10,
class_weight='balanced', random_state=42),
}
pipelines = {nom: Pipeline([('prep', preprocessor), ('model', algo)])
for nom, algo in candidats.items()}


from sklearn.model_selection import cross_val_score
for nom, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    print(f'{nom:22s} AUC = {scores.mean():.3f} ± {scores.std():.3f}')

# Performance globale modérée (AUC $\approx$ 0,69) :Un score AUC de 0,5 correspond à un choix aléatoire (jet de pièce), tandis qu'un score de 1,0 représente un modèle parfait. Avec environ 0,69, vos deux modèles apprennent un signal utile, mais leurs performances restent moyennes.
# 
# 
# Avantage pour le Random Forest :Le Random Forest obtient une meilleure capacité de discrimination (AUC légèrement plus élevé à 0,696) et, surtout, une meilleure stabilité ($\pm$ 0,095 contre $\pm$ 0,133 pour la Régression Logistique). Une variance plus faible indique qu'il réagit de façon plus constante d'un pli de validation croisée à un autre.Variabilité importante ($\pm$ 0,095 à $\pm$ 0,133) :L'écart-type est relativement élevé. Cela s'explique généralement par la taille réduite de l'échantillon ou par le déséquilibre de la variable cible (environ 10 % de résiliations).


from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
confusion_matrix, classification_report)
pipeline = pipelines['Random Forest']
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print('Accuracy :', round(accuracy_score(y_test, y_pred),3))
print('F1:', round(f1_score(y_test, y_pred),3))
print('ROC-AUC :', round(roc_auc_score(y_test, y_proba),3))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Reste', 'Résilie']))

# Etape 12 sauvegarde du pipe ligne


import joblib, os
joblib.dump(pipeline, '../models/pipeline_resiliation.pkl')
print(os.path.getsize('../models/pipeline_resiliation.pkl') / 1024, 'Ko')

# Etape 13 Sauvegarde des eto Donée 


import json
meta = {
'modele': 'Random Forest',
'auc_test': round(float(roc_auc_score(y_test, y_proba)), 3),
'num_cols': num_cols,
'cat_cols': cat_cols,
'num_ranges': {c: {'min': float(X[c].min()), 'max': float(X[c].max()),
'median': float(X[c].median())} for c in num_cols},
'cat_values': {c: sorted(X[c].unique().tolist()) for c in cat_cols},
}
with open('../models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)


modele = joblib.load('../models/pipeline_resiliation.pkl')
client = pd.DataFrame([{
'Âge': 34, 'Salaire Annuel (€)': 28000, 'Prime Annuelle (€)': 950,
'Ancienneté (mois)': 6, 'Coeff. Bonus-Malus': 1.25, 'Nb Sinistres (3 ans)': 3,
'Montant Sinistres (€)': 4200, 'Score Risque (0-100)': 72,
'Type Contrat': 'Bronze', 'Catégorie Prof.': 'Entrepreneur',
'Usage Véhicule': 'Professionnel', 'Dernier Sinistre': 'Vol',
}])
print('Classe :', modele.predict(client))
print('Proba :', modele.predict_proba(client)[0, 1].round(3))

# Etape 15 erreur classique 


try:
    modele.predict(client.drop(columns=['Score Risque (0-100)']))
except Exception as e:
    print('ERREUR :', e)





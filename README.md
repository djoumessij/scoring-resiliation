☁️ Où déployer une application Streamlit ?
Une application Streamlit est un serveur Python qui tourne en permanence et garde une connexion ouverte
avec le navigateur (WebSocket). Elle a donc besoin d'un hébergeur capable de faire tourner un processus
Python long. Le choix naturel — gratuit, officiel, sans configuration — est Streamlit Community Cloud : on
connecte son compte GitHub, on désigne le dépôt, et l'application est en ligne en 2 minutes.

| Hébergeur | Streamlit ? | Gratuit ? | Remarque |
| :--- | :---: | :---: | :--- |
| **Streamlit Community Cloud** | ✅ Natif | Oui | Notre choix — [share.streamlit.io](https://share.streamlit.io) |
| **Hugging Face Spaces** | ✅ Natif | Oui | Bonne alternative, même principe |
| **Render / Railway** | ✅ Via Docker ou commande | Limité | Plus technique |
| **Vercel** | ❌ Non adapté | Oui | Voir encadré ci-dessous |


❓ Et Vercel ?
Vercel est conçu pour les sites statiques et les fonctions « serverless » : chaque requête démarre une
fonction, qui répond puis s'éteint en quelques secondes. Streamlit fonctionne à l'inverse (processus
permanent + WebSocket) : il ne peut pas tourner sur Vercel. Pour déployer sur Vercel, il faudrait
remplacer Streamlit par une API Flask ou FastAPI (une fonction /predict qui reçoit du JSON et renvoie la
probabilité) et écrire la page web à part. C'est une excellente suite pour aller plus loin — voir le Défi
bonus — mais ce n'est pas l'outil adapté à une première mise en ligne.
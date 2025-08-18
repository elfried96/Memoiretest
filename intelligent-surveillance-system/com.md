# Commandes utilisées pour l'installation et la gestion de l'environnement

```bash
# Installer less (si besoin)
apt update
apt install less

# Installer Node.js 18 et npm
apt remove --purge nodejs libnode-dev
apt autoremove
apt clean
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs
node -v
npm -v

# Installer le package Claude Code
npm install -g @anthropic-ai/claude-code

# Installer unzip (si besoin)
apt install unzip

# Dézipper le contenu dans le dossier videos
unzip '*.zip' -d videos/

# Installer le client Kaggle
pip install kaggle

# Placer le fichier d'API Kaggle
git clone https://github.com/username/intelligent-surveillance-system.git
mkdir -p ~/.kaggle
mv /chemin/vers/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Télécharger un dataset Kaggle
kaggle datasets download -d elfried/Testvideo

# Ajouter le binaire kaggle au PATH si besoin
export PATH="$PWD/.venv/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"
```

Ajoutez/modifiez selon vos besoins spécifiques.

FROM python:3.11-slim

# Build args pour UID/GID
ARG USER_UID=1000
ARG USER_GID=1000

# Créer user jehan avec UID/GID identique à l'hôte
RUN groupadd -g $USER_GID jehan \
    && useradd -m -u $USER_UID -g $USER_GID -s /bin/bash jehan

# Installer Python packages
WORKDIR /workspace
RUN pip install --upgrade pip
RUN pip install notebook jupyterlab pandas numpy matplotlib scipy scikit-image scikit-learn tensorflow

# Donner les droits sur le dossier de travail
RUN chown -R jehan:jehan /workspace

# Switch user pour que tout tourne avec jehan
USER jehan

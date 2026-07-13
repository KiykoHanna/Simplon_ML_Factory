# flow/Dockerfile
FROM prefecthq/prefect:3-python3.11

# On force la création du dossier
WORKDIR /app

# On installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie tout le contenu du dossier actuel dans /app
COPY . /app

# On s'assure que Python regarde dans /app
ENV PYTHONPATH=/app